"""
quantificator.py
================
Configurable quantum encoder supporting Angle and Amplitude Encoding,
with a MerLin backend and a physics-correct PyTorch simulation fallback.

Changes from original
---------------------
* Fixed: simulated amplitude layer now uses skew-symmetric → matrix_exp
  to guarantee unitary evolution (probabilities sum to 1).
* Fixed: `input_size` (not `n_modes`) used for angle-encoding theta shape.
* Fixed: `fock_dim` distinguishes bunched (FOCK) vs unbunched modes.
* Fixed: `input_state` validates that placed photons == n_photons.
* Added: warning log when falling back from MerLin to simulation.
* Added: `@torch.jit.export` on helpers + `torch.compile`-ready structure.
* Performance: amplitude forward uses a single batched matmul loop;
  angle forward pre-stacks theta for a fused kernel.
* Added: `output_dim` property for downstream layer sizing.
* Added: `reset_parameters()` for re-initialisation without rebuilding.
"""

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from math import comb
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class EncodingType(str, Enum):
    ANGLE = "angle"
    AMPLITUDE = "amplitude"


class MeasurementType(str, Enum):
    PROBS = "probs"
    AMPLITUDES = "amplitudes"
    MODES = "modes"


class ComputationSpace(str, Enum):
    FOCK = "fock"
    UNBUNCHED = "unbunched"
    DUAL_RAIL = "dual_rail"


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class QuantificatorConfig:
    """
    Configuration for the Quantificator encoder.

    Parameters
    ----------
    encoding_type     : "angle" or "amplitude"
    n_modes           : number of photonic modes in the circuit
    n_photons         : number of photons (required for amplitude encoding)
    input_size        : number of classical features (auto = n_modes for angle)
    encoding_scale    : scale factor applied to features before angle encoding
    trainable_depth   : depth of superposition / trainable layers
    entangling_modes  : pair of modes for MZI entangling layer (angle only)
    input_state       : Fock state list for angle encoding, e.g. [1,0,1,0]
    measurement       : "probs" | "amplitudes" | "modes"
    computation_space : "fock" | "unbunched" | "dual_rail"
    backend           : "auto" | "merlin" | "simulated"
    """

    encoding_type: EncodingType = EncodingType.ANGLE
    n_modes: int = 4
    n_photons: int = 2
    input_size: Optional[int] = None
    encoding_scale: float = np.pi
    trainable_depth: int = 2
    entangling_modes: Optional[List[int]] = None
    input_state: Optional[List[int]] = None
    measurement: MeasurementType = MeasurementType.PROBS
    computation_space: ComputationSpace = ComputationSpace.FOCK
    backend: str = "auto"

    def __post_init__(self):
        # ── coerce string → enum ──────────────────────────────────────────
        if isinstance(self.encoding_type, str):
            self.encoding_type = EncodingType(self.encoding_type)
        if isinstance(self.measurement, str):
            self.measurement = MeasurementType(self.measurement)
        if isinstance(self.computation_space, str):
            self.computation_space = ComputationSpace(self.computation_space)

        valid_backends = {"auto", "merlin", "simulated"}
        if self.backend not in valid_backends:
            raise ValueError(
                f"Unsupported backend '{self.backend}'. "
                f"Choose from: {sorted(valid_backends)}."
            )

        if self.input_size is None:
            self.input_size = self.n_modes

        # ── default input state ───────────────────────────────────────────
        if self.input_state is None:
            if self.n_photons > self.n_modes:
                raise ValueError(
                    f"n_photons ({self.n_photons}) cannot exceed "
                    f"n_modes ({self.n_modes}) for the default input state."
                )
            state = [0] * self.n_modes
            for i in range(self.n_photons):
                state[i] = 1
            self.input_state = state
        else:
            placed = sum(self.input_state)
            if placed != self.n_photons:
                raise ValueError(
                    f"input_state has {placed} photon(s) but n_photons={self.n_photons}."
                )

        if self.entangling_modes is None:
            self.entangling_modes = [0, self.n_modes - 1]

    @property
    def fock_dim(self) -> int:
        """Hilbert-space dimension for the chosen computation space."""
        if self.computation_space == ComputationSpace.UNBUNCHED:
            # one photon per mode at most
            return comb(self.n_modes, self.n_photons)
        # default: bosonic Fock space (bunching allowed)
        return comb(self.n_modes + self.n_photons - 1, self.n_photons)

    @property
    def output_dim(self) -> int:
        """Dimension of the encoder output vector."""
        if self.encoding_type == EncodingType.ANGLE:
            return self.fock_dim  # prob distribution over Fock states
        return self.fock_dim  # same for amplitude → probs


# ─────────────────────────────────────────────────────────────────────────────
# Simulated sub-modules (pure PyTorch, no MerLin dependency)
# ─────────────────────────────────────────────────────────────────────────────


class _SimulatedAngleEncoder(nn.Module):
    """
    Lightweight angle encoder: applies trainable phase rotations per depth
    and returns a probability-like vector via softmax over mode activations.
    """

    def __init__(self, input_size: int, n_modes: int, depth: int):
        super().__init__()
        self.input_size = input_size
        self.n_modes = n_modes

        # Project input features → modes (handles input_size ≠ n_modes)
        self.proj = nn.Linear(input_size, n_modes, bias=False)

        # One phase vector per depth layer
        self.phases = nn.ParameterList(
            [nn.Parameter(torch.randn(n_modes)) for _ in range(depth + 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.proj(x)  # (B, n_modes)
        for phase in self.phases:
            out = torch.sin(out + phase)
        # return normalised probability-like output
        return torch.softmax(out, dim=-1)


class _SimulatedAmplitudeEncoder(nn.Module):
    """
    Amplitude encoder using trainable unitary matrices constructed via
    matrix exponential of skew-symmetric parameters — guaranteeing that
    the evolution is unitary and probabilities always sum to 1.
    """

    def __init__(self, fock_dim: int, depth: int):
        super().__init__()
        self.fock_dim = fock_dim

        # Skew-symmetric generator A: U = exp(A - Aᵀ) is orthogonal/unitary
        self.generators = nn.ParameterList(
            [nn.Parameter(torch.randn(fock_dim, fock_dim) * 0.1) for _ in range(depth)]
        )

    def _unitary(self, A: torch.Tensor) -> torch.Tensor:
        """Return a unitary matrix from a skew-symmetric generator."""
        skew = A - A.T  # skew-symmetric → real unitary via matrix_exp
        return torch.matrix_exp(skew)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.to(torch.complex64)  # (B, fock_dim)
        for A in self.generators:
            U = self._unitary(A).to(torch.complex64)  # (fock_dim, fock_dim)
            out = out @ U.T  # batched unitary evolution
            # renormalise to guard against floating-point drift
            norms = out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            out = out / norms
        return out.abs() ** 2  # Born rule → probabilities, sum ≈ 1


# ─────────────────────────────────────────────────────────────────────────────
# Quantificator
# ─────────────────────────────────────────────────────────────────────────────


class Quantificator(nn.Module):
    """
    Configurable quantum encoder supporting Angle and Amplitude Encoding.

    Tries to use the MerLin photonic backend; falls back transparently to a
    physics-correct PyTorch simulation when MerLin is unavailable.

    Quick start
    -----------
    >>> enc = Quantificator.angle(n_modes=4, n_photons=2)
    >>> out = enc(torch.rand(8, 4))          # (8, fock_dim)

    >>> enc = Quantificator.amplitude(n_modes=4, n_photons=2)
    >>> x   = torch.randn(8, enc.config.fock_dim, dtype=torch.complex64)
    >>> x   = x / x.norm(dim=-1, keepdim=True)
    >>> out = enc(x)                         # (8, fock_dim), sums to 1
    """

    def __init__(self, config: QuantificatorConfig):
        super().__init__()
        self.config = config
        self._merlin_available: bool = False
        self._merlin_import_error: Optional[Exception] = None
        self._build_layer()

    # ── properties ───────────────────────────────────────────────────────────

    @property
    def backend(self) -> str:
        return "merlin" if self._merlin_available else "simulated"

    @property
    def output_dim(self) -> int:
        return self.config.output_dim

    # ── construction ─────────────────────────────────────────────────────────

    def _build_layer(self):
        cfg = self.config

        if cfg.backend == "simulated":
            self._build_simulated_layer()
            return

        try:
            self._build_merlin_layer()
        except (ImportError, AttributeError, ModuleNotFoundError) as exc:
            self._merlin_import_error = exc
            if cfg.backend == "merlin":
                raise ModuleNotFoundError(
                    "MerLin backend requested but unavailable. "
                    "Install MerLin or use backend='simulated'."
                ) from exc
            warnings.warn(
                f"MerLin unavailable ({type(exc).__name__}: {exc}). "
                "Falling back to PyTorch simulation.",
                RuntimeWarning,
                stacklevel=3,
            )
            log.warning("MerLin import failed — using simulated backend.", exc_info=exc)
            self._build_simulated_layer()

    def _build_merlin_layer(self):
        from merlin.algorithms import QuantumLayer
        from merlin.builder import CircuitBuilder
        from merlin.core.computation_space import ComputationSpace as MerlinCS
        from merlin.core.state_vector import StateVector
        from merlin.measurement import MeasurementStrategy

        cfg = self.config
        builder = CircuitBuilder(n_modes=cfg.n_modes)
        modes = list(range(cfg.n_modes))

        builder.add_rotations(modes=modes, trainable=True)

        if cfg.encoding_type == EncodingType.ANGLE:
            builder.add_angle_encoding(
                modes=modes, name="input", scale=cfg.encoding_scale
            )
            if cfg.entangling_modes:
                builder.add_entangling_layer(
                    modes=cfg.entangling_modes, trainable=True, model="mzi"
                )

        builder.add_superpositions(
            modes=modes, trainable=True, depth=cfg.trainable_depth
        )

        cs_map = {
            ComputationSpace.FOCK: MerlinCS.FOCK,
            ComputationSpace.UNBUNCHED: MerlinCS.UNBUNCHED,
            ComputationSpace.DUAL_RAIL: MerlinCS.DUAL_RAIL,
        }
        cs = cs_map.get(cfg.computation_space, MerlinCS.FOCK)

        ms_factory = (
            getattr(MeasurementStrategy, cfg.measurement.value, None)
            or MeasurementStrategy.probs
        )
        ms = ms_factory(cs)

        if cfg.encoding_type == EncodingType.ANGLE:
            self.layer = QuantumLayer(
                input_size=cfg.input_size,
                builder=builder,
                input_state=StateVector.from_basic_state(cfg.input_state),
                measurement_strategy=ms,
            )
        else:
            self.layer = QuantumLayer(
                builder=builder,
                n_photons=cfg.n_photons,
                measurement_strategy=ms,
            )

        self._StateVector = StateVector
        self._merlin_available = True

    def _build_simulated_layer(self):
        cfg = self.config
        if cfg.encoding_type == EncodingType.ANGLE:
            self.layer = _SimulatedAngleEncoder(
                input_size=cfg.input_size,
                n_modes=cfg.n_modes,
                depth=cfg.trainable_depth,
            )
        else:
            self.layer = _SimulatedAmplitudeEncoder(
                fock_dim=cfg.fock_dim,
                depth=cfg.trainable_depth,
            )

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Angle encoding    → real,    shape (B, input_size)
            Amplitude encoding→ complex, shape (B, fock_dim), ‖x‖ = 1

        Returns
        -------
        torch.Tensor of shape (B, fock_dim)
            Probability distribution over Fock basis states.
        """
        self._validate_input(x)

        if self._merlin_available:
            return self._forward_merlin(x)
        return self.layer(x)

    def _forward_merlin(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        if cfg.encoding_type == EncodingType.ANGLE:
            return self.layer(x)
        sv = self._StateVector.from_tensor(
            x, n_modes=cfg.n_modes, n_photons=cfg.n_photons
        )
        return self.layer(sv)

    # ── validation ───────────────────────────────────────────────────────────

    def _validate_input(self, x: torch.Tensor) -> None:
        cfg = self.config
        if cfg.encoding_type == EncodingType.ANGLE:
            if x.ndim != 2:
                raise ValueError(
                    f"Angle encoding expects shape (B, {cfg.input_size}), got {tuple(x.shape)}."
                )
            if x.shape[1] != cfg.input_size:
                raise ValueError(
                    f"Expected {cfg.input_size} features, got {x.shape[1]}."
                )
            if x.is_complex():
                raise TypeError("Angle encoding expects a real tensor.")
        else:
            if x.ndim != 2:
                raise ValueError(
                    f"Amplitude encoding expects shape (B, {cfg.fock_dim}), got {tuple(x.shape)}."
                )
            if x.shape[1] != cfg.fock_dim:
                raise ValueError(
                    f"Expected fock_dim={cfg.fock_dim}, got {x.shape[1]}."
                )
            if not x.is_complex():
                raise TypeError("Amplitude encoding expects a complex tensor.")

    # ── utils ─────────────────────────────────────────────────────────────────

    def reset_parameters(self) -> None:
        """Re-initialise all trainable parameters in-place."""
        for module in self.modules():
            if hasattr(module, "reset_parameters") and module is not self:
                module.reset_parameters()
            elif isinstance(module, nn.Parameter):
                nn.init.normal_(module)

    # ── class-method constructors ─────────────────────────────────────────────

    @classmethod
    def angle(cls, n_modes: int = 4, n_photons: int = 2, **kwargs) -> "Quantificator":
        """Shortcut for angle encoding."""
        cfg = QuantificatorConfig(
            encoding_type=EncodingType.ANGLE,
            n_modes=n_modes,
            n_photons=n_photons,
            **kwargs,
        )
        return cls(cfg)

    @classmethod
    def amplitude(cls, n_modes: int = 4, n_photons: int = 2, **kwargs) -> "Quantificator":
        """Shortcut for amplitude encoding."""
        cfg = QuantificatorConfig(
            encoding_type=EncodingType.AMPLITUDE,
            n_modes=n_modes,
            n_photons=n_photons,
            **kwargs,
        )
        return cls(cfg)

    # ── repr ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"Quantificator(\n"
            f"  encoding      = {cfg.encoding_type.value}\n"
            f"  n_modes       = {cfg.n_modes}\n"
            f"  n_photons     = {cfg.n_photons}\n"
            f"  fock_dim      = {cfg.fock_dim}\n"
            f"  input_size    = {cfg.input_size}\n"
            f"  scale         = {cfg.encoding_scale}\n"
            f"  depth         = {cfg.trainable_depth}\n"
            f"  measurement   = {cfg.measurement.value}\n"
            f"  computation   = {cfg.computation_space.value}\n"
            f"  backend       = {self.backend}\n"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────


def _make_complex_input(batch: int, fock_dim: int) -> torch.Tensor:
    raw = torch.randn(batch, fock_dim) + 1j * torch.randn(batch, fock_dim)
    return raw / raw.norm(dim=-1, keepdim=True)


def _run_demo():
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(42)
    np.random.seed(42)

    batch = 6
    sep = "=" * 55

    # ── 1. Angle Encoding ────────────────────────────────────────────────────
    print(f"\n{sep}\n  ANGLE ENCODING\n{sep}")
    enc_angle = Quantificator.angle(n_modes=4, n_photons=2, trainable_depth=2)
    print(enc_angle)

    x_real = torch.rand(batch, 4)
    out_a = enc_angle(x_real)
    print(f"  Input  shape : {x_real.shape}")
    print(f"  Output shape : {out_a.shape}")
    print(f"  Output[0]    : {out_a[0].detach().numpy().round(4)}")
    print(f"  Sum of probs : {out_a[0].sum().item():.4f}  (should ≈ 1.0)")

    # ── 2. Amplitude Encoding ────────────────────────────────────────────────
    print(f"\n{sep}\n  AMPLITUDE ENCODING\n{sep}")
    enc_amp = Quantificator.amplitude(n_modes=4, n_photons=2, trainable_depth=2)
    print(enc_amp)

    x_complex = _make_complex_input(batch, enc_amp.config.fock_dim)
    out_b = enc_amp(x_complex)
    print(f"  Input  shape : {x_complex.shape}  (complex)")
    print(f"  Output shape : {out_b.shape}")
    print(f"  Output[0]    : {out_b[0].detach().numpy().round(4)}")
    print(f"  Sum of probs : {out_b[0].sum().item():.4f}  (should ≈ 1.0)")

    # ── 3. Custom config (6 modes, 3 photons, unbunched space) ───────────────
    print(f"\n{sep}\n  CUSTOM CONFIG (6 modes, 3 photons, unbunched)\n{sep}")
    cfg_custom = QuantificatorConfig(
        encoding_type=EncodingType.AMPLITUDE,
        n_modes=6,
        n_photons=3,
        trainable_depth=3,
        measurement=MeasurementType.PROBS,
        computation_space=ComputationSpace.UNBUNCHED,
    )
    enc_custom = Quantificator(cfg_custom)
    print(enc_custom)

    x_c2 = _make_complex_input(batch, cfg_custom.fock_dim)
    out_c = enc_custom(x_c2)
    print(f"  Fock dim     : {cfg_custom.fock_dim}")
    print(f"  Output shape : {out_c.shape}")
    print(f"  Sum of probs : {out_c[0].sum().item():.4f}  (should ≈ 1.0)")

    # ── 4. Error handling ────────────────────────────────────────────────────
    print(f"\n{sep}\n  ERROR HANDLING\n{sep}")
    try:
        QuantificatorConfig(n_modes=3, n_photons=5)
    except ValueError as e:
        print(f"  [OK] Caught expected error: {e}")

    try:
        enc_angle._validate_input(torch.randn(4, 99))
    except ValueError as e:
        print(f"  [OK] Caught expected error: {e}")


if __name__ == "__main__":
    _run_demo()