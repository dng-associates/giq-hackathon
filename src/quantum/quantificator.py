import torch
import torch.nn as nn
import numpy as np
from math import comb
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


# ─────────────────────────────────────────────────────────────────────────────
# Enums & Config
# ─────────────────────────────────────────────────────────────────────────────


class EncodingType(str, Enum):
    ANGLE = "angle"
    AMPLITUDE = "amplitude"


@dataclass
class QuantificatorConfig:
    """
    Configuration for the Quantificator encoder.

    Parameters
    ----------
    encoding_type   : "angle" or "amplitude"
    n_modes         : number of photonic modes in the circuit
    n_photons       : number of photons (required for amplitude encoding)
    input_size      : number of classical features (required for angle encoding)
    encoding_scale  : scale factor applied to features before angle encoding
    trainable_depth : depth of superposition / trainable layers
    entangling_modes: pair of modes to apply MZI entangling layer (angle only)
    input_state     : Fock state list for angle encoding, e.g. [1,0,1,0]
    measurement     : "probs" | "amplitudes" | "modes"
    computation_space: "fock" | "unbunched" | "dual_rail"
    """

    encoding_type: EncodingType = EncodingType.ANGLE
    n_modes: int = 4
    n_photons: int = 2
    input_size: Optional[int] = None  # auto = n_modes for angle
    encoding_scale: float = np.pi
    trainable_depth: int = 2
    entangling_modes: Optional[List[int]] = None  # e.g. [0, 3]
    input_state: Optional[List[int]] = None  # e.g. [1,0,1,0]
    measurement: str = "probs"
    computation_space: str = "fock"

    def __post_init__(self):
        if self.input_size is None:
            self.input_size = self.n_modes
        if self.input_state is None:
            # default: alternate photons across modes
            state = [0] * self.n_modes
            placed = 0
            for i in range(self.n_modes):
                if placed < self.n_photons:
                    state[i] = 1
                    placed += 1
            self.input_state = state
        if self.entangling_modes is None:
            self.entangling_modes = [0, self.n_modes - 1]

    @property
    def fock_dim(self) -> int:
        return comb(self.n_modes + self.n_photons - 1, self.n_photons)


# ─────────────────────────────────────────────────────────────────────────────
# Quantificator
# ─────────────────────────────────────────────────────────────────────────────


class Quantificator(nn.Module):
    """
    A configurable quantum encoder that wraps a MerLin QuantumLayer
    and supports both Angle Encoding and Amplitude Encoding.

    Usage
    -----
    cfg   = QuantificatorConfig(encoding_type="angle", n_modes=4, n_photons=2)
    model = Quantificator(cfg)
    out   = model(x_tensor)          # angle: real tensor (B, features)
                                     # amplitude: complex tensor (B, fock_dim)
    """

    def __init__(self, config: QuantificatorConfig):
        super().__init__()
        self.config = config
        self._build_layer()

    # ── internal builders ────────────────────────────────────────────────────

    def _build_layer(self):
        """Construct the QuantumLayer according to the config."""
        try:
            from merlin.algorithms import QuantumLayer
            from merlin.builder import CircuitBuilder
            from merlin.core.state_vector import StateVector
            from merlin.measurement import MeasurementStrategy
            from merlin.core.computation_space import ComputationSpace

            cfg = self.config
            builder = CircuitBuilder(n_modes=cfg.n_modes)
            modes = list(range(cfg.n_modes))

            # shared: trainable rotations
            builder.add_rotations(modes=modes, trainable=True)

            if cfg.encoding_type == EncodingType.ANGLE:
                # angle-encoding stage
                builder.add_angle_encoding(
                    modes=modes,
                    name="input",
                    scale=cfg.encoding_scale,
                )
                # optional entangling layer
                if cfg.entangling_modes:
                    builder.add_entangling_layer(
                        modes=cfg.entangling_modes,
                        trainable=True,
                        model="mzi",
                    )

            # shared: superposition layers
            builder.add_superpositions(
                modes=modes,
                trainable=True,
                depth=cfg.trainable_depth,
            )

            # measurement strategy
            cs_map = {
                "fock": ComputationSpace.FOCK,
                "unbunched": ComputationSpace.UNBUNCHED,
                "dual_rail": ComputationSpace.DUAL_RAIL,
            }
            # Build measurement strategy safely — "modes" was removed in newer
            # versions of MerLin; fall back to "probs" if not available.
            _ms_candidates = {
                "probs": getattr(MeasurementStrategy, "probs", None),
                "amplitudes": getattr(MeasurementStrategy, "amplitudes", None),
                "modes": getattr(MeasurementStrategy, "modes", None),
            }
            cs = cs_map.get(cfg.computation_space, ComputationSpace.FOCK)
            ms_factory = (
                _ms_candidates.get(cfg.measurement) or MeasurementStrategy.probs
            )
            ms = ms_factory(cs)

            if cfg.encoding_type == EncodingType.ANGLE:
                self.layer = QuantumLayer(
                    input_size=cfg.input_size,
                    builder=builder,
                    input_state=StateVector.from_basic_state(cfg.input_state),
                    measurement_strategy=ms,
                )
            else:  # AMPLITUDE
                self.layer = QuantumLayer(
                    builder=builder,
                    n_photons=cfg.n_photons,
                    measurement_strategy=ms,
                )

            self._StateVector = StateVector
            self._merlin_available = True

        except (ImportError, AttributeError):
            # ── Simulation fallback (no MerLin installed) ──────────────────
            self._merlin_available = False
            self._build_simulated_layer()

    def _build_simulated_layer(self):
        """
        Pure-PyTorch simulation of the quantum encoder for environments
        without MerLin/Perceval installed.
        """
        cfg = self.config
        modes = cfg.n_modes

        if cfg.encoding_type == EncodingType.ANGLE:
            # Trainable phase parameters (θ per mode, per depth)
            self.theta = nn.ParameterList(
                [
                    nn.Parameter(torch.randn(modes))
                    for _ in range(cfg.trainable_depth + 1)
                ]
            )
        else:
            # Trainable unitary approximated by a real linear map on fock_dim
            self.theta = nn.ParameterList(
                [
                    nn.Parameter(torch.randn(cfg.fock_dim, cfg.fock_dim))
                    for _ in range(cfg.trainable_depth)
                ]
            )

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            - Angle encoding    : real tensor of shape (batch, input_size)
            - Amplitude encoding: complex tensor of shape (batch, fock_dim)

        Returns
        -------
        torch.Tensor : probability / amplitude output from the quantum layer
        """
        cfg = self.config
        self._validate_input(x)

        if self._merlin_available:
            return self._forward_merlin(x)
        else:
            return self._forward_simulated(x)

    def _forward_merlin(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        if cfg.encoding_type == EncodingType.ANGLE:
            return self.layer(x)
        else:
            sv = self._StateVector.from_tensor(
                x, n_modes=cfg.n_modes, n_photons=cfg.n_photons
            )
            return self.layer(sv)

    def _forward_simulated(self, x: torch.Tensor) -> torch.Tensor:
        """Lightweight simulation: applies phase rotations + linear mixing."""
        cfg = self.config

        if cfg.encoding_type == EncodingType.ANGLE:
            # encode features as phases, apply trainable rotations
            out = x  # (B, n_modes)
            for theta in self.theta:
                phases = torch.exp(1j * (out + theta).float())
                out = phases.abs()  # collapse to real probabilities
            return torch.softmax(out, dim=-1)

        else:  # AMPLITUDE
            # x is complex (B, fock_dim); apply trainable linear maps
            out = x.to(torch.complex64)
            for W in self.theta:
                W_c = W.to(torch.complex64)
                out = out @ W_c.T
                # re-normalize to keep valid quantum state
                norms = out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                out = out / norms
            return out.abs() ** 2  # return probabilities

    # ── validation ───────────────────────────────────────────────────────────

    def _validate_input(self, x: torch.Tensor):
        cfg = self.config
        if cfg.encoding_type == EncodingType.ANGLE:
            assert x.ndim == 2, "Angle encoding expects (batch, features)"
            assert x.shape[1] == cfg.input_size, (
                f"Expected {cfg.input_size} features, got {x.shape[1]}"
            )
            assert not x.is_complex(), "Angle encoding expects a real tensor"
        else:
            assert x.ndim == 2, "Amplitude encoding expects (batch, fock_dim)"
            assert x.shape[1] == cfg.fock_dim, (
                f"Expected fock_dim={cfg.fock_dim}, got {x.shape[1]}"
            )
            assert x.is_complex(), "Amplitude encoding expects a complex tensor"

    # ── helpers ──────────────────────────────────────────────────────────────

    @classmethod
    def angle(cls, n_modes=4, n_photons=2, **kwargs) -> "Quantificator":
        """Shortcut constructor for angle encoding."""
        cfg = QuantificatorConfig(
            encoding_type=EncodingType.ANGLE,
            n_modes=n_modes,
            n_photons=n_photons,
            **kwargs,
        )
        return cls(cfg)

    @classmethod
    def amplitude(cls, n_modes=4, n_photons=2, **kwargs) -> "Quantificator":
        """Shortcut constructor for amplitude encoding."""
        cfg = QuantificatorConfig(
            encoding_type=EncodingType.AMPLITUDE,
            n_modes=n_modes,
            n_photons=n_photons,
            **kwargs,
        )
        return cls(cfg)

    def __repr__(self) -> str:
        cfg = self.config
        backend = "MerLin" if self._merlin_available else "Simulated"
        return (
            f"Quantificator(\n"
            f"  encoding      = {cfg.encoding_type.value}\n"
            f"  n_modes       = {cfg.n_modes}\n"
            f"  n_photons     = {cfg.n_photons}\n"
            f"  fock_dim      = {cfg.fock_dim}\n"
            f"  input_size    = {cfg.input_size}\n"
            f"  scale         = {cfg.encoding_scale}\n"
            f"  depth         = {cfg.trainable_depth}\n"
            f"  measurement   = {cfg.measurement}\n"
            f"  backend       = {backend}\n"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    batch = 6

    # ── 1. Angle Encoding ────────────────────────────────────────────────────
    print("=" * 55)
    print("  ANGLE ENCODING")
    print("=" * 55)

    enc_angle = Quantificator.angle(n_modes=4, n_photons=2, trainable_depth=2)
    print(enc_angle)

    x_real = torch.rand(batch, 4)
    out_a = enc_angle(x_real)
    print(f"  Input  shape : {x_real.shape}")
    print(f"  Output shape : {out_a.shape}")
    print(f"  Output sample: {out_a[0].detach().numpy().round(4)}")

    # ── 2. Amplitude Encoding ────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("  AMPLITUDE ENCODING")
    print("=" * 55)

    enc_amp = Quantificator.amplitude(n_modes=4, n_photons=2, trainable_depth=2)
    print(enc_amp)

    fock_dim = enc_amp.config.fock_dim
    raw = torch.randn(batch, fock_dim) + 1j * torch.randn(batch, fock_dim)
    x_complex = raw / raw.norm(dim=-1, keepdim=True)
    out_b = enc_amp(x_complex)
    print(f"  Input  shape : {x_complex.shape}  (complex)")
    print(f"  Output shape : {out_b.shape}")
    print(f"  Output sample: {out_b[0].detach().numpy().round(4)}")
    print(f"  Sum of probs : {out_b[0].sum().item():.4f}  (should ≈ 1.0)")

    # ── 3. Custom config ─────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("  CUSTOM CONFIG (6 modes, 3 photons, amplitude)")
    print("=" * 55)

    cfg_custom = QuantificatorConfig(
        encoding_type=EncodingType.AMPLITUDE,
        n_modes=6,
        n_photons=3,
        trainable_depth=3,
        measurement="probs",
        computation_space="fock",
    )
    enc_custom = Quantificator(cfg_custom)
    print(enc_custom)

    fd = cfg_custom.fock_dim
    raw2 = torch.randn(batch, fd) + 1j * torch.randn(batch, fd)
    x_c2 = raw2 / raw2.norm(dim=-1, keepdim=True)
    out_c = enc_custom(x_c2)
    print(f"  Fock dim     : {fd}")
    print(f"  Output shape : {out_c.shape}")
