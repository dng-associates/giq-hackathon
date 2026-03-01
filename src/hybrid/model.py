from __future__ import annotations

from typing import Literal

try:
    import torch
    from torch import nn
except ModuleNotFoundError as err:
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = err

from src.quantum.quantificator import EncodingType, Quantificator, QuantificatorConfig


if nn is None:

    class MerlinHybridRegressor:
        """Placeholder template when PyTorch is not installed."""

        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "PyTorch is required to use MerlinHybridRegressor. Install torch first."
            ) from _TORCH_IMPORT_ERROR

        def forward(self, x):
            return x

else:

    class MerlinHybridRegressor(nn.Module):
        """
        Hybrid regressor: classical projection -> quantum encoder -> classical head.
        """

        def __init__(
            self,
            input_dim: int,
            *,
            n_modes: int = 4,
            n_photons: int = 2,
            trainable_depth: int = 2,
            measurement: str = "probs",
            encoding_type: Literal["angle", "amplitude"] = "angle",
        ) -> None:
            super().__init__()

            encoding = EncodingType(encoding_type)
            self.encoding_type = encoding

            if encoding == EncodingType.ANGLE:
                projected_dim = n_modes
                self.projector = nn.Sequential(
                    nn.Linear(input_dim, projected_dim),
                    nn.Tanh(),
                )
                q_input_size = projected_dim
            else:
                cfg_tmp = QuantificatorConfig(
                    encoding_type=EncodingType.AMPLITUDE,
                    n_modes=n_modes,
                    n_photons=n_photons,
                    trainable_depth=trainable_depth,
                    measurement=measurement,
                )
                projected_dim = cfg_tmp.fock_dim * 2
                self.projector = nn.Sequential(
                    nn.Linear(input_dim, projected_dim),
                    nn.Tanh(),
                )
                q_input_size = cfg_tmp.fock_dim

            self.qcfg = QuantificatorConfig(
                encoding_type=encoding,
                n_modes=n_modes,
                n_photons=n_photons,
                input_size=q_input_size if encoding == EncodingType.ANGLE else None,
                trainable_depth=trainable_depth,
                measurement=measurement,
            )
            self.quantum = Quantificator(self.qcfg)

            # LazyLinear lets the model adapt to different quantum output shapes.
            self.head = nn.Sequential(
                nn.LazyLinear(32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            z = self.projector(x)

            if self.encoding_type == EncodingType.AMPLITUDE:
                fock_dim = self.qcfg.fock_dim
                real_part, imag_part = z[:, :fock_dim], z[:, fock_dim:]
                amp = torch.complex(real_part, imag_part)
                norm = amp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                q_input = amp / norm
            else:
                q_input = z

            q_out = self.quantum(q_input)
            q_out = q_out.reshape(q_out.shape[0], -1).float()
            return self.head(q_out)
