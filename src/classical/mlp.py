from __future__ import annotations

from typing import Iterable

try:
    import torch
    from torch import nn
except ModuleNotFoundError as err:
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = err


if nn is None:

    class MLP:
        """
        Placeholder template when PyTorch is not installed.
        """

        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError(
                "PyTorch is required to use MLP. Install torch first."
            ) from _TORCH_IMPORT_ERROR

        def forward(self, x):
            return x

else:

    class MLP(nn.Module):
        """
        Minimal MLP template for regression tasks.

        Example:
            model = MLP(input_dim=16, hidden_dims=(64, 32), output_dim=1)
            y_pred = model(x_batch)
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dims: Iterable[int] = (64, 32),
            output_dim: int = 1,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()

            dims = [input_dim, *list(hidden_dims), output_dim]
            layers: list[nn.Module] = []

            for idx in range(len(dims) - 1):
                in_dim = dims[idx]
                out_dim = dims[idx + 1]
                layers.append(nn.Linear(in_dim, out_dim))

                is_last = idx == len(dims) - 2
                if not is_last:
                    layers.append(nn.ReLU())
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))

            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)
