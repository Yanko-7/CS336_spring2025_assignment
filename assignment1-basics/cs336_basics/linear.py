import torch
import numpy as np
import einops


class Linear(torch.nn.Module):
    """Linear layout"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Construct a linear transformation module. This function should accept the following parameters"""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        std = 2.0 / (in_features + out_features)
        self.M = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(out_features, in_features),
                0,
                std,
                -3 * np.sqrt(std),
                3 * np.sqrt(std),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to the input."""
        return x @ self.M.T
