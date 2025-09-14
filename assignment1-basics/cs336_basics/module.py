import torch
import numpy as np


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


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Construct an embedding module. This function should accept the following parameters:"""
        super().__init__()
        self.M = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(num_embeddings, embedding_dim), 0, 1, -3, 3
            )
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup the embedding vectors for the given token IDs."""
        return self.M[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.ones(d_model, dtype=dtype, device=device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape."""
        in_type = x.dtype
        x = x.to(torch.float32)
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        return (x / rms * self.weight).to(in_type)


class SwiGLU(torch.nn.Module):

    @classmethod
    def _silu(cls, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def __init__(self, d_mdodel: int, d_ff=None, device=None, dtype=None):
        super().__init__()
        self.d_ff = np.floor(d_mdodel * 8 / 3) if d_ff == None else d_ff
        self.w1 = Linear(d_mdodel, d_ff)
        self.w2 = Linear(d_ff, d_mdodel)
        self.w3 = Linear(d_mdodel, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2.forward((self._silu(self.w1.forward(x)) * self.w3.forward(x)))
