import math
from typing import Dict
import torch
import numpy as np
import einops


class Linear(torch.nn.Module):
    """Linear layout"""

    def __init__(
        self,
        in_features_dim: int,
        out_features_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Construct a linear transformation module. This function should accept the following parameters"""
        super().__init__()
        std = 2.0 / (in_features_dim + out_features_dim)
        self.weight = torch.nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(out_features_dim, in_features_dim),
                0,
                std,
                -3 * np.sqrt(std),
                3 * np.sqrt(std),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to the input."""
        return x @ self.weight.T


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
        self.d_ff = np.floor(d_mdodel * int(8 / 3)) if d_ff == None else d_ff
        self.w1 = Linear(d_mdodel, d_ff)
        self.w2 = Linear(d_ff, d_mdodel)
        self.w3 = Linear(d_mdodel, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2.forward((self._silu(self.w1.forward(x)) * self.w3.forward(x)))


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        matrix_list = []
        for i in range(0, max_seq_len):
            blocks = []
            for k in range(1, d_k // 2 + 1):
                angle = i / np.power(theta, (2 * k - 2) / d_k)
                block = torch.tensor(
                    [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
                    device=device,
                    dtype=torch.float32,
                )
                blocks.append(block)

            r_i = torch.block_diag(*blocks)
            matrix_list.append(r_i)

        self.register_buffer("rope_matrix", torch.stack(matrix_list), False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        return einops.einsum(
            x, self.rope_matrix[token_positions], "... s d, ... s d1 d -> ... s d1"
        )


def softmax(x: torch.Tensor, i: int):
    """
    softmax operator

    Args:
        x : input_feature
        i : softmax dimension
    """

    x_max = x.max(dim=i, keepdim=True).values
    x_exp = (x - x_max).exp()
    return x_exp / x_exp.sum(dim=i, keepdim=True)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
):
    atten_score = (query @ key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
    if mask is not None:
        atten_score = atten_score.masked_fill(~mask, float("-inf"))
    return softmax(atten_score, -1) @ value


class multihead_self_attention(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        enable_rope: bool = False,
        theta=None,
        max_seq_len=None,
        device=None,
    ):
        super().__init__()
        self.Q = Linear(d_model, d_model)
        self.K = Linear(d_model, d_model)
        self.V = Linear(d_model, d_model)
        self.O = Linear(d_model, d_model)
        self.enable_rope = enable_rope
        if enable_rope:
            self.rope = RotaryPositionalEmbedding(
                theta, d_model // num_heads, max_seq_len, device=device
            )
        self.num_heads = num_heads

    def forward(self, in_features: torch.Tensor, token_positions=None):
        B, S, _ = in_features.shape
        Q = self.Q.forward(in_features)
        K = self.K.forward(in_features)
        V = self.V.forward(in_features)

        Q_chunks = Q.chunk(self.num_heads, -1)
        K_chunks = K.chunk(self.num_heads, -1)
        V_chunks = V.chunk(self.num_heads, -1)
        mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=in_features.device))
        return self.O.forward(
            torch.concat(
                [
                    scaled_dot_product_attention(
                        (
                            self.rope.forward(Q_chunks[i], token_positions)
                            if self.enable_rope
                            else Q_chunks[i]
                        ),
                        (
                            self.rope.forward(K_chunks[i], token_positions)
                            if self.enable_rope
                            else K_chunks[i]
                        ),
                        V_chunks[i],
                        mask,
                    )
                    for i in range(self.num_heads)
                ],
                -1,
            )
        )


class PreNormTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        enable_rope: bool = False,
        theta=None,
        max_seq_len=None,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.mha = multihead_self_attention(
            d_model, num_heads, enable_rope, theta, max_seq_len
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor):
        token_positions = torch.arange(0, x.shape[1], 1)
        x = x + self.mha(self.norm1(x), token_positions)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        enable_rope: bool = False,
        theta=None,
    ):
        super().__init__()
        self.embd = Embedding(vocab_size, d_model)
        self.trans_blocks = torch.nn.ModuleList(
            [
                PreNormTransformerBlock(
                    d_model, num_heads, d_ff, enable_rope, theta, context_length
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_size)

    def forward(
        self,
        in_features: torch.Tensor,
    ):
        x = self.embd(in_features)
        for block in self.trans_blocks:
            x = block(x)
        x = self.norm(x)
        x = self.linear(x)
        return x


def cross_entropy(logits: torch.Tensor, tragets: torch.Tensor):
    max_logit = logits.max(dim=-1, keepdim=True).values
    logits_shifted = logits - max_logit
    sum_logits = logits_shifted.exp().sum(dim=-1, keepdim=True)
    res = (
        -((logits_shifted.gather(-1, tragets.unsqueeze(-1)) - sum_logits.log()))
    ).mean()
    return res


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        hparams = {"lr": lr}
        super().__init__(params, hparams)

    def step(self):

        pass
