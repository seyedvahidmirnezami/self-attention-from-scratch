"""The same attention, as PyTorch modules you can train.

Works on batches: input shape (batch, n, d_model).
"""

import math

import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """Single-head scaled dot-product self-attention."""

    def __init__(self, d_model: int, d_k: int):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_k, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        Q, K, V = self.W_Q(x), self.W_K(x), self.W_V(x)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(K.size(-1))
        if mask is not None:  # mask: True where attention is allowed
            scores = scores.masked_fill(~mask, float("-inf"))
        A = scores.softmax(dim=-1)
        return A @ V, A


class MultiHeadAttention(nn.Module):
    """h heads of size d_model // h, concatenated and projected by W_O."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must divide evenly into heads"
        self.h = num_heads
        self.d_k = d_model // num_heads
        # One big projection per Q/K/V is the same as h small ones side by side.
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        b, n, _ = t.shape
        return t.view(b, n, self.h, self.d_k).transpose(1, 2)  # (b, h, n, d_k)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        b, n, d_model = x.shape
        Q, K, V = self._split(self.W_Q(x)), self._split(self.W_K(x)), self._split(self.W_V(x))
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # (b, h, n, n)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        A = scores.softmax(dim=-1)
        heads = A @ V  # (b, h, n, d_k)
        concat = heads.transpose(1, 2).reshape(b, n, d_model)
        return self.W_O(concat), A


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 5, 64)  # batch of 2 sentences, 5 tokens, 64-dim embeddings
    mha = MultiHeadAttention(d_model=64, num_heads=8)
    out, A = mha(x)
    print("input  ", tuple(x.shape))
    print("output ", tuple(out.shape), "(same shape, so blocks can stack)")
    print("weights", tuple(A.shape), "(batch, heads, n, n)")
