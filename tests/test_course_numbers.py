"""Check the code against the hand-worked numbers from the course."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from self_attention import (  # noqa: E402
    X_EXAMPLE,
    multi_head_attention,
    self_attention,
    simple_self_attention,
)
from self_attention_torch import MultiHeadAttention, SelfAttention  # noqa: E402

TOL = dict(atol=1e-3)


def test_simple_self_attention_matches_session_2():
    Y, A = simple_self_attention(X_EXAMPLE)
    np.testing.assert_allclose(
        A,
        [[0.665, 0.090, 0.245], [0.090, 0.665, 0.245], [0.212, 0.212, 0.576]],
        **TOL,
    )
    np.testing.assert_allclose(
        Y,
        [[0.910, 0.335, 0.665, 0.090], [0.335, 0.910, 0.090, 0.665], [0.788, 0.788, 0.212, 0.212]],
        **TOL,
    )


W_Q = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=float)
W_K = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
W_V = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=float)


def test_qkv_attention_matches_session_3():
    Y, A = self_attention(X_EXAMPLE, W_Q, W_K, W_V)
    np.testing.assert_allclose(
        A,
        [[0.707, 0.070, 0.223], [0.333, 0.333, 0.333], [0.533, 0.168, 0.299]],
        **TOL,
    )
    np.testing.assert_allclose(
        Y,
        [[1.000, 0.707, 0.363], [1.000, 0.333, 1.000], [1.000, 0.533, 0.635]],
        **TOL,
    )


def test_multi_head_matches_session_4():
    head1 = (
        np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float),
        np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=float),
        np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=float),
    )
    head2 = (
        np.array([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=float),
        np.array([[1, 1], [0, 0], [1, 0], [0, 1]], dtype=float),
        np.array([[0, 1], [1, 0], [1, 1], [0, 0]], dtype=float),
    )
    out, (A1, A2) = multi_head_attention(X_EXAMPLE, [head1, head2], np.eye(4))
    np.testing.assert_allclose(A1[0], [0.045, 0.768, 0.187], **TOL)
    np.testing.assert_allclose(A2[0], [0.576, 0.140, 0.284], **TOL)
    np.testing.assert_allclose(out, [[1, 1, 1.000, 1.436]] * 3, **TOL)


def test_torch_single_head_matches_numpy():
    attn = SelfAttention(d_model=4, d_k=3)
    with torch.no_grad():  # nn.Linear stores weights transposed
        attn.W_Q.weight.copy_(torch.tensor(W_Q.T))
        attn.W_K.weight.copy_(torch.tensor(W_K.T))
        attn.W_V.weight.copy_(torch.tensor(W_V.T))
    Y, _ = attn(torch.tensor(X_EXAMPLE, dtype=torch.float32).unsqueeze(0))
    expected, _ = self_attention(X_EXAMPLE, W_Q, W_K, W_V)
    np.testing.assert_allclose(Y[0].detach().numpy(), expected, atol=1e-5)


def test_multi_head_keeps_shape_and_rows_sum_to_one():
    mha = MultiHeadAttention(d_model=16, num_heads=4)
    x = torch.randn(3, 7, 16)
    out, A = mha(x)
    assert out.shape == x.shape
    assert A.shape == (3, 4, 7, 7)
    torch.testing.assert_close(A.sum(-1), torch.ones(3, 4, 7))
