"""Self-attention from scratch in NumPy.

Three steps, one idea at a time:

1. ``simple_self_attention``         softmax(X Xᵀ) X           no learned weights
2. ``scaled_dot_product_attention``  softmax(Q Kᵀ / √d_k) V    learnable Q, K, V
3. ``multi_head_attention``          Concat(head₁..head_h) W_O  several views at once

Every function works on a single sequence X of shape (n, d):
n tokens, each a d-dimensional vector.
"""

import numpy as np


def softmax(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax. Each row becomes positive weights that sum to 1."""
    shifted = scores - scores.max(axis=-1, keepdims=True)  # numerical stability
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def simple_self_attention(X: np.ndarray):
    """Attention with no parameters: score, normalize, mix.

    S = X Xᵀ        (n, n)  how similar every token is to every other token
    A = softmax(S)  (n, n)  each row is a "focus recipe" that sums to 1
    Y = A X         (n, d)  each output is a weighted mix of all input tokens
    """
    S = X @ X.T
    A = softmax(S)
    Y = A @ X
    return Y, A


def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray):
    """Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V.

    Dividing by √d_k keeps scores in a range where softmax still has
    useful gradients instead of collapsing to a near one-hot vector.
    """
    d_k = K.shape[-1]
    S = Q @ K.T / np.sqrt(d_k)
    A = softmax(S)
    return A @ V, A


def self_attention(X: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray):
    """Project X into queries, keys and values, then attend."""
    return scaled_dot_product_attention(X @ W_Q, X @ W_K, X @ W_V)


def multi_head_attention(X: np.ndarray, heads, W_O: np.ndarray):
    """Run several attention heads in parallel and combine them.

    heads: list of (W_Q, W_K, W_V) tuples, one per head, each (d_model, d_k)
    W_O:   (h * d_v, d_model) output projection
    """
    outputs, weights = [], []
    for W_Q, W_K, W_V in heads:
        out, A = self_attention(X, W_Q, W_K, W_V)
        outputs.append(out)
        weights.append(A)
    concat = np.concatenate(outputs, axis=-1)
    return concat @ W_O, weights


# The running example used throughout the course: "bank of river"
X_EXAMPLE = np.array(
    [
        [1, 0, 1, 0],  # bank
        [0, 1, 0, 1],  # of
        [1, 1, 0, 0],  # river
    ],
    dtype=float,
)
TOKENS = ["bank", "of", "river"]


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)

    Y, A = simple_self_attention(X_EXAMPLE)
    print("Step 1 — attention with no weights")
    print("Focus matrix A:\n", A)
    print("New embeddings Y:\n", Y)

    W_Q = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=float)
    W_K = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    W_V = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=float)
    Y, A = self_attention(X_EXAMPLE, W_Q, W_K, W_V)
    print("\nStep 2 — learnable Q, K, V")
    print("Focus matrix A:\n", A)
    print("New embeddings Y:\n", Y)
