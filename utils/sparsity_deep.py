import numpy as np


def forward_pass_tanh_deep(X, W1, W_internals, W_L, b_hidden, b_out):
    """
    Multi-layer tanh forward pass for a single posterior sample.

    X:           (N, P)
    W1:          (P, H)          — input-to-first-hidden weights
    W_internals: list of (H, H)  — one matrix per internal (hidden-to-hidden) layer
    W_L:         (H, O)          — last-hidden-to-output weights
    b_hidden:    (num_layers, H) — bias for each hidden layer (input + internal)
    b_out:       (O,) or scalar  — output bias

    Returns (N, O) predictions.
    """
    h = np.tanh(X @ W1 + b_hidden[0])
    for i, W_int in enumerate(W_internals):
        h = np.tanh(h @ W_int + b_hidden[i + 1])
    return h @ W_L + np.atleast_1d(b_out).reshape(1, -1)


def build_posterior_mask(W_samples, sparsity):
    """
    Binary mask based on E|w| across posterior samples.

    W_samples: (S, ...) array of posterior draws for one weight matrix.
    sparsity:  fraction in [0, 1) to zero out (lowest E|w| weights).

    Returns a float mask with the same shape as one draw.
    """
    if sparsity == 0.0:
        return np.ones(W_samples.shape[1:], dtype=float)

    score = np.abs(W_samples).mean(axis=0)
    k = int(np.floor(sparsity * score.size))
    if k == 0:
        return np.ones_like(score, dtype=float)

    thresh = np.partition(score.ravel(), k - 1)[k - 1]
    mask = (score > thresh).astype(float)

    # resolve ties so exactly k weights are zeroed
    diff = int(mask.sum()) - (score.size - k)
    if diff > 0:
        tied = np.where(score.ravel() == thresh)[0]
        mask.ravel()[tied[:diff]] = 0.0
    elif diff < 0:
        tied = np.where(score.ravel() == thresh)[0]
        keep = tied[mask.ravel()[tied] == 0.0]
        mask.ravel()[keep[: -diff]] = 1.0

    return mask
