"""Transparent Week 2 routines: NumPy, float64, full batches, mean log loss.

Adapted from HealthML/deep-learning-book src/dlbook/logistic.py at commit
7803e0bcbbe1b20d7563db09cee97f4976242e6b (Christoph Lippert and contributors,
MIT). The dataset's separate CC BY 4.0 attribution is bundled beside its CSV.
No book checkout, network access, or clinical software is required.
"""
from importlib.resources import files

import numpy as np


def load_wdbc():
    """Return mean concavity, mean texture, and malignant=1; preserve book rows."""
    with files("dlexercises").joinpath("datasets/wdbc/two_features.csv").open("rb") as stream:
        data = np.loadtxt(stream, delimiter=",", skiprows=1, dtype=np.float64)
    return data[:, :2], data[:, 2]


def stratified_split(y, seed=8):
    """Book split: 60/20/remainder per class, then shuffle each combined split."""
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for label in (0, 1):
        indices = rng.permutation(np.flatnonzero(y == label))
        n_train, n_valid = int(0.6 * len(indices)), int(0.2 * len(indices))
        for part, group in zip(parts, np.split(indices, [n_train, n_train + n_valid])):
            part.extend(group)
    return tuple(rng.permutation(part) for part in parts)


def design(features):
    """Rows are examples; prepend intercept: theta=(b,w_concavity,w_texture)."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("features must have shape (n, d)")
    return np.column_stack((np.ones(len(features)), features))


def sigmoid(z):
    """Evaluate sigmoid without overflowing at large positive or negative scores."""
    return np.exp(-np.logaddexp(0.0, -np.asarray(z, dtype=np.float64)))


def _check_shapes(matrix, targets, theta):
    if matrix.ndim != 2 or theta.shape != (matrix.shape[1],):
        raise ValueError("matrix must be (n, d), theta must be (d,)")
    if targets.shape != (len(matrix),):
        raise ValueError("targets must have shape (n,), not (n, 1)")


def loss(matrix, targets, theta):
    """Mean negative Bernoulli log likelihood, calculated directly from logits."""
    _check_shapes(matrix, targets, theta)
    z = matrix @ theta
    return np.mean(np.logaddexp(0.0, z) - targets * z)


def gradient(matrix, targets, theta):
    _check_shapes(matrix, targets, theta)
    return matrix.T @ (sigmoid(matrix @ theta) - targets) / len(targets)


def hessian(matrix, theta):
    p = sigmoid(matrix @ theta)
    return matrix.T @ ((p * (1 - p))[:, None] * matrix) / len(matrix)


def gradient_descent(matrix, targets, steps=2000):
    """Same zero initialization and eta=1/L rule in every coordinate system."""
    L = np.linalg.eigvalsh(matrix.T @ matrix / (4 * len(targets)))[-1]
    theta = np.zeros(matrix.shape[1])
    history = [loss(matrix, targets, theta)]
    for _ in range(steps):
        theta -= gradient(matrix, targets, theta) / L
        history.append(loss(matrix, targets, theta))
    return theta, np.array(history)


def newton(matrix, targets, max_steps=30, tol=1e-10):
    """Solve Hs=g and backtrack; no inverse, hidden penalty, or diagonal jitter.

    Stop when half the squared Newton decrement is <= tol. This is a local
    quadratic diagnostic, not an exact loss-gap certificate. Singular problems
    and failures to converge raise errors rather than changing the objective.
    """
    theta = np.zeros(matrix.shape[1])
    history, accepted_steps = [loss(matrix, targets, theta)], []
    for _ in range(max_steps):
        g = gradient(matrix, targets, theta)
        direction = np.linalg.solve(hessian(matrix, theta), g)
        decrement_squared = float(g @ direction)
        if decrement_squared / 2 <= tol:
            return theta, np.array(history), np.array(accepted_steps)
        step = 1.0
        while loss(matrix, targets, theta - step * direction) > history[-1] - 1e-4 * step * decrement_squared:
            step *= 0.5
            if step < 2**-40:
                raise RuntimeError("Newton line search failed; inspect curvature and data.")
        theta -= step * direction
        history.append(loss(matrix, targets, theta))
        accepted_steps.append(step)
    raise RuntimeError("Newton did not meet its stopping criterion.")


def binary_metrics(targets, probabilities, threshold=0.5):
    """Positive=malignant. Pairwise AUROC includes half credit for tied scores."""
    if targets.ndim != 1 or probabilities.shape != targets.shape:
        raise ValueError("targets and probabilities must have identical shape (n,)")
    pred, positive = probabilities >= threshold, targets == 1
    tp, fn = np.sum(pred & positive), np.sum(~pred & positive)
    fp, tn = np.sum(pred & ~positive), np.sum(~pred & ~positive)
    comparisons = probabilities[positive, None] - probabilities[~positive][None, :]
    # Probability clipping is for reporting only; training uses stable logits.
    q = np.clip(probabilities, 1e-15, 1 - 1e-15)
    return {
        "log_loss": float(-np.mean(targets * np.log(q) + (1 - targets) * np.log1p(-q))),
        "accuracy": float(np.mean(pred == positive)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "auroc": float(np.mean((comparisons > 0) + 0.5 * (comparisons == 0))) if comparisons.size else float("nan"),
        "brier": float(np.mean((probabilities - targets)**2)),
    }


def fit_penalized(matrix, targets, penalty, strength, max_steps=10000):
    """L1 proximal gradient or L2 gradient; intercept is never penalized."""
    if penalty not in ("L1", "L2") or strength < 0:
        raise ValueError("choose penalty='L1' or 'L2' and strength >= 0")
    theta = np.zeros(matrix.shape[1])
    L = np.linalg.eigvalsh(matrix.T @ matrix / (4 * len(targets)))[-1]
    eta = 1 / (L + (strength if penalty == "L2" else 0))
    for _ in range(max_steps):
        g = gradient(matrix, targets, theta)
        if penalty == "L2":
            g[1:] += strength * theta[1:]
        candidate = theta - eta * g
        if penalty == "L1":
            candidate[1:] = np.sign(candidate[1:]) * np.maximum(np.abs(candidate[1:]) - eta * strength, 0)
        residual = np.max(np.abs(candidate - theta)) / eta
        theta = candidate
        if residual <= 1e-8:
            return theta
    raise RuntimeError("Penalized solver did not converge.")
