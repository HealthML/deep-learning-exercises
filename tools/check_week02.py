"""Execute the student notebook and independently audit its scientific controls.

No Jupyter server is required. CI runs the exact delivered code cells, verifies
the default test guard, and checks derivatives and coordinate invariance at
nonzero probes rather than only mirroring symmetric initialization.
"""
from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dlexercises.week02 import (
    binary_metrics, design, gradient, hessian, load_wdbc, loss, newton,
    sigmoid, stratified_split,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    csv = ROOT / "src/dlexercises/datasets/wdbc/two_features.csv"
    assert hashlib.sha256(csv.read_bytes()).hexdigest() == "5f31487120e1f130155555883088eb6132733174b452bd58b09b132e5a0a310f"
    X_all, y_all = load_wdbc()
    assert X_all.shape == (569, 2) and y_all.dtype == np.float64
    assert np.array_equal(np.bincount(y_all.astype(int)), [357, 212])
    train, valid, test = stratified_split(y_all)
    assert tuple(map(len, (train, valid, test))) == (341, 113, 115)
    assert np.array_equal(np.sort(np.r_[train, valid, test]), np.arange(569))
    X, y = X_all[train], y_all[train]
    A, B = design(X), design((X - X.mean(axis=0)) / X.std(axis=0))

    probe, eps, basis = np.array([-0.4, 0.7, -0.2]), 1e-5, np.eye(3)
    fd_g = np.array([(loss(B, y, probe + eps * e) - loss(B, y, probe - eps * e)) / (2 * eps) for e in basis])
    fd_H = np.column_stack([(gradient(B, y, probe + eps * e) - gradient(B, y, probe - eps * e)) / (2 * eps) for e in basis])
    np.testing.assert_allclose(gradient(B, y, probe), fd_g, atol=1e-10, rtol=1e-7)
    np.testing.assert_allclose(hessian(B, probe), fd_H, atol=1e-10, rtol=1e-7)
    np.testing.assert_allclose(sigmoid(np.array([-1000.0, 0, 1000.0])), [0, 0.5, 1])
    try:
        gradient(B, y[:, None], probe)
    except ValueError:
        pass
    else:
        raise AssertionError("Column targets must fail rather than silently broadcast.")

    theta, history, steps = newton(B, y)
    raw_theta, raw_history, _ = newton(A, y)
    assert np.max(np.diff(history)) <= 1e-12
    assert np.linalg.norm(gradient(B, y, theta), ord=np.inf) < 1e-8
    assert len(steps) == len(history) - 1
    np.testing.assert_allclose(A @ raw_theta, B @ theta, atol=1e-8)
    np.testing.assert_allclose(history, raw_history, atol=1e-12)
    # An asymmetric confusion matrix catches sensitivity/specificity swaps.
    toy_y = np.array([0., 0., 0., 1., 1.])
    toy_p = np.array([0.1, 0.2, 0.9, 0.8, 0.3])
    m = binary_metrics(toy_y, toy_p)
    assert m["sensitivity"] == 1 / 2 and m["specificity"] == 2 / 3
    assert binary_metrics(toy_y, np.full(5, 0.5))["auroc"] == 0.5

    notebook = json.loads((ROOT / "exercises/week02/logistic_regression.ipynb").read_text())
    scope = {"__name__": "__main__"}
    output = StringIO()
    with redirect_stdout(output):
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                source = cell["source"]
                if isinstance(source, list):
                    source = "".join(source)
                exec(compile(source, f"week02 notebook cell {index}", "exec"), scope)
                plt.close("all")
    assert scope["OPEN_TEST"] is False
    assert "p_test" not in scope and "y_test" not in scope
    assert "Test checkpoint closed." in output.getvalue()
    assert all(not cell.get("outputs") for cell in notebook["cells"])
    assert np.count_nonzero(scope["paths"]["L1"][-1, 1:]) == 0
    assert scope["frozen_threshold"] == 0.5
    print("Week 2 passes: exact data/split, stable logits, finite differences, shape guard, Newton invariance, metrics, all notebook cells, L1 path, and closed test checkpoint.")


if __name__ == "__main__":
    main()
