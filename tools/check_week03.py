"""Execute Week 3 and check numerical controls independently of its plots."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    notebook = json.loads((ROOT / 'exercises/week03/optimization.ipynb').read_text())
    scope = {'__name__': '__main__'}
    with redirect_stdout(StringIO()):
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                assert not cell['outputs'] and cell['execution_count'] is None
                exec(compile(''.join(cell['source']), f'week03 cell {i}', 'exec'), scope)
                plt.close('all')
    # An unfinished warm-up must neither reveal a fallback nor block experiments.
    assert scope['student_step'](np.array([2., -1.]), np.array([2., -10.]), .05) is None
    # Run A is an actual gradient-descent history with known quadratic dynamics.
    from build_week03_figure import recorded_run
    steps, train_a, valid_a = recorded_run()
    np.testing.assert_allclose(train_a, 5 * 1.5625**steps)
    np.testing.assert_allclose(valid_a, 8.5 * 1.5625**steps)
    # Closed-form coordinate evolution checks the GD loop at every step.
    path, _ = scope['quadratic_run'](.15)
    np.testing.assert_allclose(path, np.array([.85, -.5])[None, :] ** np.arange(41)[:, None], atol=1e-14)
    gd = scope['quadratic_run'](.15)[1]
    momentum = scope['quadratic_run'](.15, .4)[1]
    unstable = scope['quadratic_run'](.4, .9)[1]
    assert momentum[-1] < gd[-1] and unstable[-1] > unstable[0]
    # Independently differentiate the exact objective with autograd at a nonzero probe.
    X, y = scope['X_train'], scope['y_train']
    w = torch.linspace(-.2, .3, X.shape[1], dtype=torch.float64, requires_grad=True)
    lam = .03
    loss = .5 * ((X @ w - y)**2).mean() + lam * (w[1:]**2).sum()
    expected = torch.autograd.grad(loss, w)[0]
    penalty = 2 * lam * w.detach().clone(); penalty[0] = 0.
    analytic = X.T @ (X @ w.detach() - y) / len(y) + penalty
    torch.testing.assert_close(expected, analytic)
    # First full-batch update from zero has no penalty contribution.
    one = scope['fit'](epochs=1, lam=lam)
    np.testing.assert_allclose(one['weights'][1], (.05 * X.T @ y / len(y)).numpy(), atol=1e-14)
    for size, run in scope['batch_runs'].items():
        assert run['updates'] == 12 * int(np.ceil(50 / size))
        assert len(run['train']) == run['updates'] + 1
        assert np.isfinite(run['weights']).all()
    partial = scope['fit'](epochs=2, batch_size=16)
    assert partial['updates'] == 8  # Final partial batch retained.
    # Sampling changes small-batch paths but not full-batch paths.
    a = scope['fit'](epochs=2, batch_size=10, seed=0)
    b = scope['fit'](epochs=2, batch_size=10, seed=1)
    assert not np.allclose(a['weights'][-1], b['weights'][-1])
    a = scope['fit'](epochs=2, batch_size=50, seed=0)
    b = scope['fit'](epochs=2, batch_size=50, seed=1)
    np.testing.assert_allclose(a['weights'], b['weights'], atol=1e-14)
    runs = scope['regularized_runs']; base = runs[0.]
    assert base['train'][600] < base['train'][20]
    assert base['valid'][600] > base['valid'][20]
    assert runs[.3]['valid'][-1] < base['valid'][-1]
    assert 0 < np.argmin(base['valid']) < 600
    # Every checkpoint score must correspond to the saved weights, without penalty.
    for run in runs.values():
        best = int(np.argmin(run['valid']))
        prediction = scope['X_valid'].numpy() @ run['weights'][best]
        np.testing.assert_allclose(.5 * np.mean((prediction - scope['y_valid'].numpy())**2), run['valid'][best])
    # All data can be regenerated; no held-out test arrays are exposed by make_data.
    regenerated = scope['make_data']()
    assert len(regenerated) == 4
    for actual, expected in zip(regenerated, [X, y, scope['X_valid'], scope['y_valid']]):
        torch.testing.assert_close(actual, expected)
    assert 'X_test' not in scope and 'y_test' not in scope
    print('Week 3 passes: all code cells, GD recurrence, momentum cases, explicit gradient, batch budgets, checkpoint alignment, and reproducible data.')


if __name__ == '__main__':
    main()
