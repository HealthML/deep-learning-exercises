"""Technical checks for Week 5; optionally execute a completed notebook path.

The starter intentionally has unfinished student implementations. This checker
runs its guarded cells and independently exercises the supplied infrastructure.
It does not insert solutions into student tasks or claim to grade them.
"""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from dlexercises.week05 import make_data, translate, random_translate, fit, evaluate, parameter_count

ROOT = Path(__file__).resolve().parents[1]


def probe_model():
    # Independent library-only test fixture for supplied training infrastructure.
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 3),
    )


def main():
    torch.set_num_threads(2)
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/'exercises/week5/experiments.ipynb'
    notebook = json.loads(path.read_text())
    scope = {'__name__': '__main__'}
    with redirect_stdout(StringIO()):
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                exec(compile(''.join(cell['source']), f'{path.name} cell {i}', 'exec'), scope)
                plt.close('all')
    # Extra cases stay here rather than cluttering the beginner notebook.
    image = np.arange(30, dtype=float).reshape(5, 6)
    kernel = np.array([[1., 0., -1.], [.5, 0., -.5]])
    result = scope['conv2d_numpy'](image, kernel, .25)
    if result is not None:
        expected = torch.nn.functional.conv2d(
            torch.tensor(image)[None, None], torch.tensor(kernel)[None, None],
            bias=torch.tensor([.25], dtype=torch.float64))
        np.testing.assert_allclose(result, expected.numpy()[0, 0])
    result = scope['maxpool2d_numpy'](image)
    if result is not None:
        expected = torch.nn.functional.max_pool2d(torch.tensor(image)[None, None], 2)
        np.testing.assert_allclose(result, expected.numpy()[0, 0])
    if scope['residual_ready']:
        block = scope['ResidualBlock'](8)
        with torch.no_grad():
            for p in block.parameters():
                p.zero_()
        x = torch.linspace(-1, 1, 2*8*15*17).reshape(2, 8, 15, 17).requires_grad_()
        result = block(x)
        torch.testing.assert_close(result, x.relu())
        result.sum().backward()
        torch.testing.assert_close(x.grad, (x.detach() > 0).float())
    # Same seed must reproduce data exactly; shifted development pairs preserve labels.
    train, valid = make_data(120, seed=11), make_data(40, seed=22)
    repeat, shifted = make_data(40, seed=22), make_data(40, seed=22, shifted=True)
    torch.testing.assert_close(valid[0], repeat[0], rtol=0, atol=0)
    torch.testing.assert_close(valid[1], shifted[1])
    assert not torch.equal(valid[0], shifted[0])
    assert train[0].shape == (360, 1, 32, 32)
    assert train[1].bincount().tolist() == [120, 120, 120]
    assert train[0].dtype == torch.float32 and train[1].dtype == torch.long
    assert 0 <= train[0].min() <= train[0].max() <= 1
    assert not torch.equal(train[0][:120], valid[0])
    # Check translations against an independent zero-pad-and-slice construction.
    x = torch.arange(24.).reshape(1, 1, 4, 6)
    for dy in range(-4, 5):
        for dx in range(-6, 7):
            padded = torch.nn.functional.pad(x, (6, 6, 4, 4))
            expected = padded[..., 4-dy:8-dy, 6-dx:12-dx]
            torch.testing.assert_close(translate(x, dy, dx), expected)
    torch.testing.assert_close(translate(x, 20, 0), torch.zeros_like(x))
    a = random_translate(valid[0], torch.Generator().manual_seed(1))
    b = random_translate(valid[0], torch.Generator().manual_seed(1))
    torch.testing.assert_close(a, b)
    assert not torch.equal(a, valid[0])
    # All foreground strokes remain inside the image in the supplied shifted set.
    for original, moved in zip(valid[0], shifted[0]):
        assert int((original > .3).sum()) == int((moved > .3).sum())
    assert parameter_count(probe_model()) == 1299
    baseline = fit(probe_model, train, valid)
    augmented = fit(probe_model, train, valid, augment=True)
    again = fit(probe_model, train, valid)
    assert baseline['updates'] == augmented['updates'] == 18*6
    assert baseline['weight_change'] > 0
    assert np.isfinite(baseline['history']['train']).all()
    assert baseline['history']['train'][-1] < baseline['history']['train'][0]
    np.testing.assert_allclose(baseline['history']['train'], again['history']['train'], rtol=0, atol=0)
    # Restored weights must agree with the chosen validation checkpoint's loss.
    for run in [baseline, augmented]:
        score = evaluate(run['model'], valid)
        np.testing.assert_allclose(score['loss'], min(run['history']['valid']), rtol=1e-6)
        assert run['best_epoch'] == int(np.argmin(run['history']['valid']))+1
        assert not score['logits'].requires_grad
        assert score['logits'].shape == (120, 3)
    # Verify real autograd reaches the first convolution, not just the head.
    model = probe_model()
    torch.nn.functional.cross_entropy(model(train[0][:60]), train[1][:60]).backward()
    assert model[0].weight.grad.norm() > 0
    # Evaluation mode must not update BatchNorm running statistics.
    bn_model = nn.Sequential(nn.Conv2d(1, 3, 3, padding=1), nn.BatchNorm2d(3),
                             nn.AdaptiveAvgPool2d(1), nn.Flatten())
    bn_model.train(); bn_model(train[0][:60])
    before = bn_model[1].running_mean.clone()
    evaluate(bn_model, valid)
    torch.testing.assert_close(before, bn_model[1].running_mean)
    if not scope['RUN_FINAL_TEST']:
        assert 'test_centered' not in scope and 'test_shifted' not in scope
    for label, run in [('CNN', baseline), ('CNN + translations', augmented)]:
        print(f'{label}: centered={evaluate(run["model"], valid)["accuracy"]:.1%}, '
              f'shifted={evaluate(run["model"], shifted)["accuracy"]:.1%}')
    print('Week 5 passes: notebook cells, reproducible balanced data, zero-fill shifts, '
          'foreground preservation, gradient updates, training budgets, restored checkpoints, '
          'BatchNorm evaluation, and default test protection. Student gaps are not auto-filled.')


if __name__ == '__main__':
    main()
