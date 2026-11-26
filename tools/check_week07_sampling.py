"""Technical checks for the second Week 7 notebook; optionally execute a completed notebook path.

The starter intentionally has unfinished student implementations. This checker
runs its guarded cells and independently exercises the supplied code. Given a
completed notebook, it also audits the DDIM step, the jump plan, the paired
change, the mixed step, and guidance against independent constructions. It does
not insert solutions into student tasks or grade them.
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

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT/'exercises/week7/diffusion_sampling.ipynb'
CORE = {'step_ready': 'TODO 1 (the DDIM step)', 'plan_ready': 'TODO 2 (the jump plan)',
        'change_ready': 'TODO 3 (the paired change)'}


class Fixed(nn.Module):
    """A stand-in denoiser: returns one fixed noise tensor and records its time labels."""
    def __init__(self, noise):
        super().__init__()
        self.noise, self.labels = noise, []

    def forward(self, x, t):
        self.labels.append(int(t[0]))
        return self.noise


class Stub(nn.Module):
    """A stand-in network with a label input: every pixel of its output equals the label."""
    def forward(self, x, t, label):
        return label.float().view(-1, 1, 1, 1)*torch.ones_like(x)


def check_supplied(scope):
    schedule, T, train, valid, model = (scope[name] for name in ('schedule', 'T', 'train', 'valid', 'model'))
    # The same data and schedule as in the first notebook of the week.
    torch.testing.assert_close(train, scope['make_disks'](256, seed=7), rtol=0, atol=0)
    assert train.shape == (256, 1, 16, 16) and valid.shape == (64, 1, 16, 16)
    assert train.dtype == torch.float32 and -1 <= train.min() and train.max() <= 1
    abar, beta = schedule.abar.double(), schedule.beta.double()
    assert schedule.steps == T == 64 and len(abar) == T + 1
    assert abar[0] == 1 and beta[0] == 0 and (abar[1:] < abar[:-1]).all() and abar[T] < 1e-5
    torch.testing.assert_close(abar, torch.cumprod(1 - beta, 0), rtol=1e-5, atol=1e-9)
    # Forward process and clean estimate, tested on inputs built without the notebook's formulas.
    generator = torch.Generator().manual_seed(0)
    clean, noise = valid[:5], torch.randn(5, 1, 16, 16, generator=generator)
    t = torch.tensor([1, 7, 32, 50, 64])
    level = schedule.abar[t].view(-1, 1, 1, 1)
    noisy = level.sqrt()*clean + (1 - level).sqrt()*noise
    torch.testing.assert_close(scope['q_sample'](clean, t, noise, schedule), noisy)
    torch.testing.assert_close(scope['predict_clean'](noisy, t, noise, schedule), clean, atol=1e-3, rtol=0)
    # Reverse-step tables: given x_0, mean and variance of x_{t-1} must equal direct noising to t-1.
    clean_weight, state_weight, variance = (getattr(schedule, name).double()[1:]
                                            for name in ('clean_weight', 'state_weight', 'variance'))
    earlier, now = abar[:-1], abar[1:]
    torch.testing.assert_close(clean_weight + state_weight*now.sqrt(), earlier.sqrt(), atol=1e-4, rtol=0)
    torch.testing.assert_close(state_weight**2*(1 - now) + variance, 1 - earlier, atol=1e-4, rtol=0)
    for step in (64, 32, 2):  # DDPM's share of fresh noise for neighbouring levels
        np.testing.assert_allclose(scope['ddpm_share'](step, step - 1, schedule), beta[step]/(1 - abar[step]), rtol=1e-4)
    # One trained network with the lesson's size; frozen and unchanged by every experiment.
    losses = scope['losses']
    assert sum(p.numel() for p in model.parameters()) == 35513
    assert len(losses) == 1800 and np.isfinite(losses).all() and np.mean(losses[-50:]) < np.mean(losses[:50])/5
    assert not any(p.requires_grad for p in model.parameters())
    assert all(torch.equal(value, scope['frozen'][name]) for name, value in model.state_dict().items())
    # Diagnostics on images with known answers.
    flat = torch.zeros(2, 1, 16, 16)
    board = ((torch.arange(16)[:, None] + torch.arange(16)[None])%2*2. - 1)[None, None]  # checkerboard of -1, +1
    assert scope['roughness'](flat).abs().max() == 0 and scope['variety'](flat) == 0
    torch.testing.assert_close(scope['roughness'](board), torch.ones(1))
    torch.testing.assert_close(scope['bright_area'](board), torch.tensor([.5]))
    # The oracle leads the clean estimate to its image and measures the noise in a state.
    oracle = scope['Oracle'](clean[:1], schedule)
    step = torch.full((64,), 20)
    state = schedule.abar[20].sqrt()*clean[:1] \
        + (1 - schedule.abar[20]).sqrt()*torch.randn(64, 1, 16, 16, generator=generator)
    recovered = scope['predict_clean'](state, step, oracle(state, step), schedule)
    torch.testing.assert_close(recovered, clean[:1].expand(64, -1, -1, -1), atol=1e-4, rtol=0)
    label, measured = oracle.log[-1]
    assert label == 20 and abs(measured - (1 - schedule.abar[20]).sqrt().item()) < .02
    # The first finding: the DDPM sampler ends at the oracle's image, the shortcut does not.
    reference, shortcut, books = scope['reference'], scope['shortcut'], scope['logbooks']
    for images in (reference, shortcut):
        assert images.shape == (64, 1, 16, 16) and torch.isfinite(images).all()
    area = scope['bright_area'](reference)
    assert reference.abs().max() < 1.001 and (area == 0).sum() <= 8 and .05 < area.mean() < .2
    assert books['DDPM, 64 evaluations'].deviation < 1e-3 and len(books['DDPM, 64 evaluations'].log) == 64
    skipped = books['Shortcut, 16 evaluations']
    assert skipped.deviation > .5 and [label for label, _ in skipped.log] == list(range(64, 0, -4))
    assert scope['roughness'](shortcut).mean() > 2*scope['roughness'](reference).mean()
    return clean


def expected_jump(x, predicted, step, earlier, schedule, share=0., z=0.):
    """One jump from `step` to `earlier`, written out independently of the notebook."""
    a, b = schedule.abar[step].double(), schedule.abar[earlier].double()
    x, predicted = x.double(), predicted.double()
    estimate = ((x - (1 - a).sqrt()*predicted)/a.sqrt()).clamp(-1, 1)
    old = (x - a.sqrt()*estimate)/(1 - a).sqrt()
    return (b.sqrt()*estimate + ((1 - share)*(1 - b)).sqrt()*old + (share*(1 - b)).sqrt()*z).float()


def check_completed(scope, clean):
    schedule, T, model, start = (scope[name] for name in ('schedule', 'T', 'model', 'start'))
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(4, 1, 16, 16, generator=generator)
    fixed = Fixed(torch.randn(4, 1, 16, 16, generator=generator))
    jumps = [(64, 43), (40, 13), (9, 8), (5, 0)]
    # TODO 1: single jumps of any size; the fixed prediction makes the clipping matter.
    for step, earlier in jumps:
        torch.testing.assert_close(scope['ddim_step'](fixed, x, step, earlier, schedule),
                                   expected_jump(x, fixed.noise, step, earlier, schedule), atol=1e-4, rtol=1e-4)
    assert fixed.labels == [step for step, _ in jumps]
    # TODO 2: every plan starts at T, lands on the next label, ends at 0, and has one jump per evaluation.
    for evaluations in (1, 2, 3, 7, 16, 64):
        plan = scope['jump_plan'](T, evaluations)
        labels, landings = zip(*plan)
        assert len(plan) == evaluations and labels[0] == T and landings[-1] == 0
        assert landings[:-1] == labels[1:] and all(step > earlier for step, earlier in plan)
        assert evaluations == 1 or labels[-1] == 1
    # The DDIM sampler ends at the oracle's image for any budget, with one evaluation per jump.
    for evaluations in (64, 16, 7, 1):
        oracle = scope['Oracle'](clean[:1], schedule)
        result = scope['sample_ddim'](oracle, start[:8], schedule, evaluations)
        assert (result - clean[:1]).abs().max() < 2e-3 and len(oracle.log) == evaluations
    # TODO 3: the median of one Euclidean distance per pair.
    first, second = torch.randn(2, 9, 1, 16, 16, generator=generator)
    distances = sorted(float((first[i] - second[i]).double().square().sum().sqrt()) for i in range(9))
    assert abs(float(scope['paired_change'](first, second)) - distances[4]) < 1e-4
    # The comparison: every budget gives finite images in range; DDIM repeats exactly.
    samples = scope['samples']
    for evaluations, images in samples.items():
        assert images.shape == (64, 1, 16, 16) and torch.isfinite(images).all() and images.abs().max() < 1.001
        torch.testing.assert_close(images, scope['sample_ddim'](model, start, schedule, evaluations), rtol=0, atol=0)
    change = None
    if {64, 16} <= set(samples):  # a quarter of the evaluations must not turn images into unrelated ones
        change = float(scope['paired_change'](samples[16], samples[64]))
        assert change < float(scope['paired_change'](samples[64], samples[64].roll(1, dims=0)))/3
    assert all(torch.equal(value, scope['frozen'][name]) for name, value in model.state_dict().items())
    done = ['the DDIM step', 'the jump plan', 'oracle sampling with 64, 16, 7 and 1 evaluations', 'the paired change']
    # TODO 4: any mixture of old and fresh noise; share 0 is DDIM, DDPM's share is the DDPM sampler.
    if scope['mix_ready']:
        z = torch.randn(4, 1, 16, 16, generator=generator)
        for share in (0., .3, 1.):
            mixed = scope['mixed_step'](fixed, x, 40, 13, torch.tensor(share), z, schedule)
            torch.testing.assert_close(mixed, expected_jump(x, fixed.noise, 40, 13, schedule, share, z.double()),
                                       atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(scope['sample_mixed'](model, start, schedule, 16, 0., seed=3),
                                   scope['sample_ddim'](model, start, schedule, 16), atol=1e-5, rtol=0)
        torch.testing.assert_close(scope['sample_mixed'](model, start, schedule, T, 'ddpm', seed=31),
                                   scope['reference'], atol=1e-3, rtol=0)
        done.append('the mixed step and its DDIM and DDPM settings')
    # TODO 5 (extended exercise): the guided prediction, and a second, frozen network with a label input.
    if scope['guidance_ready']:
        guided = scope['Guided'](Stub(), scope['RIGHT'], 3.)  # without a label 2, with RIGHT 1
        torch.testing.assert_close(guided(x, torch.full((4,), 32)), torch.full_like(x, 2 + 3*(1 - 2)))
        assert guided.calls == 2
        conditional = scope['conditional']
        assert sum(p.numel() for p in conditional.parameters()) == 35513 + 3*32
        assert not any(p.requires_grad for p in conditional.parameters())
        done.append('guidance')
    return change, done


def main():
    path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else STARTER
    notebook = json.loads(path.read_text(encoding='utf-8'))
    code_cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    scope = {'__name__': '__main__'}
    with redirect_stdout(StringIO()):
        for i, cell in enumerate(code_cells):
            exec(compile(''.join(cell['source']), f'{path.name} code cell {i}', 'exec'), scope)
            plt.close('all')
    clean = check_supplied(scope)
    ready = [bool(scope[name]) for name in CORE]
    if all(ready) and 'samples' in scope:
        change, done = check_completed(scope, clean)
        if change is not None:
            print(f'Median change of an image from 64 to 16 DDIM evaluations: {change:.2f}')
        unfinished = [name for name, flag in (('activity 4', 'mix_ready'), ('activity 5', 'guidance_ready'))
                      if not scope[flag]]
        print('Week 7 sampling passes: supplied code, ' + ', '.join(done) + '.'
              + (f' Unfinished: {" and ".join(unfinished)}.' if unfinished else ''))
    elif any(ready):
        missing = ', '.join(task for task, flag in zip(CORE.values(), ready) if not flag)
        print(f'Supplied code passes. Still unfinished: {missing}. '
              'Complete these tasks and rerun for the full audit.')
    else:
        # Unfinished gaps return None, and no downstream experiment may have run.
        schedule = scope['schedule']
        x = torch.zeros(2, 1, 16, 16)
        assert 'samples' not in scope and scope['conditional'] is None
        assert scope['ddim_step'](Fixed(x), x, 32, 10, schedule) is None
        assert scope['jump_plan'](64, 4) is None and scope['paired_change'](x, x) is None
        if any(cell['outputs'] or cell['execution_count'] is not None for cell in code_cells):
            print('Note: this notebook contains outputs. Clear them before committing the starter.')
        print('Week 7 sampling passes: all starter cells, reproducible disk data, schedule and reverse-step '
              'tables, the trained and frozen 35,513-parameter network, diagnostics, the oracle with its '
              'logbook, and the failing shortcut. Student gaps are not auto-filled.')


if __name__ == '__main__':
    main()
