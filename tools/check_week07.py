"""Technical checks for Week 7; optionally execute a completed notebook path.

The starter intentionally has unfinished student implementations. This checker
runs its guarded cells and independently exercises the supplied infrastructure.
Given a completed notebook, it also audits the forward process, loss, time
input, sampler, training outcome, and inpainting against independent
constructions. It does not insert solutions into student tasks or grade them.
"""
from contextlib import redirect_stdout
from io import StringIO
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT/'exercises/week7/diffusion.ipynb'
READY = ('forward_ready', 'clean_ready', 'loss_ready', 'time_ready', 'sampler_ready')


def check_supplied(scope):
    schedule, T, train, valid = scope['schedule'], scope['T'], scope['train'], scope['valid']
    # Same seed must reproduce the data exactly; validation images are independent.
    torch.testing.assert_close(train, scope['make_disks'](256, seed=7), rtol=0, atol=0)
    assert train.shape == (256, 1, 16, 16) and valid.shape == (64, 1, 16, 16)
    assert train.dtype == torch.float32 and -1 <= train.min() and train.max() <= 1
    assert not torch.equal(train[:64], valid)
    for images in (train, valid):
        # Every image holds one disk of plausible size that the border does not cut off.
        area = (images > 0).float().sum(dim=(1, 2, 3))
        assert 9 <= area.min() and area.max() <= 60
        border = torch.cat([images[..., 0, :], images[..., -1, :], images[..., 0], images[..., -1]], dim=-1)
        assert border.max() < -.5
    # Schedule tables are indexed by t with a placeholder at 0 and end in noise.
    abar, beta = schedule.abar.double(), schedule.beta.double()
    assert schedule.steps == T == 64 and len(abar) == T + 1
    assert abar[0] == 1 and beta[0] == 0 and (abar[1:] < abar[:-1]).all()
    assert 0 < beta[1:].min() and beta[1:].max() < .9991 and abar[T] < 1e-5
    torch.testing.assert_close(abar, torch.cumprod(1 - beta, 0), rtol=1e-5, atol=1e-9)
    torch.testing.assert_close(schedule.alpha, 1 - schedule.beta)
    cosine = lambda t: math.cos((t/T + .008)/1.008*math.pi/2)**2
    for t in (1, 16, 32, 48, 63):
        np.testing.assert_allclose(abar[t], cosine(t)/cosine(0), rtol=1e-4)
    np.testing.assert_allclose(scope['copied_linear'].abar[T], np.prod(1 - np.linspace(1e-4, .02, T)), rtol=1e-5)
    # Reverse-step tables: given x_0, the mean and variance of x_{t-1} must equal
    # direct noising to level t-1. This does not reuse the notebook's formulas.
    clean_weight, state_weight, variance = (getattr(schedule, name).double()[1:]
                                            for name in ('clean_weight', 'state_weight', 'variance'))
    earlier, now = abar[:-1], abar[1:]
    torch.testing.assert_close(clean_weight + state_weight*now.sqrt(), earlier.sqrt(), atol=1e-4, rtol=0)
    torch.testing.assert_close(state_weight**2*(1 - now) + variance, 1 - earlier, atol=1e-4, rtol=0)
    assert abs(clean_weight[0] - 1) < 1e-4 and state_weight[0] == 0 and variance[0] == 0
    # Reference denoisers, tested on inputs built without the student's q_sample.
    generator = torch.Generator().manual_seed(0)
    clean, noise = valid[:5], torch.randn(5, 1, 16, 16, generator=generator)
    t = torch.tensor([1, 7, 32, 50, 64])
    level = schedule.abar[t].view(-1, 1, 1, 1)
    noisy = level.sqrt()*clean + (1 - level).sqrt()*noise
    torch.testing.assert_close(scope['Oracle'](clean, schedule)(noisy, t), noise, atol=1e-4, rtol=0)
    assert not scope['ZeroNoise']()(noisy, t).any() and scope['CopyInput']()(noisy, t) is noisy
    features = scope['time_features'](torch.tensor([0, 32, 64]), T)
    expected = torch.tensor([[0., 0, 1, 0, 1], [.5, 1, 0, 0, -1], [1, 0, -1, 0, 1]])
    torch.testing.assert_close(features, expected, atol=1e-6, rtol=0)
    # The network matches the lesson's size, keeps the image shape, and trains end to end.
    torch.manual_seed(0)
    network = scope['TinyTimeUNet'](T)
    assert sum(p.numel() for p in network.parameters()) == 35513
    output = network(noisy, t)
    assert output.shape == noisy.shape
    F.mse_loss(output, noise).backward()
    assert network.enc1.conv1.weight.grad.norm() > 0 and network.output.weight.grad.norm() > 0
    return clean, noise, t, noisy


def check_completed(scope, clean, noise, t, noisy):
    schedule, T, train = scope['schedule'], scope['T'], scope['train']
    generator = torch.Generator().manual_seed(1)
    # TODOs 1-2 with a different timestep for every image, including both ends.
    torch.testing.assert_close(scope['q_sample'](clean, t, noise, schedule), noisy)
    torch.testing.assert_close(scope['predict_clean'](noisy, t, noise, schedule), clean, atol=1e-3, rtol=0)
    # TODO 3: one scalar from the supplied draws (timesteps first, then noise), with gradients.
    network = scope['TinyTimeUNet'](T)
    loss = scope['diffusion_loss'](network, train[:16], schedule, torch.Generator().manual_seed(5))
    replay = torch.Generator().manual_seed(5)
    steps = torch.randint(1, T + 1, (16,), generator=replay)
    draws = torch.randn(16, 1, 16, 16, generator=replay)
    level = schedule.abar[steps].view(-1, 1, 1, 1)
    replayed = network(level.sqrt()*train[:16] + (1 - level).sqrt()*draws, steps)
    torch.testing.assert_close(loss, F.mse_loss(replayed, draws))
    assert loss.requires_grad
    # TODO 4: read the features entering conv2. With a silent first convolution they
    # must equal SiLU(shift): one value per channel, the same at every pixel.
    block, seen = scope['TimeBlock'](1, 4), []
    block.conv2.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0]))
    shift = torch.tensor([-1., 0., 1., 2.])
    with torch.no_grad():
        block.conv1.weight.zero_()
        block.conv1.bias.zero_()
        block.time.weight.zero_()
        block.time.bias.copy_(shift)
        block(torch.randn(3, 1, 8, 8, generator=generator), torch.randn(3, 32, generator=generator))
    torch.testing.assert_close(seen[0], F.silu(shift).view(1, 4, 1, 1).expand(3, 4, 8, 8))
    # TODO 5: one step must agree with the noise form of the mean, which needs no clean estimate.
    oracle = scope['Oracle'](clean[:1], schedule)
    state = scope['q_sample'](clean[:1].repeat(4, 1, 1, 1), torch.full((4,), 20), noise[:4], schedule)
    fresh = torch.randn(4, 1, 16, 16, generator=generator)
    mean = (state - schedule.beta[20]/(1 - schedule.abar[20]).sqrt()*noise[:4])/schedule.alpha[20].sqrt()
    torch.testing.assert_close(scope['ddpm_step'](oracle, state, 20, fresh, schedule),
                               mean + schedule.variance[20].sqrt()*fresh, atol=1e-5, rtol=0)
    start = torch.randn(4, 1, 16, 16, generator=generator)
    result, states = scope['sample_ddpm'](oracle, start, schedule, seed=3, record=(64, 1))
    torch.testing.assert_close(result, clean[:1].expand(4, -1, -1, -1), atol=1e-3, rtol=0)
    assert list(states) == [64, 1] and torch.equal(states[64], start)
    # Training ran for both budgets, reduced the loss, and beats the baselines where it should.
    losses, errors = scope['losses'], scope['errors']
    assert len(losses) == 1800 and np.isfinite(losses).all()
    assert np.mean(losses[-50:]) < np.mean(losses[:50])/5
    short, final = errors['Network, 200 updates'], errors['Network, 1800 updates']
    assert (final < errors['Always zero']).all() and (final[:50] < errors['Copy the input'][:50]).all()
    assert final.mean() < short.mean()
    # Samples are finite images in range; most show a disk; fresh noise supplies the variety.
    samples, first, plain = scope['samples'], scope['first_samples'], scope['without_noise']
    for images in (samples, first, plain):
        assert images.shape == (64, 1, 16, 16) and torch.isfinite(images).all() and images.abs().max() < 1.001
    area = (samples > 0).float().mean(dim=(1, 2, 3))
    assert (area == 0).sum() <= 8 and .05 < area.mean() < .2
    assert plain.std(dim=0).mean() < samples.std(dim=0).mean()/2
    # TODO 6 (extended exercise): known pixels survive exactly; hidden pixels vary between runs.
    if scope['inpaint_ready']:
        mask, known = scope['mask'], scope['valid'][:1].repeat(8, 1, 1, 1)
        filled = scope['inpaint'](scope['model'], known, mask, schedule, seed=5)
        torch.testing.assert_close(filled*mask, known*mask, atol=1e-5, rtol=0)
        assert (filled*(1 - mask)).std(dim=0).max() > 0
    return short, final, area


def main():
    path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else STARTER
    notebook = json.loads(path.read_text(encoding='utf-8'))
    code_cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    scope = {'__name__': '__main__'}
    with redirect_stdout(StringIO()):
        for i, cell in enumerate(code_cells):
            exec(compile(''.join(cell['source']), f'{path.name} code cell {i}', 'exec'), scope)
            plt.close('all')
    clean, noise, t, noisy = check_supplied(scope)
    ready = [bool(scope[name]) for name in READY]
    if all(ready) and scope['samples'] is not None:
        short, final, area = check_completed(scope, clean, noise, t, noisy)
        print(f'Validation noise error, mean over timesteps: {short.mean():.4f} after 200 updates, '
              f'{final.mean():.4f} after 1800 | empty samples: {int((area == 0).sum())} of 64')
        extended = 'and inpainting' if scope['inpaint_ready'] else 'without the unfinished extended exercise'
        print('Week 7 passes: supplied infrastructure, direct noising, clean estimate, loss, time shift, '
              'DDPM step against the noise form of the mean, oracle sampling, training budgets, '
              f'sample range and variety, {extended}.')
    elif any(ready):
        missing = ', '.join(name for name, done in zip(READY, ready) if not done) or 'the sampling cells'
        print(f'Supplied infrastructure passes. Still unfinished: {missing}. '
              'Complete these tasks and rerun for the full audit.')
    else:
        # Unfinished gaps return None, and no downstream experiment may have run.
        schedule = scope['schedule']
        assert scope['losses'] is None and scope['samples'] is None
        assert scope['q_sample'](clean, t, noise, schedule) is None
        assert scope['predict_clean'](noisy, t, noise, schedule) is None
        assert scope['diffusion_loss'](scope['ZeroNoise'](), clean, schedule, torch.Generator()) is None
        if any(cell['outputs'] or cell['execution_count'] is not None for cell in code_cells):
            print('Note: this notebook contains outputs. Clear them before committing the starter.')
        print('Week 7 passes: all starter cells, reproducible disk data, cosine and copied schedules, '
              'reverse-step tables, reference denoisers, time features, and the 35,513-parameter network. '
              'Student gaps are not auto-filled.')


if __name__ == '__main__':
    main()
