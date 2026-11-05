"""Supplied CPU data, training, and plotting support for the Week 5 lab.

Model definitions and NumPy operations belong in the student notebook.
Synthetic inspection symbols are generated locally; no external data are used.
"""
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CLASSES = ('L', 'T', '+')


def translate(images, dy, dx):
    """Translate an NCHW batch with zero fill, without circular wraparound."""
    if images.ndim != 4:
        raise ValueError('Expected NCHW images')
    height, width = images.shape[-2:]
    result = torch.zeros_like(images)
    if abs(dy) >= height or abs(dx) >= width:
        return result
    sy, ey = max(0, -dy), min(height, height - dy)
    sx, ex = max(0, -dx), min(width, width - dx)
    result[..., sy + dy:ey + dy, sx + dx:ex + dx] = images[..., sy:ey, sx:ex]
    return result


def make_data(n_per_class=120, seed=11, shifted=False):
    """Balanced 32x32 symbols; same seed pairs centered and shifted images.

    Centers jitter by at most one pixel, half-size is 4 or 5, thickness 1 or 2.
    A shift of +/-8 pixels leaves every symbol fully visible. Brightness and
    background noise are independent of class. Shifts move the complete image.
    """
    rng = np.random.default_rng(seed)
    images, labels, offsets = [], [], []
    for label in range(3):
        for _ in range(n_per_class):
            image = np.zeros((32, 32), dtype=np.float32)
            cy, cx = 16 + rng.integers(-1, 2, size=2)
            half, thick = int(rng.integers(4, 6)), int(rng.integers(1, 3))
            lo_y, hi_y, lo_x, hi_x = cy-half, cy+half+1, cx-half, cx+half+1
            if label == 0:
                image[lo_y:hi_y, lo_x:lo_x+thick] = 1
                image[hi_y-thick:hi_y, lo_x:hi_x] = 1
            elif label == 1:
                image[lo_y:lo_y+thick, lo_x:hi_x] = 1
                image[lo_y:hi_y, cx:cx+thick] = 1
            else:
                image[cy:cy+thick, lo_x:hi_x] = 1
                image[lo_y:hi_y, cx:cx+thick] = 1
            image *= rng.uniform(.6, 1.)
            image += rng.normal(0, .035, image.shape).astype(np.float32)
            images.append(image.clip(0, 1))
            labels.append(label)
            offsets.append(tuple(int(v) for v in rng.choice([-8, 8], size=2)))
    x = torch.from_numpy(np.stack(images))[:, None]
    y = torch.tensor(labels, dtype=torch.long)
    if shifted:
        x = torch.cat([translate(image[None], *offset) for image, offset in zip(x, offsets)])
    order = torch.from_numpy(rng.permutation(len(y)))
    return x[order], y[order]


def random_translate(images, generator, max_shift=8):
    """Supplied label-preserving augmentation; independent offset per image."""
    offsets = torch.randint(-max_shift, max_shift+1, (len(images), 2), generator=generator)
    return torch.cat([translate(image[None], int(dy), int(dx))
                      for image, (dy, dx) in zip(images, offsets)])


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model, data):
    model.eval()  # BatchNorm must use its running statistics during evaluation.
    x, y = data
    logits = model(x)
    return {'loss': F.cross_entropy(logits, y).item(),
            'accuracy': (logits.argmax(1) == y).float().mean().item(),
            'logits': logits.detach().clone()}


def fit(factory, train, valid, *, seed=7, epochs=18, augment=False):
    """Equal update budgets; best centered-validation checkpoint is restored.

    Factory is called after seeding, so paired plain/augmented runs start with
    identical weights. Independent RNG streams isolate batching and augmentation.
    Shifted validation is diagnostic, not used by this checkpoint rule.
    """
    torch.manual_seed(seed)
    model = factory()
    initial = next(model.parameters()).detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    batches = torch.Generator().manual_seed(seed+100)
    augmentation = torch.Generator().manual_seed(seed+200)
    x, y = train
    history = {'train': [], 'valid': []}
    best_loss, best_epoch, best_state = float('inf'), 0, None
    for epoch in range(epochs):
        model.train()
        for indices in torch.randperm(len(y), generator=batches).split(60):
            inputs = random_translate(x[indices], augmentation) if augment else x[indices]
            optimizer.zero_grad()
            loss = F.cross_entropy(model(inputs), y[indices])
            loss.backward()
            optimizer.step()
        history['train'].append(evaluate(model, train)['loss'])
        score = evaluate(model, valid)['loss']
        history['valid'].append(score)
        if score < best_loss:
            best_loss, best_epoch = score, epoch+1
            best_state = deepcopy(model.state_dict())
    if best_state is None:
        raise ValueError('Training needs at least one epoch and finite validation loss')
    model.load_state_dict(best_state)
    return {'model': model, 'history': history, 'best_epoch': best_epoch,
            'weight_change': (next(model.parameters()).detach()-initial).norm().item(),
            'updates': epochs * int(np.ceil(len(y)/60))}


def show_images(data, model=None, mistakes_only=False, limit=9):
    x, y = data
    indices = torch.arange(len(y))
    if model is not None:
        logits = evaluate(model, data)['logits']
        confidence, prediction = logits.softmax(1).max(1)
        if mistakes_only:
            indices = indices[prediction != y]
            indices = indices[confidence[indices].argsort(descending=True)]
    indices = indices[:limit]
    if not len(indices):
        print('No mistakes in this set. Try a different evaluation condition.')
        return
    fig, axes = plt.subplots(1, len(indices), figsize=(2*len(indices), 2.4), squeeze=False)
    for ax, index in zip(axes.flat, indices):
        ax.imshow(x[index, 0], cmap='gray', vmin=0, vmax=1)
        title = f'True: {CLASSES[y[index]]}'
        if model is not None:
            title += f'\nPred: {CLASSES[prediction[index]]} ({confidence[index]:.0%})'
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    fig.tight_layout()
    plt.show()


def plot_losses(runs):
    fig, ax = plt.subplots(figsize=(7, 3))
    for name, run in runs.items():
        line, = ax.plot(range(1, len(run['history']['train'])+1), run['history']['train'], label=f'{name}: train')
        ax.plot(range(1, len(run['history']['valid'])+1), run['history']['valid'], '--', color=line.get_color(), label=f'{name}: valid')
    ax.set(xlabel='Epoch', ylabel='Cross-entropy (clean images)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


def report(runs, centered, shifted):
    print(f'{"Model":<20} {"Params":>8} {"Updates":>8} {"Epoch*":>7} {"Centered":>10} {"Shifted":>10}')
    for name, run in runs.items():
        model = run['model']
        print(f'{name:<20} {parameter_count(model):>8} {run["updates"]:>8} {run["best_epoch"]:>7}'
              f' {evaluate(model, centered)["accuracy"]:>10.1%}'
              f' {evaluate(model, shifted)["accuracy"]:>10.1%}')
    print('* Checkpoint selected by centered-validation loss.')
