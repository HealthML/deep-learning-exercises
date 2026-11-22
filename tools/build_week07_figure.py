"""Rebuild the Week 7 overview figure: one disk noised with the notebook's cosine schedule."""
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
import numpy as np

STEPS = 64
SHOWN = [0, 8, 16, 32, 48, 64]


def main():
    # The same disk recipe and schedule as exercises/week7/diffusion.ipynb.
    yy, xx = np.mgrid[:16, :16]
    distance = np.sqrt((yy - 8.5)**2 + (xx - 6.5)**2)
    clean = 2 / (1 + np.exp((distance - 3.2) / 0.6)) - 1
    t = np.arange(STEPS + 1)
    f = np.cos((t / STEPS + 0.008) / 1.008 * math.pi / 2)**2
    beta = np.zeros(STEPS + 1)
    beta[1:] = np.minimum(1 - f[1:] / f[:-1], 0.999)
    abar = np.cumprod(1 - beta)
    noise = np.random.default_rng(7).normal(size=clean.shape)

    fig, axes = plt.subplots(1, len(SHOWN), figsize=(9, 2.9))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.70, bottom=0.30, wspace=0.12)
    for ax, step in zip(axes, SHOWN):
        noisy = np.sqrt(abar[step]) * clean + np.sqrt(1 - abar[step]) * noise
        ax.imshow(noisy, cmap='gray', vmin=-1, vmax=1, interpolation='nearest')
        ax.set(xticks=[], yticks=[])
        ax.spines[:].set_visible(False)
        ax.set_xlabel(f'$x_{{{step}}}$', fontsize=12, labelpad=4)
    arrow = dict(arrowstyle='-|>', linewidth=2, mutation_scale=16, shrinkA=0, shrinkB=0)
    fig.text(0.5, 0.93, 'Forward process: add a little noise at every step (fixed, nothing to learn)',
             ha='center', va='center', fontsize=10.5)
    fig.patches.append(FancyArrowPatch(
        (0.06, 0.80), (0.94, 0.80), transform=fig.transFigure, color='#6b6b6b', **arrow))
    fig.patches.append(FancyArrowPatch(
        (0.94, 0.13), (0.06, 0.13), transform=fig.transFigure, color='#0072B2', **arrow))
    fig.text(0.5, 0.045, 'Reverse process: a network removes a little noise at every step (learned)',
             ha='center', va='center', fontsize=10.5)
    output = Path(__file__).resolve().parents[1] / 'exercises/week7/figures/diffusion-chain.png'
    fig.savefig(output, dpi=160, facecolor='white', metadata={'Software': None})
    plt.close(fig)


if __name__ == '__main__':
    main()
