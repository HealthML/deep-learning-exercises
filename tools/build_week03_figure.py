"""Rebuild the diagnostic figure: measured synthetic-data run A, illustrative B/C."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def recorded_run():
    x_train, x_valid = np.array([1., 2.]), np.array([1.5, 2.5])
    y_train, y_valid = 2*x_train, 2*x_valid
    w, lr = 0., .9
    train, valid = [], []
    for update in range(13):
        train.append(.5*np.mean((w*x_train-y_train)**2))
        valid.append(.5*np.mean((w*x_valid-y_valid)**2))
        if update < 12:
            gradient = np.mean(x_train*(w*x_train-y_train))
            w -= lr*gradient
    return np.arange(13), np.array(train), np.array(valid)


def main():
    plt.rcParams['svg.hashsalt'] = 'week03-diagnostic-runs'
    t = np.arange(61)
    a = recorded_run()
    series = [a, (t, 1.6*np.exp(-t/130)+.3, 1.6*np.exp(-t/130)+.5),
              (t, 1.4*np.exp(-t/12)+.12, .6*np.exp(-t/10)+.35+.00045*t*t)]
    fig, axes = plt.subplots(1,3,figsize=(11,3.3))
    for ax,label,(updates,train,valid) in zip(axes,'ABC',series):
        ax.plot(updates,train,label='Training',color='#0072B2')
        ax.plot(updates,valid,label='Validation',color='#D55E00')
        ax.set(title=f'Run {label}',xlabel='Updates',ylabel='Loss')
        ax.legend(fontsize=8); ax.spines[['top','right']].set_visible(False)
    axes[0].set_yscale('log'); axes[0].set_ylabel('Loss (log scale)')
    fig.tight_layout()
    output = Path(__file__).resolve().parents[1]/'exercises/week03/figures/diagnostic-runs.svg'
    fig.savefig(output,facecolor='white',metadata={'Date':None})
    plt.close(fig)


if __name__ == '__main__':
    main()
