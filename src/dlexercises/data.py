"""Reproduce the synthetic sensor data used in the Week 1 lecture.

Source: HealthML/deep-learning-book, src/dlbook/week1.py (MIT).
Only data generation is shared; training uses the 40 training observations.
"""
from __future__ import annotations

import torch

DTYPE = torch.float64


def calibration_data() -> dict[str, torch.Tensor]:
    """Return the seed-7 sensor example and its fixed 40/10/10 split."""
    rng = torch.Generator().manual_seed(7)
    x_all = torch.linspace(-2, 2, 60, dtype=DTYPE)
    y_all = 1.7 * x_all + 0.4 + 0.35 * torch.randn(60, generator=rng, dtype=DTYPE)
    order = torch.randperm(60, generator=rng)
    train, valid, test = order[:40], order[40:50], order[50:]
    return {
        "x_all": x_all, "y_all": y_all,
        "x": x_all[train], "y": y_all[train],
        "x_valid": x_all[valid], "y_valid": y_all[valid],
        "x_test": x_all[test], "y_test": y_all[test],
    }
