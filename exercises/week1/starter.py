"""Week 1 classroom experiment: change constants or pass command-line options.

Run from the repository root after ``python -m pip install -e .``.
The mathematical mechanisms stay visible here; only data creation is shared.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import torch

from dlexercises.data import calibration_data

# Start by editing these three values. For the one-parameter model B stays zero.
W = 0.0
B = 0.0
MODEL = "one"                 # Change to "affine" when the bias is introduced.
MODE = "manual"               # manual, partials, coordinate, sgd, batch, solve
LR = 0.025
STEPS = 40


def loss(x, y, w, b):
    prediction = w * x + b
    residual = prediction - y
    return (residual ** 2).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--w", "--weight", dest="weight", type=float, default=W)
    parser.add_argument("--b", "--bias", dest="bias", type=float, default=B)
    parser.add_argument("--model", choices=["one", "affine"], default=MODEL)
    parser.add_argument("--method", "--mode", dest="mode",
                        choices=["manual", "partials", "coordinate", "sgd", "batch", "lstsq", "solve"],
                        default=MODE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--plot", action="store_true", help="Open a plot of the fitted line")
    parser.add_argument("--save", "--output", dest="save",
                        help="Save the plot at this path instead of opening a window")
    args = parser.parse_args()
    if args.model == "one" and args.bias != 0:
        parser.error("The one-parameter model fixes bias at zero; use --model affine to change it.")
    if args.mode in {"coordinate", "sgd", "batch", "lstsq", "solve"} and args.model != "affine":
        parser.error("For this fitting comparison, choose --model affine.")
    if args.steps < 0 or args.lr <= 0:
        parser.error("--steps must be nonnegative and --lr must be positive.")

    data = calibration_data()
    x, y = data["x"], data["y"]
    w, b = args.weight, args.bias
    print(f"PyTorch {torch.__version__}; device=cpu; dtype={x.dtype}; training examples={len(x)}")
    print(f"Before: w={w:.6f}, b={b:.6f}, training MSE={loss(x, y, w, b).item():.6f}")

    if args.mode == "partials":
        epsilon = 1e-5
        # Change only one parameter; keep the other fixed in both evaluations.
        finite_dw = (loss(x, y, w + epsilon, b) - loss(x, y, w - epsilon, b)) / (2 * epsilon)
        residual = w * x + b - y
        dw = 2 * (residual * x).mean()
        print(f"dL/dw: finite difference={finite_dw.item():.6f}, formula={dw.item():.6f}")
        if args.model == "affine":
            finite_db = (loss(x, y, w, b + epsilon) - loss(x, y, w, b - epsilon)) / (2 * epsilon)
            db = 2 * residual.mean()
            print(f"dL/db: finite difference={finite_db.item():.6f}, formula={db.item():.6f}")

    elif args.mode == "coordinate":
        for cycle in range(args.steps):
            # First optimize w with the OLD b held fixed.
            w = ((x * (y - b)).sum() / (x ** 2).sum()).item()
            # Then optimize b with the NEW w held fixed.
            b = (y - w * x).mean().item()
            if cycle == 0 or cycle == args.steps - 1:
                print(f"Cycle {cycle + 1}: w={w:.6f}, b={b:.6f}, full MSE={loss(x, y, w, b):.6f}")

    elif args.mode == "sgd":
        rng = torch.Generator().manual_seed(args.seed)
        for step in range(args.steps):
            i = torch.randint(len(x), (), generator=rng).item()
            before_sample = loss(x[i:i + 1], y[i:i + 1], w, b).item()
            # Just one example: there is no sum or mean to compute.
            residual = w * x[i] + b - y[i]
            dw, db = (2 * residual * x[i]).item(), (2 * residual).item()
            # These two derivatives use the SAME old (w,b).
            w, b = w - args.lr * dw, b - args.lr * db
            if step == 0 or step == args.steps - 1:
                after_sample = loss(x[i:i + 1], y[i:i + 1], w, b).item()
                print(f"Update {step + 1}, example {i}: its loss {before_sample:.6f} -> {after_sample:.6f}; "
                      f"full MSE={loss(x, y, w, b):.6f}")
        print("One example's loss decreasing does not guarantee that full training MSE decreases.")

    elif args.mode == "batch":
        for step in range(args.steps):
            residual = w * x + b - y
            dw = (2 * (residual * x).mean()).item()
            db = (2 * residual.mean()).item()
            # A simultaneous update: do not recompute db after changing w.
            w, b = w - args.lr * dw, b - args.lr * db
            if step == 0 or step == args.steps - 1:
                print(f"Update {step + 1}: w={w:.6f}, b={b:.6f}, full MSE={loss(x, y, w, b):.6f}")

    elif args.mode in {"lstsq", "solve"}:
        # Each row represents y_i approximately equal to w*x_i + b.
        A = torch.stack((x, torch.ones_like(x)), dim=1)
        parameters = torch.linalg.lstsq(A, y).solution
        w, b = parameters.tolist()
        print(f"Design matrix shape: {tuple(A.shape)}; target shape: {tuple(y.shape)}")

    print(f"After:  w={w:.6f}, b={b:.6f}, training MSE={loss(x, y, w, b).item():.6f}")
    if args.plot or args.save:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(x, y, label="training measurements")
        ax.plot(data["x_all"], w * data["x_all"] + b, color="tab:orange",
                label=f"prediction: w={w:.3f}, b={b:.3f}")
        ax.set(xlabel="sensor reading x", ylabel="reference target y")
        ax.legend()
        fig.tight_layout()
        if args.save:
            fig.savefig(args.save, dpi=150)
        else:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
