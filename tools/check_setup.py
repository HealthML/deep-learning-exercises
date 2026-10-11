"""Check the active interpreter, course package, CPU data, and image export."""
from __future__ import annotations

from io import BytesIO
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import dlexercises
from dlexercises.data import calibration_data


def main() -> None:
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Course package: {dlexercises.__file__}")
    print(f"PyTorch: {torch.__version__}; Matplotlib: {matplotlib.__version__}")
    data = calibration_data()
    assert tuple(map(lambda key: len(data[key]), ("x", "x_valid", "x_test"))) == (40, 10, 10)
    assert all(t.device.type == "cpu" and t.dtype == torch.float64 for t in data.values())
    fig, ax = plt.subplots()
    ax.scatter(data["x"], data["y"])
    with BytesIO() as buffer:
        fig.savefig(buffer, format="png")
        assert buffer.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
    plt.close(fig)
    print("Ready: CPU float64 data (40/10/10), imports, and PNG export work.")


if __name__ == "__main__":
    main()
