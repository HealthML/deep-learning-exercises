"""Check the student CLI independently of book packages or instructor fixtures."""
from __future__ import annotations

import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "exercises/week1/starter.py"


def run(*args: str, expected: int = 0) -> str:
    environment = dict(os.environ, MPLBACKEND="Agg")
    result = subprocess.run([sys.executable, str(STARTER), *args], cwd=ROOT,
                            env=environment, text=True, capture_output=True)
    if result.returncode != expected:
        raise AssertionError(f"{args}: {result.stdout}\n{result.stderr}")
    return result.stdout + result.stderr


def main() -> None:
    run("--help")
    assert "training examples=40" in run()
    for mode in ("partials", "coordinate", "sgd", "batch", "lstsq", "solve"):
        assert "After:" in run("--model", "affine", "--method", mode, "--steps", "3")
    # The residual convention changes the signs in the analytical formulas.
    # Check them independently against the CLI's centered finite differences.
    partials = run("--model", "affine", "--method", "partials", "--w", "0.7", "--b", "-0.2")
    derivatives = re.findall(
        r"dL/d([wb]): finite difference=([-\d.]+), formula=([-\d.]+)", partials
    )
    assert [name for name, _, _ in derivatives] == ["w", "b"]
    for name, numerical, analytical in derivatives:
        assert math.isclose(float(numerical), float(analytical), abs_tol=2e-6), name
    assert run("--model", "affine", "--method", "sgd", "--steps", "3") == run(
        "--model", "affine", "--method", "sgd", "--steps", "3")
    run("--b", "1", expected=2)
    run("--steps", "-1", expected=2)
    run("--lr", "0", expected=2)
    run("--method", "batch", expected=2)
    with tempfile.TemporaryDirectory() as directory:
        image = Path(directory) / "fit.png"
        run("--w", "1", "--save", str(image))
        assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    print("All experiment modes, gradient checks, repeatable sampling, invalid arguments, and PNG export pass.")


if __name__ == "__main__":
    main()
