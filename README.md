# Deep Learning exercises

Student material for the HPI Deep Learning course, taught by Christoph Lippert
with Noel Danz, Pedro De Macedo Gomes, and Anees Hashmi.

Start with [Week 1: fitting a line](exercises/week01/README.md) and its
[worksheet](exercises/week01/worksheet.md). The repository includes the runnable
experiment, its synthetic data generator, and the lecture figures. It works
independently of the book repository. Instructor notes and worksheet answer keys
are maintained separately.

Read the rendered [classroom worksheets online](https://healthml.github.io/deep-learning-exercises/).
Updates are published automatically after merging into `main`. For the one-time
Pages setup and the permanent link to add in Moodle, see
[publishing instructions](docs/publishing.md).

## Get the code

```sh
git clone https://github.com/HealthML/deep-learning-exercises.git
cd deep-learning-exercises
```

For an existing checkout, commit or save your own changes before updating with
`git pull --ff-only`. You can change the starter locally or keep experiment copies
under `exercises/week01/output/`.

## Install Python and create an environment

Use [uv](https://docs.astral.sh/uv/) to manage Python and the course packages.
Install it once using the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/):

### macOS and Linux

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows PowerShell

```powershell
winget install --id=astral-sh.uv -e
```

Open a new terminal after installation so `uv` is available on your `PATH`.
Then run these commands from the repository root on **all platforms**:

```sh
uv sync --locked
uv run --locked python tools/check_setup.py
uv run --locked python exercises/week01/starter.py
```

`uv sync --locked` downloads Python 3.11 if needed, creates `.venv`, and installs
the student package in editable mode together with the exact dependency versions
in `uv.lock`. Python 3.11 is the default in `.python-version`; CI also checks
Python 3.12 and 3.13. No separate Python installation or environment activation
is required. Use `uv run --locked python ...` for worksheet commands on every
platform, including Windows PowerShell.

The project selects CPU-only PyTorch wheels on Linux and Windows; macOS uses
PyPI. On macOS, the PyTorch dependency supports Apple Silicon with macOS 11 or
newer. All tensors in Week 1 stay on CPU. The dependency and index settings live
in `pyproject.toml`, so setup and CI use the same choices.

If you already have a `.venv` from the previous setup, `uv sync --locked` updates
it. If it uses a different Python version, uv recreates it for Python 3.11.

## In a new terminal

From the repository root, continue with `uv run --locked python ...`.
After updating the repository with `git pull --ff-only`, run `uv sync --locked`
to apply any dependency changes.

## Troubleshooting

- **`uv` is not found:** reopen the terminal after installing uv and follow the
  installer's `PATH` instructions.
- **`No module named dlexercises`:** run `uv sync --locked` from the repository
  root and use `uv run --locked python tools/check_setup.py`. The script reports
  the interpreter and package location so you can identify an environment mismatch.
- **The lockfile needs updating:** after changing dependencies, use `uv add`
  (or `uv remove`) and commit both `pyproject.toml` and `uv.lock`. For an intentional
  dependency refresh, run `uv lock --upgrade`, `uv sync --locked`, and the checks
  below before committing the updated lockfile.
- **No plot window:** use `--save fit.png` instead of `--plot`; on Linux servers,
  set `MPLBACKEND=Agg` if needed. A graphical desktop is needed for a plot window.
- **No compatible PyTorch wheel:** use the pinned Python 3.11 environment and
  check your platform against the [official PyTorch installer](https://pytorch.org/get-started/locally/).
- **Quarto or Jupyter is missing:** both are optional for this initial exercise.
  Read the Markdown worksheet and run the Python script directly.

## Files and checks

| Path | Purpose |
|---|---|
| `exercises/week01/starter.py` | Manual parameter trials and guided fitting experiments |
| `exercises/week01/worksheet.md` | Questions and spaces for your explanations |
| `exercises/week01/worksheet.qmd` | Optional Quarto source of the same worksheet |
| `exercises/week01/figures/` | Lecture visuals accompanying the worksheet |
| `src/dlexercises/data.py` | Reproducible synthetic sensor data |
| `pyproject.toml` / `uv.lock` | Dependencies, platform-specific PyTorch index, and exact versions |
| `.python-version` | Default Python version managed by uv |
| `tools/check_setup.py` | Interpreter, dependency, data, and image-export check |
| `tools/smoke_check.py` | Run each experiment mode and check CLI/image behavior |

You can run the repository checks with:

```sh
uv run --locked python -m compileall -q src exercises tools
uv run --locked python tools/smoke_check.py
```

## Attribution and license

Week 1 material is adapted from HealthML/deep-learning-book at commit
`7919dce810877339178c33725e17959c83c02d47` by Christoph Lippert and contributors.
The dataset and equations match the course book and slides. See [LICENSE.md](LICENSE.md):
original source code is MIT; prose, exercises, and figures are CC BY-SA 4.0.
