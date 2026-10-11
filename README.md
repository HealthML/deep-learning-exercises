# Deep Learning exercises

Student material for the HPI Deep Learning course, taught by Christoph Lippert
with Noel Danz, Pedro De Macedo Gomes, and Anees Hashmi.

Start with [Week 1: fitting a line](exercises/week1/README.md) and its
[worksheet](exercises/week1/worksheet.md). The repository includes the runnable
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
under `exercises/week1/output/`.

## Install Python and create an environment

Use **Python 3.11 or newer**. Check `python3 --version` (Windows: `py --version`).
If Python is missing, install it from [python.org](https://www.python.org/downloads/).
Run the setup from the repository root. The virtual environment keeps course
packages separate from your other Python projects.

### macOS

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch
python -m pip install -e .
python tools/check_setup.py
python exercises/week1/starter.py
```

macOS uses the normal PyTorch package index. All tensors in Week 1 stay on CPU.

### Linux

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python tools/check_setup.py
python exercises/week1/starter.py
```

### Windows PowerShell

These commands use the environment directly and do not require changing
PowerShell's script-execution settings:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe tools/check_setup.py
.\.venv\Scripts\python.exe exercises/week1/starter.py
```

Use `.\.venv\Scripts\python.exe` in place of `python` in the worksheet commands.
Linux and Windows install CPU wheels to avoid unnecessary GPU dependencies.
See the [official PyTorch installer](https://pytorch.org/get-started/locally/) for
platform-specific alternatives.

## In a new terminal

From the repository root, activate the existing environment on macOS/Linux:

```sh
. .venv/bin/activate
```

On Windows, continue using `.\.venv\Scripts\python.exe` directly.

## Troubleshooting

- **`No module named dlexercises`:** activate the environment used for installation
  and run `python -m pip install -e .` from the repository root. Then check with
  `python tools/check_setup.py`. The script reports the interpreter and package
  location so you can identify an environment mismatch.
- **No plot window:** use `--save fit.png` instead of `--plot`; on Linux servers,
  set `MPLBACKEND=Agg` if needed. A graphical desktop is needed for a plot window.
- **No compatible PyTorch wheel:** check your Python version and platform against
  the official installer. Use an interpreter supported by the available wheel.
- **Quarto or Jupyter is missing:** both are optional for this initial exercise.
  Read the Markdown worksheet and run the Python script directly.

## Files and checks

| Path | Purpose |
|---|---|
| `exercises/week1/starter.py` | Manual parameter trials and guided fitting experiments |
| `exercises/week1/worksheet.md` | Questions and spaces for your explanations |
| `exercises/week1/worksheet.qmd` | Optional Quarto source of the same worksheet |
| `exercises/week1/figures/` | Lecture visuals accompanying the worksheet |
| `src/dlexercises/data.py` | Reproducible synthetic sensor data |
| `tools/check_setup.py` | Interpreter, dependency, data, and image-export check |
| `tools/smoke_check.py` | Run each experiment mode and check CLI/image behavior |

You can run the repository checks with:

```sh
python -m compileall -q src exercises tools
python tools/smoke_check.py
```

## Attribution and license

Week 1 material is adapted from HealthML/deep-learning-book at commit
`7919dce810877339178c33725e17959c83c02d47` by Christoph Lippert and contributors.
The dataset and equations match the course book and slides. See [LICENSE.md](LICENSE.md):
original source code is MIT; prose, exercises, and figures are CC BY-SA 4.0.
