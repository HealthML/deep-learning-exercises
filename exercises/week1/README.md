# Week 1: fitting a line

Use **Python 3.11 or newer and PyTorch on CPU**. Follow the platform-specific
[installation instructions](../../README.md) first, then run all commands below
from the repository root (`deep-learning-exercises`). No book repository, GPU,
dataset download, Jupyter kernel, or Quarto installation is needed to run the code.

The experiment reproduces the lecture's synthetic sensor data, seed, 40/10/10
split, residual sign, mean squared loss, and parameter order `[weight, bias]`.
Only the 40 training examples participate in fitting. The 10 validation and 10
test observations remain separate and are not used for parameter selection here.

## Check your installation

```sh
python -m compileall -q src exercises
python tools/check_setup.py
python exercises/week1/starter.py
```

On Windows, use `.\.venv\Scripts\python.exe` instead of `python` if you have not
activated the environment. The default run reports the PyTorch version, CPU
device, float64 precision, and training-set size. It starts with a zero slope and
zero bias. `compileall` checks Python syntax; it does not train a model.

## Classroom experiment

Open [worksheet.md](worksheet.md) and `starter.py`. Work in pairs: one student
predicts and records; the other operates the computer. Swap roles after three
manual trials. No earlier reading is assumed. The lecturer introduces each task.

Begin with the one-parameter model. Edit the constants at the top of the script
or use command-line options:

```sh
python exercises/week1/starter.py --w 0 --plot
python exercises/week1/starter.py --w 1 --plot
python exercises/week1/starter.py --w 2 --plot
python exercises/week1/starter.py --model affine --w 1.5 --b 0.5 --plot
```

A plot window opens with `--plot`. On a server or another machine without a
graphical display, save an image instead:

```sh
python exercises/week1/starter.py --w 1 --save fit.png
```

If downloads are still running after four minutes in class, pair with a working
machine and finish independent installation later.

## Guided fitting modes

The script contains runnable implementations for exploring the methods introduced
in the lecture. Predict what will change before using each next mode. The task
is to explain and compare their behavior; a completed program alone does not
answer the worksheet.

```sh
python exercises/week1/starter.py --model affine --method partials --w 0.5 --b 0
python exercises/week1/starter.py --model affine --method coordinate --steps 5
python exercises/week1/starter.py --model affine --method sgd --lr 0.025 --steps 40
python exercises/week1/starter.py --model affine --method batch --lr 0.1 --steps 40
python exercises/week1/starter.py --model affine --method lstsq
```

One SGD update processes one example; one batch update processes all 40. Thus 40
SGD updates and one batch update process the same number of examples, although
their trajectories and overhead differ. Use full training MSE to compare progress:
the sampled example can differ between consecutive SGD updates.

## Lecture figures

These figures reproduce the data and visuals used in the lecture:

- [One-parameter lines](figures/one-parameter.svg)
- [Loss with fixed bias](figures/loss-slice.svg)
- [Two-dimensional gradient](figures/gradient.svg)
- [Fitting trajectories](figures/methods.svg)

The worksheet is available as Markdown for reading on GitHub and as editable
[Quarto source](worksheet.qmd). Rendering the worksheet is optional:

```sh
quarto render exercises/week1/worksheet.qmd --to html
```
