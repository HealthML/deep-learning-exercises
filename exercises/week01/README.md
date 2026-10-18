# Week 1: fitting a line

Use **uv with Python 3.11 and PyTorch on CPU**. Follow the
[installation instructions](../../README.md) first, then run all commands below
from the repository root (`deep-learning-exercises`). No book repository, GPU,
dataset download, Jupyter kernel, or Quarto installation is needed to run the code.

The experiment reproduces the lecture's synthetic sensor data, seed, 40/10/10
split, observed-minus-predicted residuals, mean squared loss, and parameter
order `[weight, bias]`.
Only the 40 training examples participate in fitting. The 10 validation and 10
test observations remain separate and are not used for parameter selection here.

## Check your installation

```sh
uv run --locked python -m compileall -q src exercises
uv run --locked python tools/check_setup.py
uv run --locked python exercises/week01/starter.py
```

The same `uv run --locked` commands work on macOS, Linux, and Windows.
The default run reports the PyTorch version, CPU device, float64 precision, and
training-set size. It starts with a zero slope and
zero bias. `compileall` checks Python syntax; it does not train a model.

## Classroom experiment

Open [worksheet.md](worksheet.md) and `starter.py`. Work in pairs: one student
predicts and records; the other operates the computer. Swap roles after three
manual trials. No earlier reading is assumed. The lecturer introduces each task.

Begin with the one-parameter model. Edit the constants at the top of the script
or use command-line options:

```sh
uv run --locked python exercises/week01/starter.py --w 0 --plot
uv run --locked python exercises/week01/starter.py --w 1 --plot
uv run --locked python exercises/week01/starter.py --w 2 --plot
uv run --locked python exercises/week01/starter.py --model affine --w 1.5 --b 0.5 --plot
```

A plot window opens with `--plot`. On a server or another machine without a
graphical display, save an image instead:

```sh
uv run --locked python exercises/week01/starter.py --w 1 --save fit.png
```

If downloads are still running after four minutes in class, pair with a working
machine and finish independent installation later.

## Guided fitting modes

The script contains runnable implementations for exploring the methods introduced
in the lecture. Predict what will change before using each next mode. The task
is to explain and compare their behavior; a completed program alone does not
answer the worksheet.

```sh
uv run --locked python exercises/week01/starter.py --model affine --method partials --w 0.5 --b 0
uv run --locked python exercises/week01/starter.py --model affine --method coordinate --steps 5
uv run --locked python exercises/week01/starter.py --model affine --method sgd --lr 0.025 --steps 40
uv run --locked python exercises/week01/starter.py --model affine --method batch --lr 0.1 --steps 40
uv run --locked python exercises/week01/starter.py --model affine --method lstsq
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
quarto render exercises/week01/worksheet.qmd --to html
```

The rendered HTML opens with a pure black (`#000000`) background, matching the
lecture slides and the book's dark appearance. Use the appearance toggle to
switch to a light page. Inline and display equations use native MathML, and the
styles and figure are embedded so the HTML can be viewed offline. A previously
saved light preference may override the initial dark appearance; use the toggle
to return to dark.
