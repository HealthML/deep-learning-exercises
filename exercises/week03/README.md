# Week 3: Diagnose and Improve Training

A 95-minute classroom session in a guided notebook, including discussion and help. Read the book's
[Chapter 3 concepts](https://healthml.github.io/deep-learning-book/chapters/03-optimization-regularization/concepts.html)
first. Optional additional material is not a prerequisite.

## Before class

From the repository root, follow the main README's uv installation instructions,
then run:

```sh
uv sync --locked --group notebook
uv run --locked --group notebook jupyter lab exercises/week03/optimization.ipynb
```

In VS Code, open the notebook and choose this repository's `.venv` interpreter.
The supplied code uses NumPy, Matplotlib, and CPU PyTorch. No GPU, data download,
book package, or Quarto installation is needed. Synthetic data are generated in
the notebook, with fixed splits and training-only preprocessing.

## During class

1. Complete activity 1 on the [online worksheet](https://healthml.github.io/deep-learning-exercises/exercises/week03/worksheet.html).
2. Follow activities 2–4 in `optimization.ipynb`, running cells from top to bottom.
   Record predictions and explanations in its Markdown response cells.
3. Return to worksheet activity 5 for diagnosis and individual reflection.

Work in pairs and alternate who runs the notebook. The update warm-up prints a reminder if left unfinished, so it does not block
the supplied experiments. Try completing it before continuing. All other code is supplied; change the marked settings.
Keep a personal notebook copy outside the checkout or under `exercises/week03/output/`
before pulling repository updates.

## File roles

- `worksheet.qmd`: pen-and-paper activities and source for the website page.
- `optimization.ipynb`: all coding prompts, experiments, and response spaces.
- `figures/diagnostic-runs.svg`: synthetic scenarios for the final paper task.

Worked answers will be released after the
teaching week; the worksheet URL will remain the same.

To check the delivered notebook and its numerical controls:

```sh
uv run --locked python tools/check_week03.py
```

The checker executes the supplied Python cells without opening Jupyter and
checks mathematical invariants. This is a technical check, not a completed
student submission. To preview the worksheet, use `quarto preview` from the root.
