# Week 5: Build, test, and train a small CNN

A pair lab: implement convolution and max pooling in NumPy, verify
them against PyTorch, build and train a small CNN, implement a residual block,
and investigate translation robustness. Read Lesson 5: CNNs, ResNets, and
Augmentation first. You need array indexing, Python loops, basic PyTorch modules,
classification loss, and backpropagation concepts.

## Before class

From the repository root, follow the main README's uv installation instructions:

```sh
uv sync --locked --group notebook
uv run --locked --group notebook jupyter lab exercises/week5/experiments.ipynb
```

In VS Code, select this repository's `.venv` interpreter. The lab uses the existing
NumPy, Matplotlib, and CPU PyTorch environment; no new dependencies, GPU, external
dataset, or book package is needed. L, T, and + inspection symbols are generated
locally. No Quarto installation is needed to run the notebook.

## During class

1. Start the [worksheet](worksheet.qmd) warm-up, then open `experiments.ipynb`.
2. Complete the NumPy convolution and pooling gaps; verify against PyTorch.
3. Complete `TinyCNN`, train it, and inspect the losses and predictions.
4. Complete the addition in `ResidualBlock`; test its output shape and explain
   the identity shortcut. Projection shortcuts are a follow-up question.
5. Compare centered and shifted validation accuracy before and after augmentation.
6. Freeze a written recommendation before enabling the held-out test cell; finish
   the individual worksheet reflection.

Switch driver after pooling. Write predictions before running comparisons.
The NumPy loops and CNN forward pass are supplied. Fill in one calculation per
NumPy operation, two convolution definitions, and the residual addition.
Unfinished tasks print reminders; they do not silently use supplied solutions.
Complete the TODO gaps and rerun downstream cells to obtain training results.
Training the optional residual classifier is additional practice within activity 5; the required residual work is implementation and behavior checks.
Keep personal notebook copies outside the checkout or under `exercises/week5/output/`
before pulling updates. Worked answers are in `worksheet_solution.qmd` and
`experiments_solution.ipynb` in this directory.

## File roles and checks

- `worksheet.qmd`: paper predictions, diagnosis, and final reflection.
- `experiments.ipynb`: student coding gaps, visible checks, and experiment records.
- `worksheet_solution.qmd`: worked paper calculations and example explanations.
- `experiments_solution.ipynb`: completed code and example notebook answers.
- `src/dlexercises/week05.py`: supplied data, augmentation, training, and plotting.
- `tools/check_week05.py`: execute delivered notebook cells and check data,
  translations, training gradients, checkpoints, and reproducibility. This is a
  technical check, not a completed student submission.

```sh
uv run --locked python tools/check_week05.py
```

To check a completed personal notebook, append its path to that command. Its
visible assertions then check the implementations as it executes. The technical
checker also works on the unfinished starter. Use `quarto preview` from the
repository root to preview the worksheet website.
