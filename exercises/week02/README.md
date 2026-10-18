# Week 2: Logistic regression

Start with the [worksheet](worksheet.md), then open
[logistic_regression.ipynb](logistic_regression.ipynb). The notebook contains
all runnable experiments for two 90-minute sessions. Read the book's
[Section 8: concepts](https://healthml.github.io/deep-learning-book/chapters/02-linear-classification/concepts.html),
[Section 9: lab](https://healthml.github.io/deep-learning-book/chapters/02-linear-classification/lab.html),
and optional [Section 10: additional material](https://healthml.github.io/deep-learning-book/chapters/02-linear-classification/additional-material.html).

From the repository root, after installing uv as described in the main README:

```sh
uv sync --locked --group notebook
uv run --locked --group notebook python tools/check_week02.py
uv run --locked --group notebook jupyter lab exercises/week02/logistic_regression.ipynb
```

In Jupyter, select the Python kernel from this repository's `.venv` if prompted.
In VS Code select `.venv` as the notebook interpreter. Keep your work in a copy
under `exercises/week02/output/` or save a copy outside the checkout before
updating the repository.

Work in pairs and swap driver/explainer after each lettered task. Make a
prediction before each experiment. All supplied code runs as delivered; write
your answers in the **Your record** Markdown cells. Optional coding challenges
are clearly labeled and do not block the core experiments. Independent exit
tickets check your own understanding.

The exact two-feature WDBC teaching subset is bundled, so no data download or
book package is needed. NumPy is used for transparent small matrix operations;
the supplied helpers live in `src/dlexercises/week02.py`. PyTorch checks use
the same CPU environment as Week 1. All code uses `theta=(b,w1,w2)` and the
book's seed-8 training/validation/test split. Positive always means malignant.

**Test protection:** the last cell has `OPEN_TEST=False`, so Run All does not
inspect test predictions or metrics. Change it to `True` only after completing
the freeze checkpoint and validation explanations. The fixed primary model is
unpenalized MLE at threshold 0.5; the short threshold and L1/L2 explorations
do not replace it.

No Quarto installation is needed for class. The online worksheet is generated
from `worksheet.qmd`, which includes the readable Markdown source. To verify
the student experiments without a notebook server:

```sh
uv run --locked python tools/check_week02.py
```

Attribution and data license: see the worksheet's Sources and
`src/dlexercises/datasets/wdbc/{README.md,manifest.yml}`. The historical dataset
is used for a teaching experiment rather than a validated medical system.
