# Week 6: U-Net and transfer learning

Start with the [classroom worksheet](worksheet.qmd), then open
[transfer_learning_unet.ipynb](transfer_learning_unet.ipynb) for the pair lab.
The worksheet provides paper predictions and a final reflection; the notebook
contains theory, coding exercises, and experimental answer spaces. Shared
paper questions can be answered in either place. Familiarity with Week 5 CNNs
and the basic PyTorch training loop is assumed.

From the repository root:

```sh
uv sync --locked --group notebook
uv run --locked --group notebook python -m jupyterlab exercises/week6/transfer_learning_unet.ipynb
```

If your terminal has another checkout's environment activated, run `deactivate`
first (or open a fresh terminal). Launching via `python -m jupyterlab` also avoids
stale launcher paths if this checkout was moved or copied.

Choose the repository's `.venv` kernel. The notebook uses the existing NumPy,
Matplotlib, and CPU PyTorch dependencies. Data are generated locally; no extra
packages, data downloads, or GPU are needed.

## Session plan

1. **Build a small U-Net:** work out tensor shapes,
   explore encoder–decoder structure and upsampling, read our U-Net
   figure, and explain and implement skip connections,
   choose a segmentation loss, calculate IoU, and train a segmentation model
   before introducing transfer learning.
2. **What transfers?:** distinguish scratch training, freezing, and
   fine-tuning; choose what to reuse for a new task.
3. **Pretrain and reuse:** run a supplied shape classifier, load its
   encoder weights, and implement freezing.
4. **Compare approaches:** train scratch, frozen, and fine-tuned models;
   inspect masks and metrics; explain the comparison's limits.
5. **Remove the skips:** complete a decoder without skip connections, train it,
   and compare its masks and IoU with the earlier U-Net.

Complete the TODO assignments and rerun cells in order. All cells run in
the unfinished starter, with reminders for sections that need your code.
The final experiment runs after the TODOs are completed. Training, data, and
plotting helpers are supplied within the notebook. Switch driver after activity 2.
Keep personal copies under `exercises/week6/output/` or outside the checkout.

We will build the U-Net shown in `figures/image.png`.
Figures display offline. See [figure source notes](figures/README.md).

The first scratch-training demo uses 100 masks. Transfer learning uses 25 target
masks; the final comparison includes a 25-mask scratch baseline. The final metrics table
and prediction plots also include the earlier 100-mask scratch model. All models
are evaluated on the same 64 validation images.

Images vary in object size, position, contrast, and background lighting. Target
images have stronger noise than source images to make the comparison less trivial.
