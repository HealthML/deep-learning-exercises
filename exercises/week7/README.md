# Week 7: Diffusion image generation

Open [diffusion.ipynb](diffusion.ipynb) for the pair lab. The notebook contains
the theory, pen-and-paper questions, coding exercises, and answer spaces; the
[worksheet](worksheet.qmd) gives the session route. Familiarity with the Week 6
U-Net and the basic PyTorch training loop is assumed. The notation follows
[Lesson 7: Diffusion Image Generation](https://healthml.github.io/deep-learning-book/chapters/07-diffusion/concepts.html)
in the course book, and every equation is introduced before you use it.

From the repository root:

```sh
uv sync --locked --group notebook
uv run --locked --group notebook python -m jupyterlab exercises/week7/diffusion.ipynb
```

If your terminal has another checkout's environment activated, run `deactivate`
first (or open a fresh terminal). Launching via `python -m jupyterlab` also avoids
stale launcher paths if this checkout was moved or copied.

Choose the repository's `.venv` kernel. The notebook uses the existing NumPy,
Matplotlib, and CPU PyTorch dependencies. Data are generated locally; no extra
packages, data downloads, or GPU are needed. All training takes about a minute.

## Session plan

1. **From predicting to generating:** meet the disk images and see why sampling
   every pixel independently does not produce a disk.
2. **The forward process:** work out two noising steps on paper, implement the
   direct noising equation, and compare two noise schedules.
3. **Learning to denoise:** implement the clean-image estimate and the
   noise-prediction loss; test them with three reference denoisers.
4. **A U-Net that knows the noise level:** add the timestep to every block.
5. **Train across all noise levels:** run 200 updates and read the validation
   error separately at every noise level.
6. **The reverse process:** implement one DDPM step, test it with a perfect
   denoiser, generate images, and decide whether the flaws come from the sampler
   or from training. Then train longer and remove the fresh noise.
7. **Judge the generator:** look for copies of training images, compare size
   distributions, and write an individual exit ticket.
8. **Extended exercise:** fill in a hidden image region without retraining.

Complete the TODO assignments and rerun cells in order. All cells run in the
unfinished starter, with reminders for sections that need your code. Data,
network, training loop, and plotting helpers are supplied within the notebook.
Switch driver after activity 3. Keep personal copies under
`exercises/week7/output/` or outside the checkout.

The network trains on 256 synthetic 16×16 disk images. 64 independent validation
images provide the error curves and the reference for judging generated images.
Tiny numerical differences between computers grow during training, so counts and
error values differ a little from one laptop to the next. Compare with your
neighbours.

The overview figure displays offline. See [figure source notes](figures/README.md).

## Check

```sh
uv run --locked python tools/check_week07.py
```

The checker executes the notebook's cells without opening Jupyter and tests the
supplied data, schedule, reference denoisers, and network. On the unfinished
starter it is a technical check, not a completed student submission. Once your
TODOs are complete, run it again: it then also audits your forward process,
loss, time input, sampler, and inpainting, and trains the network (about a
minute). To check a personal copy, append its path to the command.

Next step in the course book:
[Lab 7: How Many Denoising Steps Do We Need?](https://healthml.github.io/deep-learning-book/chapters/07-diffusion/lab.html)
