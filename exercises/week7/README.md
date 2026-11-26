# Week 7: Diffusion image generation

Two pair labs, one for each session of the week:

1. [diffusion.ipynb](diffusion.ipynb): build, train, and judge a small
   diffusion model.
2. [diffusion_sampling.ipynb](diffusion_sampling.ipynb): keep the trained
   network fixed and change the sampler. How many denoising steps does an image
   need?

Each notebook contains the theory, pen-and-paper questions, coding exercises,
and answer spaces; the [worksheet](worksheet.qmd) gives both session routes.
Familiarity with the Week 6 U-Net and the basic PyTorch training loop is
assumed. The notation follows
[Lesson 7: Diffusion Image Generation](https://healthml.github.io/deep-learning-book/chapters/07-diffusion/concepts.html)
in the course book, and every equation is introduced before you use it.

From the repository root:

```sh
uv sync --locked --group notebook
uv run --locked --group notebook python -m jupyterlab exercises/week7/diffusion.ipynb
```

For the second session, open `exercises/week7/diffusion_sampling.ipynb` in the
same way.

If your terminal has another checkout's environment activated, run `deactivate`
first (or open a fresh terminal). Launching via `python -m jupyterlab` also avoids
stale launcher paths if this checkout was moved or copied.

Choose the repository's `.venv` kernel. The notebooks use the existing NumPy,
Matplotlib, and CPU PyTorch dependencies. Data are generated locally; no extra
packages, data downloads, or GPU are needed. All training in a notebook takes
about a minute.

## Session 1: build a diffusion model

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

## Session 2: how many denoising steps does an image need?

`diffusion_sampling.ipynb` is a classroom version of
[Lab 7: How Many Denoising Steps Do We Need?](https://healthml.github.io/deep-learning-book/chapters/07-diffusion/lab.html)
in the course book. You do not need your notebook from Session 1: its parts
are supplied, completed, at the top. Compare them with your own code.

1. **Try a shortcut:** skip timesteps in the unchanged DDPM loop, watch it
   fail, and locate the error with an oracle that keeps a logbook.
2. **A step that can jump:** work out the DDIM step on paper, implement it,
   plan the jumps, and verify the sampler with the oracle.
3. **How few evaluations are enough?** Compare 64, 16 and 4 network evaluations
   with one frozen network and the same starting noise; find the breaking
   point and explain it.
4. **Old noise or fresh noise?** Fill the noise of every state from two
   sources, recover the DDPM sampler as one setting, and find out what decides
   which image appears.
5. **Extended exercise:** train a network with a label input and steer its
   disks to the left or the right with classifier-free guidance.
6. **Individual exit ticket.**

Activities 1 to 3 and the exit ticket are the core route. For the remaining
time choose activity 4 or activity 5. Complete the TODO assignments and rerun
cells in order. All cells run in the unfinished starter, with reminders for
sections that need your code. Switch driver after activity 2.

One network trains for 1800 updates at the start, about half a minute, and is
then frozen. Activity 5 trains a second network for about 20 seconds. As in
Session 1, numbers differ a little from one laptop to the next.

## Check

```sh
uv run --locked python tools/check_week07.py
uv run --locked python tools/check_week07_sampling.py
```

Each checker executes the cells of one notebook without opening Jupyter and
tests the supplied code: data, schedule, network, reference denoisers, and the
supplied samplers. On an unfinished starter it is a technical check, not a
completed student submission. Once your TODOs are complete, run it again: it
then also audits your own functions against independent constructions and
trains the networks (about a minute each). To check a personal copy, append its
path to the command.
