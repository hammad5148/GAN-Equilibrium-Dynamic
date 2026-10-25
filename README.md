# GANs as a Minimax Game

This repository studies GAN training as a two-player game. It compares
Vanilla GAN and WGAN-GP with Adam, Optimistic Adam, and Negative Momentum,
using discriminator balance, mode coverage, gradient stability, Jacobian
diagnostics, and the Equilibrium Health Score (EHS).

The reusable implementation is in `src/gan_game_theory/`. The current checkout
contains the package, command-line scripts, tests, and exported experiment
outputs.

## Repository map

```text
src/gan_game_theory/       Reusable package
├── data/                  Dataset and imbalance loaders
├── models/                Generator, discriminator, mode classifier
├── optimizers/            Adam variants for game dynamics
├── diagnostics/           Balance, stability, EHS, Jacobian proxy
├── training/              Vanilla GAN and WGAN-GP training
├── analysis/              Metric loading and summary tables
└── visualization/         Result plotting

scripts/                   Command-line scripts
outputs/                   Metrics and figures from completed runs
tests/                     Fast smoke tests
```

## 1. Install

Python 3.10+ is recommended.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

If you prefer a requirements file:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

## 2. Run the tests

```powershell
python -m pytest -q
```

The tests only check model shapes and loss behavior; they do not train a GAN.

## 3. Run one experiment

This downloads Fashion-MNIST into `data/` on the first run and writes metrics
and generated samples under `outputs/`.

```powershell
python scripts\train.py --loss vanilla --optimizer adam --seed 42 --epochs 5 --device cpu
```

Available losses:

- `vanilla`
- `wgan_gp`

Available optimizers:

- `adam`
- `optimistic_adam`
- `negative_momentum`

Use `--device cuda` for an NVIDIA GPU or `--device auto` to select CUDA when it
is available.

## 4. Run the experiment grid

Run a short one-seed smoke test:

```powershell
python scripts\run_grid.py --smoke-test --device cpu
```

Run the full Fashion-MNIST 2×3 grid:

```powershell
python scripts\run_grid.py --device cuda
```

Run the imbalance or CIFAR-10 variants:

```powershell
python scripts\run_grid.py --imbalanced --device cuda
python scripts\run_grid.py --dataset CIFAR10 --device cuda
```

The full grid is computationally expensive. Start with one seed and a small
epoch count before launching it.

## 5. Regenerate summaries and plots

Use the existing metric JSON files without retraining:

```powershell
python scripts\plot_results.py
```

This updates `outputs/metrics/summary_table.json` and creates plots in
`outputs/figures/`.

## Results and Git policy

- `outputs/metrics/` contains tracked experiment metrics.
- `outputs/figures/` contains tracked plots and animations.
- `outputs/samples/` contains large generated sample grids and is ignored by Git.
- `data/`, checkpoints, Python caches, and build metadata are ignored.

Before committing:

```powershell
git status
git add .
git status
git commit -m "Prepare GAN game theory repository"
```

Then connect your GitHub repository and push:

```powershell
git remote add origin https://github.com/<your-user>/<your-repository>.git
git push -u origin main
```
