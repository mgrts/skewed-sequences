# skewed-sequences

> Loss-function analysis for transformer neural networks on skewed & heavy-tailed time-series data.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/packaging-poetry-cyan.svg)](https://python-poetry.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Overview

This project benchmarks several loss functions—including a novel **Skewed Generalised
T (SGT) loss**—for multi-step time-series prediction with Transformer and LSTM
models. It covers:

| Loss | Class |
|------|-------|
| SGT (parametric) | `SGTLoss` |
| MSE | `torch.nn.MSELoss` |
| MAE | `torch.nn.L1Loss` |
| Cauchy | `CauchyLoss` |
| Huber | `HuberLoss` |
| Tukey bi-weight | `TukeyLoss` |

All custom loss functions extend `torch.nn.Module` with a standard
`forward(input, target)` interface (following the PyTorch convention where
`input` = predictions and `target` = ground-truth).  The `get_loss_function()`
factory in `train.py` instantiates the correct loss by name.

Experiments run on four **synthetic** SGT datasets (normal, heavy-tailed,
skewed, heavy-tailed-skewed) and four **real-world** datasets (OWID COVID-19,
LANL earthquakes, RVR US hospitalisations, Health & Fitness wearable data).

## Project structure

```
├── pyproject.toml          <- Project config & dependencies (Poetry)
├── poetry.lock             <- Locked dependency versions
├── Makefile                <- Convenience targets (install, test, lint, …)
├── Dockerfile              <- Multi-stage production image
├── docker-compose.yml      <- MLflow + scalable worker services
├── setup.cfg               <- Flake8 config
├── .pre-commit-config.yaml <- Pre-commit hook definitions
│
├── skewed_sequences/       <- Installable Python package
│   ├── cli.py              <- Unified Typer CLI entry point (`skseq`)
│   ├── config.py           <- Paths, constants, experiment configs
│   ├── metrics.py          <- Dispersion-scaling & skewness metrics (MAD, kappa)
│   ├── plots.py            <- Boxplot comparison visualisations
│   ├── visualize_data.py   <- Dataset sample visualisation
│   │
│   ├── data/               <- Dataset loaders & generators
│   │   ├── synthetic/      <- SGT-distributed synthetic data
│   │   ├── owid_covid/     <- Our World in Data COVID-19
│   │   ├── lanl/           <- LANL earthquake catalogue
│   │   ├── rvr_us/         <- RVR US hospitalisation data
│   │   └── health_fitness/ <- Wearable health & fitness data
│   │
│   ├── modeling/           <- Training & evaluation
│   │   ├── models.py       <- TransformerWithPE, LSTM architectures
│   │   ├── loss_functions.py <- SGT, Cauchy, Huber, Tukey losses
│   │   ├── train.py        <- Training CLI / entry point
│   │   ├── trainer.py      <- Training loop logic
│   │   ├── evaluation.py   <- Model evaluation helpers
│   │   ├── data_processing.py <- SequenceDataset & dataloaders
│   │   ├── visualize.py    <- Prediction plotting (with zoom)
│   │   └── utils.py        <- Misc modelling utilities
│   │
│   └── experiments/        <- Reproducible experiment runners
│       ├── run_experiments/ <- Per-dataset experiment scripts
│       ├── calculate_metrics.py
│       └── calculate_dispersion_scaling.py
│
├── tests/                  <- Pytest test suite (77 tests)
├── data/                   <- Raw / interim / processed / external data
├── models/                 <- Saved model artefacts
├── mlruns/                 <- MLflow tracking store
├── reports/                <- Generated figures & reports
├── notebooks/              <- Exploratory Jupyter notebooks
└── references/             <- Papers, manuals, data dictionaries
```

### Key design decisions

- **Lazy CLI imports** — `cli.py` uses a `_LazyTyper` proxy pattern so that
  `poetry run skseq --help` responds instantly. Heavy dependencies (scipy,
  sklearn, torch, mlflow) are only imported when a sub-command is invoked.
- **Multi-step prediction** — the system predicts `OUTPUT_LENGTH` (default 3)
  future time steps rather than a single step. This is configured in
  `config.py` and used consistently across all training configs, experiment
  runners, CLI defaults, and model architectures.
- **Optimised data pipeline** — training data is pre-tensorised once on dataset
  creation (zero-copy slicing in `__getitem__`), metrics are collected inline
  during train/eval passes (no redundant data iterations), and DataLoaders use
  `pin_memory` and `persistent_workers` for GPU transfer acceleration.
- **MLflow tracking** — all training runs are logged to MLflow (file-based
  store under `mlruns/`). The Docker Compose stack can optionally run a
  centralised MLflow tracking server.

## Getting started

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.11+ |
| [Poetry](https://python-poetry.org/docs/#installation) | ≥ 2.0 |
| Docker *(optional)* | 24+ |

### Install

```bash
# Clone the repository
git clone <repo-url>
cd skewed-sequences

# Install all dependencies + the package itself
poetry install

# Install pre-commit hooks
poetry run pre-commit install
```

After installation, the `skseq` CLI is available via `poetry run`:

```bash
poetry run skseq --help
```

### Running your first experiment

**1. Generate synthetic data**

```bash
poetry run skseq data generate-synthetic main
```

This creates four synthetic SGT-distributed datasets (normal, heavy-tailed,
skewed, heavy-tailed-skewed) under `data/processed/`.

**2. Train a model**

```bash
poetry run skseq train main --loss-type mse
```

Training logs are recorded to `mlruns/` via MLflow. Model checkpoints are saved
to `models/`.

**3. Run a full experiment suite**

```bash
poetry run skseq experiments run-synthetic main
```

This trains all configured loss functions (SGT variants, MSE, MAE, Cauchy,
Huber, Tukey) across multiple runs and logs results to MLflow.

For faster iteration, tune dataset size, stride, and training params:

```bash
poetry run skseq experiments run-synthetic main \
  --n-sequences 2000 --n-runs 3 --stride 10 \
  --batch-size 64 --num-epochs 50 --early-stopping-patience 10 \
  --num-workers 4
```

**4. Visualise results**

```bash
# Sample sequences from each dataset
poetry run skseq visualize synthetic

# Boxplot comparison of metrics across loss functions
poetry run skseq plots main
```

## CLI reference

All commands are invoked via `poetry run skseq`:

```
poetry run skseq [OPTIONS] COMMAND [ARGS]...
```

### Top-level commands

| Command | Description |
|---------|-------------|
| `data` | Dataset generation and preprocessing |
| `train` | Model training |
| `visualize` | Dataset sample visualisation |
| `plots` | Boxplot metric comparisons |
| `experiments` | Full experiment suites |

### `skseq data`

| Sub-command | Description |
|-------------|-------------|
| `generate-synthetic` | Generate synthetic SGT-distributed data |
| `download-owid` | Download OWID COVID-19 CSV |
| `process-owid` | Process OWID COVID-19 dataset into sequences |
| `process-lanl` | Process LANL earthquake dataset |
| `process-rvr` | Process RVR US hospitalisation dataset |
| `process-health-fitness` | Process health & fitness wearable data |

### `skseq train`

```bash
poetry run skseq train main --loss-type sgt --sgt-loss-lambda 0.002 --sgt-loss-q 1.001
poetry run skseq train main --loss-type mse
```

Run `poetry run skseq train main --help` for the full list of options including
`--loss-type`, `--sgt-loss-lambda`, `--sgt-loss-q`, `--sgt-loss-sigma`, and
`--output-length`.

### `skseq visualize`

| Sub-command | Description |
|-------------|-------------|
| `synthetic` | Plot synthetic SGT sequences |
| `real` | Plot real-world dataset sequences |
| `variants` | Compare dataset variants side-by-side |

### `skseq plots`

```bash
poetry run skseq plots main
```

### `skseq experiments`

| Sub-command | Description |
|-------------|-------------|
| `run-synthetic` | Synthetic SGT datasets |
| `run-lanl` | LANL earthquake data |
| `run-owid` | OWID COVID-19 data |
| `run-rvr` | RVR US hospitalisation data |
| `dispersion-scaling` | Compute dispersion-scaling exponents |
| `metrics` | Compute dataset-level statistical metrics |

```bash
poetry run skseq experiments run-synthetic main
poetry run skseq experiments dispersion-scaling main
```

All experiment commands accept these common options:

| Option | Default | Description |
|--------|---------|-------------|
| `--n-runs` | 10 | Repetitions per configuration |
| `--stride` | 1 | Sliding window stride (higher = fewer samples) |
| `--batch-size` | 32 | Training batch size |
| `--num-epochs` | 100 | Maximum training epochs |
| `--early-stopping-patience` | 20 | Epochs without improvement before stopping |
| `--num-workers` | 0 | DataLoader worker processes |

The `run-synthetic` command additionally accepts `--n-sequences` (default 10000)
to control synthetic dataset size.

## Docker

A multi-stage Dockerfile and `docker-compose.yml` are provided for
reproducible, scalable execution.

The Dockerfile uses a two-stage build: a **builder** stage installs Poetry,
resolves dependencies and builds a wheel; a **runtime** stage copies only the
installed packages into a lean image.

The Compose stack defines two services: **mlflow** (tracking server on port
5000) and **worker** (the main `skseq` image, scalable via `--scale worker=N`).
Both share volume mounts for `data/`, `mlruns/`, `models/`, and `reports/`.

```bash
# Build the image
make docker-build    # or: docker build -t skewed-sequences .

# Quick test
docker run --rm skewed-sequences --help

# Run with MLflow tracking server + parallel workers
docker compose up --scale worker=4

# Ad-hoc experiment
docker compose run worker experiments run-synthetic main
```

## Development

```bash
make install     # poetry install
make format      # Auto-format with black + isort
make lint        # Check style (flake8, isort, black)
make test        # Run pytest suite
make pre-commit  # Run all pre-commit hooks
```

### Pre-commit hooks

| Hook | Purpose |
|------|---------|
| trailing-whitespace | Strip trailing spaces |
| end-of-file-fixer | Ensure files end with newline |
| check-yaml / check-toml | Validate config files |
| detect-private-key | Prevent accidental key commits |
| isort | Sort imports |
| black | Code formatting (line-length 99) |
| flake8 | Linting |

### Testing

The test suite (77 tests) lives in `tests/`:

| Module | What it tests |
|--------|---------------|
| `test_config.py` | Config constants, paths, experiment configs |
| `test_metrics.py` | MAD, kappa, skewness, dispersion scaling |
| `test_loss_functions.py` | All custom loss functions (output shape, gradients, edge cases) |
| `test_models.py` | Transformer & LSTM forward/infer shapes |
| `test_data_processing.py` | SequenceDataset, dataloader creation |
| `test_generate_data.py` | SGT distribution, kernels, sequence generation |
| `test_cli.py` | CLI help output for all sub-commands |
| `test_train.py` | Loss function factory & forward pass |

```bash
poetry run pytest          # or: make test
```

## Configuration

All experiment parameters live in `skewed_sequences/config.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `SEQUENCE_LENGTH` | 300 | Input sequence length |
| `CONTEXT_LENGTH` | 200 | Context window for model input |
| `OUTPUT_LENGTH` | 3 | Multi-step prediction horizon |
| `STRIDE` | 1 | Sliding window stride |
| `N_RUNS` | 10 | Repetitions per experiment |
| `SEED` | 927 | Random seed |

- **`SYNTHETIC_DATA_CONFIGS`** — defines the four synthetic dataset variants
  (λ, q, σ combinations)
- **`TRAINING_CONFIGS`** — 17 training configurations covering SGT parameter
  sweeps + baseline losses (MSE, MAE, Cauchy, Huber, Tukey)
- **`SGT_LOSS_LAMBDAS`** — λ values for the SGT loss lambda sweep experiment

## License

See [LICENSE](LICENSE) for details.
