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
| Charbonnier | `CharbonnierLoss` |

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
│       ├── run_experiments/ <- Per-dataset experiment scripts (+ shared _runner.py)
│       ├── collect_results.py     <- MLflow → results CSV
│       ├── aggregate_results.py   <- replicate summary + significance
│       ├── calculate_metrics.py
│       └── calculate_dispersion_scaling.py
│
├── tests/                  <- Pytest test suite (242 tests)
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
- **Single-step autoregressive prediction** — the system predicts
  `OUTPUT_LENGTH` (default 1) future time step using the previous context
  window as input. This is configured in `config.py` and used consistently
  across all training configs, experiment runners, CLI defaults, and model
  architectures.
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
  --batch-size 64 --num-epochs 50 --early-stopping-patience 10
```

Keep `--num-workers` at its default of 0: the datasets are in-memory tensors, so
DataLoader workers only add overhead (4 workers doubled the epoch time on a GPU pod).

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
| `download-owid` | Download the daily OWID/JHU `new_cases.csv` (wide, cadence-validated) |
| `process-owid` | Process OWID COVID-19 dataset into sequences (`--all-locations` for a rule-based country set) |
| `process-lanl` | Process LANL earthquake dataset |
| `download-rvr` | Download the CDC RVR hospitalisation timeseries via the SODA API (row-count verified) |
| `process-rvr` | Process RVR US hospitalisation dataset (aggregate jurisdictions dropped) |
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
| `run-head-sweep` | Multi-head attention study (heavy-tailed synthetic, fixed width) |
| `run-lambda-sweep` | Fine skew (λ) grid appended to the skewed synthetic experiments |
| `collect-results` | Collect MLflow runs into `reports/experiment_results.csv` |
| `aggregate-results` | Replicate summary stats + paired SGT-vs-baseline, anchor and λ-effect tests |
| `increment-fit` | MLE of SGT (λ, q) on each dataset's one-step increments (parameter guidance) |
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
| `--resume/--no-resume` | resume | Reuse the seed already logged for an experiment and skip configs that already have a FINISHED run |
| `--first-run` | 1 | Start at this run index (split one dataset's seeds across parallel processes) |

The `run-synthetic`, `run-head-sweep` and `run-lambda-sweep` commands additionally
accept `--n-sequences` and `--stride`, defaulting to `SYNTHETIC_N_SEQUENCES` (1000) and
`SYNTHETIC_STRIDE` (5) from `config.py` — the three must agree because the λ sweep
appends to the synthetic experiments.

### Turnkey sweep (resumable)

```bash
# all stages, or a subset: STAGES=owid,rvr,lambda,collect
nohup poetry run bash scripts/run_sweep.sh > sweep.log 2>&1 &
```

Every runner is called with `--resume`, so a killed sweep is simply re-launched
with the same command. Never delete `mlruns.db` between launches — it holds the
finished runs and their seeds.

### Faster single-process training

```bash
SKSEQ_COMPILE=reduce-overhead poetry run skseq experiments run-owid main   # torch.compile + CUDA graphs
SKSEQ_DEVICE=cpu poetry run skseq train main --loss-type mse               # force a device
```

The model is small enough that a GPU spends most of each batch launching tiny kernels;
`torch.compile` fuses them (`1`/`default`) or replays them as CUDA graphs
(`reduce-overhead`). Only the training `forward` is compiled; checkpoints and `infer` are
unchanged, and the mode is logged as the `compile_mode` MLflow param.

### Parallel slots on one GPU

```bash
source scripts/mps.sh start                          # NVIDIA MPS so processes share the GPU
PARTS=7 MAX_ALIVE=6 bash scripts/launch_parallel.sh  # cut into 1 + 3*PARTS slots, run 6 at once
bash scripts/status_parallel.sh                      # progress per slot + GPU utilisation
bash scripts/status_parallel.sh --merge              # collect-results over every store when done
```

Re-run the launcher to top up as slots finish (or let the notebook watcher do it). Size
`MAX_ALIVE` by the pod's CPU quota (`cat /sys/fs/cgroup/cpu.max`): every training process
needs about one core.

The launcher keeps whatever a dataset already has in the main `mlruns.db` there (one
sequential resuming process) and cuts the remaining run indices into `PARTS` range
slots, each with its own `SKSEQ_PROJ_ROOT` (own `mlruns.db`, `data/`, `reports/`) and
the shared code and virtualenv. Re-run it to resume dead slots; it refuses a different
`PARTS` once slots exist. `run-rvr --time-series` (repeatable) runs one series per
process.

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

The test suite (242 tests) lives in `tests/`:

| Module | What it tests |
|--------|---------------|
| `test_config.py` | Config constants, paths, experiment configs (incl. the skew sweep) |
| `test_metrics.py` | MAD, kappa, skewness, dispersion scaling |
| `test_loss_functions.py` | All custom loss functions (output shape, gradients, edge cases) |
| `test_sgt_consistency.py` | The three SGT implementations agree numerically |
| `test_models.py` | Transformer & LSTM forward/infer shapes + no-target-leak |
| `test_data_processing.py` | SequenceDataset, dataloader creation |
| `test_data_common.py` | Shared loader slice/scale/stack helpers |
| `test_generate_data.py` | SGT distribution, kernels, sequence generation |
| `test_evaluation.py` | Sliding-window prediction logging |
| `test_cli.py` | CLI help output for all sub-commands |
| `test_train.py` | Loss function factory & forward pass |
| `test_runner.py` | Shared grid-runner helper kwarg expansion |
| `test_mlflow_contract.py` | Producer/consumer MLflow key contract |
| `test_collect_results.py` | MLflow → CSV collection |
| `test_aggregate_results.py` | Replicate aggregation + significance (+ anchors, λ effect) |
| `test_utils.py` | Residual-scale guard, metric helpers |
| `test_owid_dataset.py` / `test_owid_load_data.py` | Daily wide-file loader + download validation |
| `test_rvr_dataset.py` / `test_rvr_load_data.py` | Aggregate filter, SODA download + row-count check |
| `test_head_attention.py` / `test_lambda_sweep.py` | Head study and λ sub-sweep runners |
| `test_increment_fit.py` | SGT (λ, q) increment fit command |

```bash
poetry run pytest          # or: make test
```

## Configuration

All experiment parameters live in `skewed_sequences/config.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `SEQUENCE_LENGTH` | 300 | Input sequence length |
| `CONTEXT_LENGTH` | 200 | Context window for model input |
| `OUTPUT_LENGTH` | 1 | Single-step prediction horizon |
| `STRIDE` | 1 | Sliding window stride |
| `N_RUNS` | 10 | Repetitions per experiment |
| `SYNTHETIC_N_SEQUENCES` | 1000 | Synthetic sequences per dataset (all synthetic runners) |
| `SYNTHETIC_STRIDE` | 5 | Window stride for the synthetic runners |
| `SEED` | 927 | Random seed |
| `MODEL_TYPES` | `(transformer,)` | Architectures swept by every runner (LSTM retained, not swept) |

- **`SYNTHETIC_DATA_CONFIGS`** — defines the four synthetic dataset variants
  (λ, q, σ, kernel_size combinations)
- **`TRAINING_CONFIGS`** — 36 training configurations: SGT parameter sweeps
  (26 symmetric + 4 skewed nonzero-λ) + 6 baseline losses (MSE, MAE, Cauchy,
  Huber, Tukey, Charbonnier)
- **`LAMBDA_SWEEP_CONFIGS`** — 12 fine-skew SGT configurations (λ ∈ {0.1, 0.2, 0.3}
  at p ∈ {2, 1.5} × q ∈ {2.5, 10}) for `run-lambda-sweep`
- **`SGT_LOSS_LAMBDAS`** — λ values for the legacy `lambdas.py` script

## License

See [LICENSE](LICENSE) for details.
