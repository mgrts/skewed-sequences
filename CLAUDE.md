# CLAUDE.md — skewed-sequences

Research codebase for **skewed / heavy-tailed time-series forecasting**. It trains
sequence models (a positional-encoded Transformer and an LSTM) on synthetic and
real datasets using a family of robust/asymmetric losses — chiefly the **Skewed
Generalized T (SGT)** loss — and compares them against classical baselines (MSE,
MAE, Cauchy, Huber, Tukey). Experiments run as in-process grid sweeps, are logged
to MLflow (sqlite `mlruns.db`), and aggregated into `reports/` CSVs/figures.
Everything is driven through a single `skseq` Typer CLI.

## Package map

- `skewed_sequences/cli.py` — single Typer entry point (`skseq`). Registers all
  sub-apps **lazily** via `_register_lazy` / `_LazyTyper`. (`_lazy_typer` near the
  top is dead/legacy — not used.)
- `skewed_sequences/config.py` — single source of truth for constants and
  experiment grids. Eagerly imported by `__init__.py`, so it runs on **every** CLI
  call — keep it lightweight.
- `skewed_sequences/modeling/`
  - `models.py` — `TransformerWithPE`, `LSTM`, `PositionalEncoding`. Both models
    expose `forward(src, tgt)` (teacher-forced) and `infer(src, tgt_len)` (autoregressive).
  - `loss_functions.py` — `SGTLoss` (asymmetric) + `CauchyLoss` / `HuberLoss` / `TukeyLoss` (symmetric).
  - `train.py` — CLI entry (`main`) + `get_loss_function` factory + device selection + MLflow run.
  - `trainer.py` — epoch loop, checkpointing, per-epoch MLflow metric logging.
  - `utils.py` — `set_seed`, `train_epoch`, `evaluate`, `compute_metrics`, `EarlyStopping`.
  - `data_processing.py` — `SlidingWindowDataset`, `create_dataloaders`.
  - `evaluation.py` — `sliding_window_predictions`, `log_val_predictions`.
- `skewed_sequences/data/` — per-dataset loaders (`synthetic/generate_data.py`,
  `owid_covid/`, `lanl/`, `rvr_us/`, `health_fitness/`). Each is a Typer app writing a `.npy`.
- `skewed_sequences/experiments/`
  - `run_experiments/{synthetic,lanl,owid_covid,rvr_us}_data.py` — grid-sweep runners (Typer).
    `lambdas.py` is an **unregistered** run-by-hand script (no `_register_lazy` entry).
  - `collect_results.py` — reads MLflow back into `reports/experiment_results.csv`.
  - `calculate_metrics.py`, `calculate_dispersion_scaling.py` — analysis scripts.
- `skewed_sequences/visualization/` — `style.py`, `predictions.py`, `plots.py`,
  `visualize_data.py`, `visualize_losses.py` (NumPy reimplementation of the SGT loss).
- `skewed_sequences/metrics.py` — skewness / kappa / dispersion metrics.
- `tests/` — one `test_<module>.py` per source module; plain pytest, no `conftest.py` (**98 tests**).

## How to run

```bash
poetry install
skseq --help

# Single-command modules expose a `main` command:
skseq train main --loss-type mse
skseq data generate-synthetic main
skseq experiments run-synthetic main
skseq experiments collect-results main      # MLflow -> reports/experiment_results.csv
skseq visualize-losses main

# `visualize` and `plots` have named subcommands (no `main`):
skseq visualize synthetic        # also: real | variants
skseq plots synthetic            # also: real
```

Make targets: `make test` (pytest) · `make lint` (flake8 + isort --check + black --check) ·
`make format` (isort + black) · `make pre-commit` · `make docker-build`.

## CRITICAL invariants

1. **Lazy CLI imports.** `cli.py` top-level imports are ONLY `importlib`, `typing`,
   `typer`. `config.py` top-level is ONLY `pathlib` / `dotenv` / `loguru` / optional
   `tqdm`. **Never** add `torch` / `scipy` / `sklearn` / `mlflow` / `pandas` / `matplotlib`
   at the top of either — `config.py` is eagerly imported on every `skseq` call, and
   `cli.py`'s whole purpose is to defer heavy deps until a sub-command runs. No test
   catches a regression here. Verify in a fresh subprocess:
   `python -c "import sys, skewed_sequences.cli; print([m for m in ('torch','mlflow','scipy','sklearn') if m in sys.modules])"` → expect `[]`.
2. **Every registered command module defines top-level `app = typer.Typer(...)`.**
   `_register_lazy` resolves `getattr(mod, "app")` only at *invocation* time, so a
   wrong dotted path/attr fails when the command runs, not at `--help`. Smoke-test
   new commands with `skseq <group> <cmd> --help`.
3. **`OUTPUT_LENGTH=1` single-step horizon.** Every `TRAINING_CONFIGS` dict MUST
   carry an explicit `output_length` key, because all 4 runners read
   `train_config.get("output_length", 5)` — the magic-`5` fallback silently switches
   the horizon if the key is dropped (no error). `test_config.py` asserts
   `output_length == OUTPUT_LENGTH` for every entry.
4. **Window must fit the data.** `CONTEXT_LENGTH(200) + OUTPUT_LENGTH(1) <= SEQUENCE_LENGTH(300)`.
   `train.py` asserts `context_length + output_length <= data.shape[1]` — the ONLY
   guard against a silently-empty `SlidingWindowDataset`
   (`n_windows = (T - window_total)//stride + 1`). Replicate it in any new caller of
   `create_dataloaders`.
5. **`config.py` is the single source of truth.** `SEED=927, SEQUENCE_LENGTH=300,
   CONTEXT_LENGTH=200, OUTPUT_LENGTH=1, STRIDE=1, N_RUNS=10, TRACKING_URI`. These are
   imported as Typer parameter defaults. Runner training-loop literals
   `batch_size=32 / num_epochs=100 / early_stopping_patience=20 / num_workers=0` are
   hard-coded in all 5 runner files (synthetic/lanl/owid/rvr + `lambdas.py`) and must
   stay in sync with `train.main` defaults. (The `output_length` magic-5 fallback in
   invariant #3 lives only in the 4 grid runners; `lambdas.py` passes `OUTPUT_LENGTH`
   directly.) Edit `config.py`, never inline copies. Tests pin counts (**4** synthetic
   configs, **31** training configs) and the constant literals.
6. **MLflow key contract.** The param/metric string keys logged in `train.main` are an
   untyped contract read by `collect_results.py` (literal `.get()`) and mocked in
   `test_collect_results.py`. The seed is logged as `random_state` (NOT `seed`);
   summary metrics are `best_{train,val,test}_{smape,mape}`. A rename on either side
   silently produces NaN CSV columns with no failing test.
7. **Seed/reproducibility reality.** `set_seed` seeds `torch` + `numpy` (+cuda) but
   NOT Python's `random`, NOR MPS, NOR cudnn. Runners draw per-run seeds with
   `random.randint` (Python `random`, never seeded), so a run is reproducible ONLY via
   its MLflow-logged `random_state`. On Apple Silicon (the dev box) runs are not
   bit-reproducible. Do not claim determinism; do not "fix" runner seeds with `set_seed`.
8. **Loss convention: `forward(input, target)` = `forward(prediction, ground_truth)`.**
   `utils.py` (`train_epoch` / `evaluate`) calls `criterion(output, tgt)`. `SGTLoss` is
   the only asymmetric loss:
   `diff = target - input + m`, skew `(1 + lam*sign(diff))**p`; swapping the args flips
   the skew **silently** (untested — every grid config uses `lam=0.0`). Symmetric
   losses use `input - target`.
9. **SGT math contracts.** `qp = q**p` everywhere; validity requires `q**p > 2/p` else
   `scipy.special.beta` returns NaN with no exception (why `p=1.0` is omitted for low
   `q`). The scale constant `v` uses **raw `q`** (`1/q`), NOT `qp` — intentional, and it
   matches `visualize_losses.sgt_loss()`. Keep the three `eps` guards. Any SGT formula
   edit must be **mirrored** between `loss_functions.py` and the NumPy duplicate in
   `visualize_losses.py` (no shared code/test ties them).
10. **Model shape / autoregressive contracts.** All tensors are batch-first
    `(B, seq, features)`; both models use `batch_first=True`; `forward`/`infer` return
    `(B, tgt_len, out_dim)`. `infer` seeds from the last input step, excludes the seed
    token, returns exactly `tgt_len` steps. Keep the causal `tgt_mask` (and its
    `.to(tgt.device)`) in `forward` — dropping it leaks future tokens (low train loss,
    garbage inference, no test failure). Model tests only assert shape + isfinite.
11. **Data leakage / normalization.** Per-sequence `StandardScaler().fit_transform`
    happens inside each loader's per-entity loop BEFORE stacking/splitting.
    `create_dataloaders` splits at the SEQUENCE level (test first, then val) with
    `val_relative = val_split / (1 - test_split)`. Never add a global scaler in
    `data_processing.py`, never split at the window level.
12. **Persistence.** The trained model lives ONLY in a `tempfile.TemporaryDirectory`
    and persists via `mlflow.log_artifact` inside the
    `with mlflow.start_run(), TemporaryDirectory()` block. There is no persistent
    checkpoint dir — results live in `mlruns.db`. Checkpoint loads keep
    `weights_only=True` (`evaluation.py` also passes `map_location=device`;
    `trainer.py`'s reload omits it, relying on the model already being on `device` —
    add `map_location=device` there if you ever reload a checkpoint across devices).

## Dev workflow

- black + isort + flake8, all at **line length 99**, scoped to `skewed_sequences tests`.
  The 99 lives in `pyproject.toml` (`[tool.black]`, `[tool.isort]`),
  `.pre-commit-config.yaml` (the isort + black hook args), and `setup.cfg [flake8]` —
  keep them in sync. (`[tool.ruff.lint.isort]` only configures import sorting, not line
  width.) After config edits, `make format` then `make pre-commit` should produce no diff.
- Tests are plain pytest (`Test<Thing>` classes, fixtures, `parametrize`); external
  boundaries (MlflowClient, models) are mocked — tests never touch the real DB/network.
  New source module ⇒ new matching `test_<module>.py`.
- Datasets are `.npy` of shape `(N, T, 1)`; tensors flow as `(B, seq, features)`.

## Repo-specific gotchas

- `train.py`'s `exp_transform` is a **label only** (logged, never applied); the real
  transform is in `data/synthetic/generate_data.py`. Runners must pass the same value
  to both `generate_data` and `train`.
- Classical (non-SGT) runs still log default `sgt_loss_q` / `sgt_loss_p` — filter on
  `loss_type == "sgt"` before using `sgt_loss_*` columns.
- `calculate_metrics.py` / `calculate_dispersion_scaling.py` **regenerate** shared
  training `.npy` files with `apply_smoothing=False`, `sample_size=1000` — running them
  clobbers real training datasets. Regenerate before the next training run.
- Experiment names MUST end in `_run_<int>`; `collect_results._derive_dataset` strips
  it via `re.sub(r"_run_\d+$", "")` and special-cases the `lanl_` prefix. New naming ⇒
  update the regex and add a `test_collect_results` case.
- `plots.py` reads hardcoded `.npy` basenames; load failures are swallowed by
  try/except + `logger.warning` (green run, missing figure). `plots.py` does not
  `mkdir` `FIGURES_DIR` (unlike the other viz modules).
- `slice_array_to_chunks` is copy-pasted verbatim across owid/rvr/health/lanl loaders —
  fix all copies together.
- `compute_metrics(predictions, targets)` — MAPE denominator is `|target|` (asymmetric);
  sMAPE is the headline metric. Swapping arg order is silent.

## What NOT to commit

`.gitignore` excludes the artifact dirs (`data/`, `models/`, `mlruns/`, `notebooks/`,
`reports/`) plus `mlruns.db` and `*.ipynb`. `.npy` / model-weight files are kept out by
living in those dirs — and are additionally blocked from staging by the
`block_large_secret` hook (regex on `.npy`/`.pt`/`.pth`/…), not by a `.gitignore` glob.
**Never `git add -f`** these (the hook blocks it). Pre-commit enforces
`check-added-large-files` (maxkb=10000) and
`detect-private-key`. Package version lives under **`[tool.poetry]`** in
`pyproject.toml` (poetry-core backend) — do NOT add a PEP 621 `[project]` table
(breaks the Dockerfile's `poetry build -f wheel`).

## Claude Code setup in this repo

- **Skills:** `/code-review` (read-only review of the working tree against the invariants
  above) and `/commit-push` (gated commit → push to `main`). See `.claude/skills/`.
- **Subagents** (`.claude/agents/`): `loss-math-reviewer`, `experiment-reproducibility-auditor`,
  `lazy-cli-guard` — `/code-review` delegates to these for deep, file-specific audits.
- **Hooks** (`.claude/settings.json` → `.claude/hooks/`): auto-format edited `.py`
  (black+isort @ 99); guard against destructive git (force-push, `reset --hard`,
  `--no-verify`, deleting `main`); block staging large/secret/artifact files; run
  pytest on stop when source changed. Disable any hook by editing `.claude/settings.json`.
