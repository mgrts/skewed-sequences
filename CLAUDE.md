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
  sub-apps **lazily** via `_register_lazy` / `_LazyTyper`.
- `skewed_sequences/config.py` — single source of truth for constants and
  experiment grids. Eagerly imported by `__init__.py`, so it runs on **every** CLI
  call — keep it lightweight.
- `skewed_sequences/modeling/`
  - `models.py` — `TransformerWithPE`, `LSTM`, `PositionalEncoding`. Both models
    expose `forward(src, tgt)` (teacher-forced) and `infer(src, tgt_len)` (autoregressive).
  - `loss_functions.py` — `SGTLoss` (asymmetric) + `CauchyLoss` / `HuberLoss` / `TukeyLoss` / `CharbonnierLoss` (symmetric).
  - `train.py` — CLI entry (`main`) + `get_loss_function` factory + device selection + MLflow run.
  - `trainer.py` — epoch loop, per-epoch MLflow logging, checkpoint/early-stop on **val MAE**.
  - `utils.py` — `set_seed`, `train_epoch`, `evaluate`, `compute_metrics`, `EarlyStopping`.
  - `data_processing.py` — `SlidingWindowDataset`, `create_dataloaders`.
  - `evaluation.py` — `sliding_window_predictions`, `log_val_predictions`.
- `skewed_sequences/data/` — per-dataset loaders (`synthetic/generate_data.py`,
  `owid_covid/`, `lanl/`, `rvr_us/`, `health_fitness/`). Each is a Typer app writing a `.npy`.
- `skewed_sequences/experiments/`
  - `run_experiments/{synthetic,lanl,owid_covid,rvr_us}_data.py` — grid-sweep runners (Typer).
    `head_attention_data.py` (`run-head-sweep`) — multi-head study: fixed embed width (256),
    head count swept {1,2,4,8} on heavy-tailed synthetic, transformer only.
    `lambdas.py` is an **unregistered** run-by-hand script (no `_register_lazy` entry).
    `lambda_sweep_data.py` (`run-lambda-sweep`) — fine skew grid `LAMBDA_SWEEP_CONFIGS`
    (λ ∈ {0.1,0.2,0.3}) appended to the existing `<dataset>_run_<i>` synthetic experiments.
  - `run_experiments/_runner.py` — shared `run_training_config` helper (one `train.main` call site)
    plus the **resume** helpers `draw_experiment_seed` (reuse the seed already logged under an
    experiment) and `has_finished_run` (skip a config that already has a FINISHED run).
  - `collect_results.py` — reads MLflow back into `reports/experiment_results.csv`.
  - `aggregate_results.py` — replicate summary stats (mean/std/95%CI/median/IQR) + SGT-vs-baseline
    **paired Wilcoxon signed-rank** + a `best_methods` table into `reports/experiment_summary{,_comparisons,_best_methods}.csv`.
  - `calculate_metrics.py`, `calculate_dispersion_scaling.py` — analysis scripts.
  - `increment_fit.py` (`increment-fit`) — MLE of SGT (λ, q) on each dataset's one-step increments
    (`metrics.sgt_increment_fit`), the reviewer-A3/A11 parameter-guidance tool →
    `reports/increment_sgt_fit.csv`.
- `skewed_sequences/visualization/` — `style.py`, `predictions.py`, `plots.py`,
  `visualize_data.py`, `visualize_losses.py` (NumPy reimplementation of the SGT loss).
- `skewed_sequences/metrics.py` — skewness / kappa / dispersion / Hill tail-index metrics +
  `fit_sgt` / `sgt_increment_fit` (SGT MLE with `p` and the residual scale fixed as in training).
- `skewed_sequences/mlflow_contract.py` — single source of the MLflow param/metric key names.
- `skewed_sequences/data/_common.py` — shared loader helpers (`slice_array_to_chunks`, `scale_and_stack`,
  `check_not_step_function`).
- `tests/` — one `test_<module>.py` per source module; plain pytest, no `conftest.py` (**242 tests**).

## How to run

```bash
poetry install
skseq --help

# Single-command modules expose a `main` command:
skseq train main --loss-type mse
skseq data generate-synthetic main
skseq experiments run-synthetic main
skseq experiments run-head-sweep main       # multi-head attention study (heavy-tailed synthetic)
skseq experiments run-lambda-sweep main     # fine lambda grid, appended to the heavy-tailed-skewed experiments
skseq experiments increment-fit main        # SGT (lambda, q) MLE on increments -> reports/increment_sgt_fit.csv
skseq experiments collect-results main      # MLflow -> reports/experiment_results.csv
skseq experiments aggregate-results main    # results.csv -> summary + SGT-vs-baseline tests (MASE default)
skseq data download-owid download && skseq data process-owid main   # daily OWID/JHU wide file
skseq data download-rvr download && skseq data process-rvr main     # CDC SODA endpoint, row-count verified
STAGES=owid,rvr,lambda,collect nohup poetry run bash scripts/run_sweep.sh > sweep.log 2>&1 &   # resumable sweep
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
   carry an explicit `output_length` key. The 4 runners delegate to
   `run_experiments/_runner.py:run_training_config`, which reads
   `train_config.get("output_length", OUTPUT_LENGTH)` — the old magic-`5` fallback is
   gone (the fallback is now the correct horizon, in one place). `test_config.py` asserts
   `output_length == OUTPUT_LENGTH` for every entry.
4. **Window must fit the data.** `CONTEXT_LENGTH(200) + OUTPUT_LENGTH(1) <= SEQUENCE_LENGTH(300)`.
   `train.py` asserts `context_length + output_length <= data.shape[1]` — the ONLY
   guard against a silently-empty `SlidingWindowDataset`
   (`n_windows = (T - window_total)//stride + 1`). Replicate it in any new caller of
   `create_dataloaders`. `create_dataloaders(min_split_sequences=…)` additionally raises when
   any split has fewer sequences than that (`train.main` passes 5; the truncated RVR download
   once trained on a 6/1/1 split without complaint).
5. **`config.py` is the single source of truth.** `SEED=927, SEQUENCE_LENGTH=300,
   CONTEXT_LENGTH=200, OUTPUT_LENGTH=1, STRIDE=1, N_RUNS=10, TRACKING_URI`. These are
   imported as Typer parameter defaults. Runner training-loop defaults now also live in
   `config.py` (`BATCH_SIZE=32 / NUM_EPOCHS=100 / LEARNING_RATE=1e-4 /
   EARLY_STOPPING_PATIENCE=20 / NUM_WORKERS=0`) and are imported by `train.main` and
   every runner — no inline copies. Every runner sweeps `MODEL_TYPES = (transformer,)`
   (the IJIMAI revision is transformer-only; the `LSTM` class is retained but not swept).
   Edit `config.py`, never inline copies. Tests pin counts (**4** synthetic configs,
   **36** training configs — 26 symmetric SGT + **4 skewed (nonzero-λ) SGT** + 6 classical
   — mse/mae/cauchy/huber/tukey/**charbonnier**; **12** `LAMBDA_SWEEP_CONFIGS`) and the constant
   literals. Runners draw the per-run seed **once per (run_idx, model_type)** via
   `_runner.draw_experiment_seed` and reuse it across the loss loop, so replicates are
   **seed-paired across loss types** (matched on `random_state`) — this is what makes
   `aggregate_results`' paired Wilcoxon valid. With `--resume` (the runner default) the seed
   already logged under the experiment name is reused and configs with a FINISHED run are
   skipped, so a killed sweep or an appended sub-sweep (`run-lambda-sweep`) stays paired;
   the match includes the training budget (`num_epochs` / `batch_size` / patience) so a
   smoke run is never mistaken for a sweep run, and the seed lookup refuses an experiment
   that already holds two seeds. `SYNTHETIC_N_SEQUENCES=1000` / `SYNTHETIC_STRIDE=5` are
   the shared defaults of the three synthetic runners (the λ runner refuses a stride that
   differs from the experiment's logged one). A *different* dataset size needs a **different
   experiment name or tracking DB** — `--no-resume` would append unpaired runs into the
   same experiment and `collect_results` would pool them.
6. **MLflow key contract.** Param/metric key names live ONCE in
   `skewed_sequences/mlflow_contract.py` and are imported by both `train.main` (producer)
   and `collect_results.py` (consumer); `test_mlflow_contract.py` pins that what
   `train.main` logs matches. The seed is logged as `random_state` (NOT `seed`); summary
   metrics are `best_{train,val,test}_{smape,mape,rmse,mae}` plus the persistence baseline
   `best_test_naive_{rmse,mae}` and `best_test_mase`, with a `residual_scale` param.
   `collect_results` reads `ALL_SUMMARY_METRIC_KEYS`. On standardized/zero-mean data prefer
   `rmse`/`mae`/`mase` — the percentage metrics are dominated by near-zero targets.
7. **Seed/reproducibility reality.** `set_seed` seeds `torch` + `numpy` (+cuda) but
   NOT Python's `random`, NOR MPS, NOR cudnn. Runners draw per-run seeds with
   `random.randint` (Python `random`, never seeded) unless `--resume` finds one already
   logged, so a run is reproducible ONLY via its MLflow-logged `random_state`. On Apple Silicon (the dev box) runs are not
   bit-reproducible. Do not claim determinism; do not "fix" runner seeds with `set_seed`.
8. **Loss convention: `forward(input, target)` = `forward(prediction, ground_truth)`.**
   `utils.py` (`train_epoch` / `evaluate`) calls `criterion(output, tgt)`. `SGTLoss` is
   the only asymmetric loss:
   `diff = target - input + m`, skew `(1 + lam*sign(diff))**p`; swapping the args flips
   the skew **silently**. The grid now includes skewed (`lam>0`) SGT configs, and
   `test_sgt_consistency.py` covers `lam != 0`. Symmetric losses use `input - target`.
9. **SGT math contracts.** `qp = q**p` everywhere; validity requires `q**p > 2/p` else
   `scipy.special.beta` returns a negative value or +inf and the loss becomes NaN/inf —
   `SGTLoss.__init__` now asserts `q**p > 2/p` to fail fast (so `p=1.0` is omitted for low
   `q`). The scale constant `v` uses **raw `q`** (`1/q`), NOT `qp` — intentional. Keep the
   three `eps` guards. The SGT math is reimplemented in three places (`loss_functions.py`,
   `generate_data.py`'s `SkewedGeneralizedT.pdf`, `visualize_losses.sgt_loss`); any formula
   edit must be **mirrored** — `test_sgt_consistency.py` pins their numerical agreement.
10. **Model shape / autoregressive contracts.** All tensors are batch-first
    `(B, seq, features)`; both models use `batch_first=True`; `forward`/`infer` return
    `(B, tgt_len, out_dim)`. `forward` uses **one-step-shifted teacher forcing**: it seeds
    the decoder from `src[:, -1]` and feeds the *previous* targets, so it never sees the
    value being scored (the Transformer shares `_run_decoder` with `infer`). Do NOT revert
    to feeding raw `tgt` as the decoder input — that leaks the answer at OUTPUT_LENGTH=1.
    `infer` seeds from the last input step, excludes the seed token, returns exactly
    `tgt_len` steps; keep the causal `tgt_mask`. `test_models.py` pins that perturbing the
    scored target does not change `forward`'s output.
11. **Data leakage / normalization.** Per-sequence `StandardScaler().fit_transform`
    happens inside each real loader's per-entity loop BEFORE stacking/splitting (now via
    `data/_common.scale_and_stack`). Both real loaders then call
    `check_not_step_function` (refuses > 50 % exactly-zero increments) and
    `residual_scale_estimate` **raises** below `min_scale=1e-4` — the weekly-reported OWID
    file produced a step function with `residual_scale = 1e-6` and 11 degenerate runs.
    OWID reads the daily OWID/JHU **wide** file (`config.DATA_URL`); RVR reads the CDC SODA
    endpoint with `$select` + a `count(*)` row check and drops the `US` / `Region N`
    aggregate jurisdictions (sums of the state series). The synthetic generator also standardizes per
    sequence (`standardize=True`, the default; diagnostic/plot callers pass `False`).
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
- Raw real-data files are `data/raw/owid_jhu_new_cases.csv` and
  `data/external/rvr_us_hospitalization_daily.csv` (new names on purpose: the old
  `dataset.csv` was the weekly-rebased OWID file and the old `rvr_us_data.csv` a 3.8 MB
  truncated export — a stale copy must never be picked up). Processed names are unchanged
  (`dataset.npy`, `rvr_us_data.npy`). `Turkmenistan` is absent from the JHU file (39 of the
  40 listed countries → 117 sequences); `--all-locations` applies a zero-fraction rule instead.
- Runner CLI params use **plain defaults**, not `typer.Option(...)` — tests call `main(...)`
  directly and an `OptionInfo` default is truthy (`resume`, `all_locations`,
  `include_aggregates` would silently flip).
- `scripts/run_sweep.sh` is stage-selectable (`STAGES=`) and always passes `--resume`;
  it must be launched with `nohup` and **`mlruns.db` must never be deleted between
  launches** (the JupyterHub notebook's first cell did exactly that).
- Classical (non-SGT) runs still log default `sgt_loss_q` / `sgt_loss_p` — filter on
  `loss_type == "sgt"` before using `sgt_loss_*` columns.
- `calculate_metrics.py` / `calculate_dispersion_scaling.py` regenerate synthetic data
  with `apply_smoothing=False`, `sample_size=1000`, but now write to
  `diagnostic_synthetic_dataset.npy` so they no longer clobber the real
  `synthetic_dataset.npy`. (They still regenerate `rvr_us_data.npy` in place.)
- Experiment names MUST end in `_run_<int>`; `collect_results._derive_dataset` strips
  it via `re.sub(r"_run_\d+$", "")` and special-cases the `lanl_` prefix. New naming ⇒
  update the regex and add a `test_collect_results` case.
- `plots.py` reads hardcoded `.npy` basenames; load failures are swallowed by
  try/except + `logger.warning` (green run, missing figure). It now `mkdir`s the figure
  dir before `savefig` (like the other viz modules).
- `slice_array_to_chunks` + the per-chunk scale/stack now live ONCE in `data/_common.py`
  (`scale_and_stack`); the four loaders import them. Chunks are **non-overlapping** (the
  short trailing remainder is dropped) to avoid leaking timesteps across the split.
- `compute_metrics(predictions, targets)` returns `smape/mape/rmse/mae`. MAPE denominator
  is `|target|` (asymmetric); on standardized/zero-mean targets the percentage metrics are
  dominated by near-zero targets, so prefer `rmse`/`mae`. Swapping arg order is silent.

## What NOT to commit

`.gitignore` excludes the artifact dirs (`data/`, `models/`, `mlruns/`, `notebooks/`,
`reports/`) plus `mlruns.db` and `*.ipynb`. `.npy` / model-weight files are kept out by
living in those dirs — and are additionally blocked from staging by the
`block_large_secret` hook (regex on `.npy`/`.pt`/`.pth`/…), not by a `.gitignore` glob.
**Never `git add -f`** these (the hook blocks it). Pre-commit enforces
`check-added-large-files` (maxkb=10000) and
`detect-private-key`. `references/` is **untracked by decision** (2026-09-12): it holds
the reviewed article DOCX, the JupyterHub `mlruns.db` / `sweep.log` copies and the
analysis CSVs. Stage by explicit path (`git add pyproject.toml CLAUDE.md README.md
scripts skewed_sequences tests .claude …`) — never `git add -A` / `-u` / `.`: the
`block_large_secret` hook treats those as a whole-tree scan and refuses on the gitignored
artifacts under `references/`. Package version lives under **`[tool.poetry]`** in
`pyproject.toml` (poetry-core backend) — do NOT add a PEP 621 `[project]` table
(breaks the Dockerfile's `poetry build -f wheel`). **Every `/commit-push` bumps it**
(`poetry version patch` by default; `--minor` / `--major` on request) and records
`Version: old -> new` in the commit body; tags only with `--release`.

## Claude Code setup in this repo

- **Skills:** `/code-review` (read-only review of the working tree against the invariants
  above; delegates to the three subagents below) and `/commit-push` (gated: review →
  tests → pre-commit → doc drift → **mandatory patch version bump** → confirm → explicit-
  path staging → commit → rebase → push to `main`; `--minor` / `--major` / `--release` /
  `--no-push` flags). See `.claude/skills/`.
- **Subagents** (`.claude/agents/`): `loss-math-reviewer`, `experiment-reproducibility-auditor`,
  `lazy-cli-guard` — `/code-review` delegates to these for deep, file-specific audits.
- **Hooks** (`.claude/settings.json` → `.claude/hooks/`): auto-format edited `.py`
  (black+isort @ 99; fires on the Edit/Write tools only — after shell edits run
  `make format`); guard against destructive git (force-push, `reset --hard`,
  `--no-verify`, deleting `main`, any Claude/AI commit attribution); block staging
  large/secret/artifact files (`git add -A`/`-u`/`.` are scanned as the whole tree);
  run pytest on stop when source changed. Disable any hook by editing
  `.claude/settings.json`.
- **Keep the setup in sync:** the skills and agents pin counts and file lists (test
  count, grid sizes, runner names, MLflow keys, loss scaling). When those change in code,
  update `.claude/skills/*/SKILL.md` and `.claude/agents/*.md` in the same commit.
