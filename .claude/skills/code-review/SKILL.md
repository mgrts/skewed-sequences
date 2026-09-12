---
name: code-review
description: Review pending changes in the skewed-sequences repo for correctness and the silent-bug classes this research codebase actually hits — SGT/robust-loss math & gradients, model tensor-shape & autoregressive contracts, the config/CLI/runner parameter-sync and OUTPUT_LENGTH=1 horizon, the MLflow key + seed reproducibility contract, data leakage/windowing/normalization, lazy-import regressions, and secrets/large-file hygiene. Read-only by default; surfaces findings grouped by severity. Use before every commit, or via /commit-push.
---

# Code review for skewed-sequences

Review the changes currently in the working tree (staged + unstaged + untracked)
against the standards that matter for this codebase specifically. Most bugs here
are **silent**: they pass `pytest` (tests assert shape + isfinite, mock MLflow, and
the whole grid uses `lam=0.0`) yet change the math, the prediction horizon, or the
logged metrics. The job is to catch those.

The review is **read-only by default** — fixes are surfaced as recommendations and
only applied if the user explicitly asks.

## Arguments

`$ARGUMENTS` — optional. Specific files or globs to scope the review (defaults to the
entire diff).

## Flow

### Step 1: Gather changes

```bash
git status --short
git diff --staged --stat
git diff --stat
```

If there is nothing pending, stop: "Nothing to review."

### Step 2: Read the diff

For each changed file, read the actual diff (not just the file list) so the review
reasons about what changed. Note which subsystems are touched — that selects which
checks below apply and which subagent to delegate to.

### Step 3: Delegate deep audits to subagents

When the diff touches a fragile subsystem, dispatch the matching subagent (via the
Agent tool, `subagent_type`) and fold its findings into the report:

- Touches `modeling/loss_functions.py`, `visualization/visualize_losses.py`,
  `data/synthetic/generate_data.py` (the `SkewedGeneralizedT` pdf/logpdf) or
  `metrics.py` (`fit_sgt` / `sgt_increment_fit`, `modeling/utils.residual_scale_estimate`)
  → **`loss-math-reviewer`**.
- Touches `cli.py`, `config.py`, or adds/changes a Typer command/runner
  → **`lazy-cli-guard`**.
- Touches `train.py`, `trainer.py`, `collect_results.py`, `config.py` grids, or
  experiment runners/naming → **`experiment-reproducibility-auditor`**.

Run independent subagents in parallel. For a small diff that clearly matches none of
these, do the checks inline.

### Step 4: CRITICAL — Loss math & gradient correctness

Applies to `modeling/loss_functions.py` and the NumPy duplicate `visualization/visualize_losses.py`.

- **SGT residual & skew:** `SGTLoss.forward` must keep `diff = target - input + m`,
  and the skew term `(1 + lam*torch.sign(diff))**p` must read that same `diff`. Confirm
  the trainer still calls `criterion(output, tgt)` (pred first, truth second) in `utils.py`.
- **Scale constant:** `v` uses **raw** `q` (`q**(-1.0)`), NOT `qp`. Beta args are exactly
  `beta(1/p, qp)`, `beta(2/p, qp - 1/p)`, `beta(3/p, qp - 2/p)`.
- **eps guards:** all three retained — `v_denom + eps`, `qp*skew_term + eps`,
  `log(1 + ratio + eps)`. Removing any can produce silent NaN for non-zero `lam` / tiny
  `sigma` (the grid never exercises it).
- **dtype/device:** every `torch.tensor` constant in `SGTLoss` (`B1,B2,B3,sigma_t,m`)
  carries `dtype=`/`device=` derived from `input`; the loss returns a scalar `.mean()`.
- **TukeyLoss:** the code uses the **masked in-place** assignment (`loss[mask]` /
  `loss[~mask]`); prefer keeping it — it avoids evaluating the bounded cubic on
  out-of-range residuals. A `torch.where` rewrite of *this* bounded cubic actually keeps
  finite gradients (the classic `torch.where` NaN-poison only bites when the unused
  branch has sqrt/log/division), so it is not forbidden — but if you refactor, run
  `backward()` with residuals `> c` and assert the grad is finite.
- **(p,q) validity:** any new `(p,q)` added to a config satisfies `q**p > 2/p` (flag
  `p=1.0` for low `q`).
- **Mirror (three-way):** the SGT math lives in `loss_functions.py` (torch),
  `generate_data.SkewedGeneralizedT.pdf/.logpdf` (NumPy) and `visualize_losses.sgt_loss`
  (NumPy); a formula edit in any one must be mirrored in the other two —
  `test_sgt_consistency.py` pins their agreement and `test_generate_data.py` pins
  `logpdf == log(pdf)`.
- **Factory constants:** `get_loss_function` (`train.py`) scales every robust threshold by
  the MAD-based `residual_scale`: `CauchyLoss(gamma=(2.3849*rs)**2)`,
  `HuberLoss(delta=1.345*rs)`, `TukeyLoss(c=4.685*rs)`, `CharbonnierLoss(eps=1.345*rs)`,
  `SGTLoss(sigma=sgt_loss_sigma*rs)` (see `references/SGT_SCALE_FINDING.md`). The
  plotting baselines in `visualize_losses.py` use the unit-scale versions. Reverting SGT
  to a fixed `sigma=1.0` silently pins it to the MSE regime.
- **Residual scale guard:** `residual_scale_estimate` RAISES on a degenerate (`<= 1e-4`)
  or non-finite scale — do not reintroduce an `eps` floor (it produced 11 degenerate OWID
  runs). `metrics.fit_sgt` keeps `p` and `sigma` fixed exactly as training does and
  bounds `q` below by `sgt_min_q(p)` (`q**p > 2/p`).

### Step 5: CRITICAL — MLflow + seed reproducibility contract

- Key names live ONCE in `mlflow_contract.py` (`PARAM_KEYS`, `ALL_SUMMARY_METRIC_KEYS` =
  `best_{train,val,test}_{smape,mape,rmse,mae}` + `best_test_naive_{rmse,mae}` +
  `best_test_mase`, plus the `residual_scale` param); `train.main` (producer) and
  `collect_results.py` (consumer) both import them and `test_mlflow_contract.py` pins the
  producer. The seed key is **`random_state`** (NOT `seed`). A rename on one side silently
  fills NaN columns.
- **Resume contract** (`run_experiments/_runner.py`): `has_finished_run` matches a FINISHED
  run on `str(value)` of `loss_type / random_state / output_length / context_length /
  stride`, the SGT params, and — when the runner passes them — `model_type / embed_dim /
  num_heads / num_layers / batch_size / num_epochs / early_stopping_patience`. This only
  works because `train.main` logs those arguments unchanged
  (`test_train_logs_resume_match_params_verbatim`); any normalisation before
  `log_params`, or a new runner kwarg that is matched but not logged, silently disables
  skipping (or, worse, never matches). `logged_experiment_param` refuses an experiment
  holding two values of the key (mixed seeds / strides).
- Training body stays inside `with mlflow.start_run(), tempfile.TemporaryDirectory()`;
  `mlflow.log_artifact(model_save_path)` runs before the block exits (the temp file is the
  only model copy). Checkpoint loads keep `weights_only=True`; `evaluation.py` also passes
  `map_location=device` (note `trainer.py`'s reload omits it — fine while the model is
  already on `device`, but flag it if a cross-device reload is introduced).
- `set_seed(seed)` precedes `create_dataloaders` and model init, and the same `seed`
  flows into `train_test_split(random_state=seed)`.
- If the diff **claims determinism/reproducibility**, flag that `set_seed` does NOT cover
  Python `random` / MPS / cudnn, and that runner per-run seeds use unseeded `random.randint`.

### Step 6: CRITICAL — Lazy-import & CLI registration regressions

- `cli.py` top-level imports are ONLY `importlib` / `typing` / `typer`; `config.py` only
  `pathlib` / `dotenv` / `loguru` / optional `tqdm`. **Reject** any
  `torch`/`scipy`/`sklearn`/`mlflow`/`pandas`/`matplotlib` top-level import. No test
  catches this — verify in a fresh subprocess:
  `python -c "import sys, skewed_sequences.cli; print([m for m in ('torch','mlflow','scipy','sklearn','pandas') if m in sys.modules])"` → must print `[]`.
- Every `_register_lazy`/`add_typer` target module exists and defines top-level
  `app = typer.Typer(...)`. Smoke-test new commands with `skseq <group> <cmd> --help`
  (a wrong path/attr only fails at invocation).
- A new `run_experiments` runner intended for the CLI has BOTH a top-level `app` AND a
  `_register_lazy` entry (cf. the unregistered `lambdas.py`).
- No command body was inlined into `cli.py` instead of being deferred via `_register_lazy`.

### Step 7: HIGH — Config / CLI / runner parameter sync & single-step horizon

- Every `TRAINING_CONFIGS` / `LAMBDA_SWEEP_CONFIGS` entry has an explicit `output_length`;
  `_runner.run_training_config` reads `.get("output_length", OUTPUT_LENGTH)` (the old
  magic-5 fallback is gone — do not reintroduce a literal). SGT entries also carry
  `sgt_loss_lambda`/`q`/`sigma`/`p`.
- All six runners (`synthetic / lanl / owid_covid / rvr_us / head_attention /
  lambda_sweep`) import `BATCH_SIZE / NUM_EPOCHS / EARLY_STOPPING_PATIENCE / NUM_WORKERS`
  and `N_RUNS` from `config.py` — no inline literals (the unregistered `lambdas.py` is
  run by hand). The three synthetic runners default `n_sequences` / `stride` to
  `SYNTHETIC_N_SEQUENCES` / `SYNTHETIC_STRIDE` (they must regenerate identical data; the
  λ runner refuses a stride that differs from the experiment's logged one). Runner CLI
  params use plain or `Annotated[..., typer.Option(help=...)]` defaults — never a bare
  `typer.Option(...)` default, which is a truthy `OptionInfo` when `main()` is called
  directly from Python/tests.
- If `SEQUENCE_LENGTH`/`CONTEXT_LENGTH`/`OUTPUT_LENGTH` changed:
  `CONTEXT_LENGTH + OUTPUT_LENGTH <= SEQUENCE_LENGTH` holds (the `train.py` assert),
  `test_config.py` literals (300/200/1/1) updated, resulting window count `> 0`.
- If `SYNTHETIC_DATA_CONFIGS` (count **4**, keys `lam`/`q`/`sigma`/`experiment_name`/
  `kernel_size`), `TRAINING_CONFIGS` (count **36**, loss set
  `{sgt,mse,mae,cauchy,huber,tukey,charbonnier}`, 4 skewed `lam>0` entries) or
  `LAMBDA_SWEEP_CONFIGS` (count **12**, every `(p,q)` with a `lam=0` twin in the main
  grid) changed, ALL consumers (`visualize_data.py`, `plots.py`, `calculate_metrics.py`,
  `calculate_dispersion_scaling.py`, `increment_fit.py`, runners) AND the pinned counts in
  `test_config.py` / `test_lambda_sweep.py` / `test_head_attention.py` / `test_train.py` /
  `test_loss_functions.py` were updated together.
- Experiment names end in `_run_<int>`; `collect_results._derive_dataset` regex/lanl
  special-case updated if naming changed, with a new `test_collect_results` case. The λ
  sub-sweep and any resumed sweep append to EXISTING experiment names on purpose
  (seed-paired); a different dataset size needs a different experiment name or tracking
  DB, never `--no-resume` into the same name.
- A new loss type is added in `get_loss_function` (`train.py`) AND mirrored into
  `config.TRAINING_CONFIGS` + `test_train.py` + `test_loss_functions.py` + the loss set.

### Step 8: HIGH — Model tensor-shape & autoregressive contracts

- `forward()`/`infer()` still return `(B, tgt_len, out_dim)` and keep `(batch, seq,
  feature)` axis order; `nn.Transformer` and `nn.LSTM` remain `batch_first=True`.
- `forward()` still builds and passes the causal `tgt_mask`, and retains its
  `.to(tgt.device)` re-move; dropping the mask leaks future tokens (low train loss,
  garbage inference, no test failure).
- `infer()` seed = `src[:,-1,:out_dim]` (Transformer) / `src[:,-1:]` (LSTM), collects only
  generated steps (excludes the seed), exactly `tgt_len` iterations. `LSTM.forward` never
  feeds `tgt[:,t]` to predict its own index `t`.
- `PositionalEncoding` stays `(1, max_len, d_model)`, sin on even / cos on odd; `embed_dim`
  even; `max_len >= max(CONTEXT_LENGTH, tgt_len)`. `in_dim == out_dim == data.shape[-1]`.
- Any new model added to the `train.py` factory implements BOTH `forward(src, tgt)` and
  `infer(src, tgt_len)`, else `log_val_predictions` crashes after a full run.

### Step 9: HIGH — Data leakage, windowing & normalization

- `create_dataloaders` splits at the SEQUENCE axis (never window level) and keeps
  `val_relative = val_split / (1 - test_split)`.
- Per-sequence `StandardScaler().fit_transform` stays inside each loader's per-entity loop
  BEFORE stacking/splitting; no global `.fit` on stacked data, no scaling moved into
  `data_processing.py`.
- Windowing: `window_total = context_len + output_len`; `n_windows = (T - window_total)//stride + 1`;
  `end_input = start + context_len`, `end_target = end_input + output_len`.
- `train.py` keeps `assert context_length + output_length <= data.shape[1]`; new
  `create_dataloaders` callers add the same guard, and pass `min_split_sequences`
  (`train.main` uses 5) so a tiny dataset cannot split into 1-sequence val/test sets.
- Real loaders: OWID reads the daily OWID/JHU **wide** file (`config.DATA_URL`;
  `validate_daily_wide_csv` rejects weekly cadence), RVR reads the CDC SODA endpoint with
  `$select` + a `count(*)` row check and drops the `US` / `Region N` aggregates. Both call
  `check_not_step_function` (> 50 % exactly-zero increments = weekly-reported step data)
  before saving. Do not point either loader back at the weekly `owid-covid-data.csv` or
  the Socrata `rows.csv` export.
- Every loader still produces `(N, T, 1)` (`np.vstack` then `[..., np.newaxis]`); a 2-D
  save breaks the `(n_seqs, T, _)` unpack and `skewness_of_diff`'s `ndim == 3` assert.
- `compute_metrics(predictions, targets)` — never swap (MAPE denominator is `|target|`,
  asymmetric). Synthetic kernels end with `safe_normalize`, `kernel_size` odd, SGT params
  satisfy `q**p > 2/p` and `-1 < lam < 1`.
- Analysis scripts (`calculate_metrics`, `calculate_dispersion_scaling`) don't clobber
  canonical training `.npy` paths with smoothing off / small sample size.

### Step 10: MEDIUM — Style, hygiene & docs drift

- Line length **99** consistent across `pyproject.toml` (`[tool.black]`, `[tool.isort]`),
  `.pre-commit-config.yaml` (isort + black hook args), and `setup.cfg [flake8]`
  (`[tool.ruff.lint.isort]` configures import sorting only, not line width);
  `make format` then `make pre-commit` yield no diff.
- Package version stays under `[tool.poetry]` in `pyproject.toml`; no PEP 621 `[project]`
  table (the Dockerfile's `poetry build -f wheel` depends on it).
- **No secrets / no hardcoded credentials or PII** in the diff. No staged files under
  `data/`, `models/`, `mlruns/`, `notebooks/`, `reports/`, nor any `*.ipynb`/`*.npy`/`*.db`;
  no `git add -f` bypassing `.gitignore`; no file `> 10 MB`. `references/` is untracked by
  decision (article DOCX, JupyterHub `mlruns.db`, sweep logs, analysis CSVs) — staging is
  by explicit path, never `git add -A` / `-u` / `.` (the `block_large_secret` hook treats
  those as a whole-tree scan and refuses on the gitignored artifacts there).
- New CLI command calls `apply_style()` before building figures and `mkdir(parents=True,
  exist_ok=True)` before saving into `FIGURES_DIR`. Every `plt.figure`/`plt.subplots` has a
  matching `plt.close()`. Paths derive from `config.*_DIR`; colors from `style.COLORS`/`PALETTE_SEQ` — no hardcoded `reports/...` or hex.
- New public function/class has a docstring + type hints; tensor shapes documented in
  `forward`/`infer` docstrings stay in sync with code.
- If tests added/removed, the README test-count claim and the CLAUDE.md package map are
  updated; a new source module gets a matching `test_<module>.py`.
- The package version under `[tool.poetry]` is bumped by `/commit-push` on every commit
  (patch by default); a diff that edits `version` by hand should match that convention.

### Step 11: Run tests + hooks

Run both and report exit status:

```bash
make test         # poetry run pytest  (242 tests as of 2026-09-12; read the live count)
make pre-commit    # black/isort/flake8 @ 99 + check-added-large-files + detect-private-key
```

Either failing is a critical finding. (`make pre-commit` auto-fixes formatting on a
re-run; report what it changed.)

### Step 12: Report findings

Group findings by severity:

- **Critical** — wrong loss math/gradients, broken autoregressive/shape contract,
  MLflow-key or seed-contract break, lazy-import regression, secret/large-file leak,
  broken test or pre-commit hook.
- **High** — config/CLI/runner desync, OUTPUT_LENGTH/window contract break, data leakage,
  a known silent-bug-class match.
- **Medium** — style/hygiene/version-table violation, missing `mkdir`/`plt.close`,
  docs drift.
- **Low** — comment / naming / docstring / type-hint polish.

For each finding: file path, the symbol or line, and a concrete suggestion. Do not make
changes unless the user asks.

If there are zero findings: report "Review passed — N files reviewed, M lines changed,
pytest <result>, pre-commit <result>."
