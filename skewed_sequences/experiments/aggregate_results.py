"""Aggregate replicate runs into summary statistics + SGT-vs-baseline tests.

``collect_results`` emits one row per MLflow run; this command turns those raw
rows into the numbers a comparison actually needs:

* ``summarize`` — per-config mean/std/median/IQR + a BCa bootstrap 95 % interval
  (``ci_low`` / ``ci_high``) over the ``N_RUNS`` replicates (the error bars reviewers
  A4/B11 asked for).
* ``compare_sgt_vs_baselines`` — within each (dataset, model) cell, the best SGT
  config vs each classical baseline, tested with an exact **sign-flip permutation
  test** on the seed-paired differences (``perm_p``, the headline) and the paired
  Wilcoxon signed-rank test (``wilcoxon_p``, kept alongside) (B12).
* ``best_methods`` — one row per (dataset, model): the overall best loss and
  whether the best SGT config beats the best baseline significantly (the B15
  summary table: dataset x best method x significance).
* ``anchor_comparisons`` — pre-registered SGT anchors (Cauchy-like, p=1.5,
  MSE-like, MAE-like) vs every baseline, *without* best-of-grid selection, so the
  numbers carry no winner's curse.
* ``lambda_effect`` — every skewed (``lambda != 0``) SGT config vs the symmetric
  config with the same (p, q), seed-paired: does the asymmetric loss help?
* ``head_sweep_summary`` — the multi-head study, per head count vs single-head.

Pairing: the grid runners draw the per-run seed ONCE per ``(run_idx, model_type)``
and reuse it across every loss type, so all losses in a run share the same data
split and the same initialization *seed* (the sklearn ``train_test_split`` and the
torch weight init are reproducible from it). The runs are NOT bit-reproducible —
``set_seed`` seeds only torch + numpy, not Python ``random`` / MPS / cudnn
(CLAUDE.md invariant #7) — so reproducibility is only via the logged
``random_state``. The shared split + init seed still makes the replicates a
legitimate matched block, so the sign-flip permutation test and the paired Wilcoxon
test (both matched on ``random_state``) are valid and far more sensitive at
``N_RUNS=10`` than an unpaired test. Both are distribution-free (``skewed_sequences.stats``). The SGT config compared is the best-of-grid, so its advantage is optimistic
(winner's curse) — read the p-values as a screen, not a confirmatory result.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import typer

from skewed_sequences.config import REPORTS_DIR
from skewed_sequences.stats import bootstrap_ci, signflip_p

app = typer.Typer(pretty_exceptions_show_locals=False)

CONFIG_KEYS = ["dataset", "model_type", "loss_type", "sgt_loss_p", "sgt_loss_q", "sgt_loss_lambda"]
SGT_CONFIG_KEYS = ["sgt_loss_p", "sgt_loss_q", "sgt_loss_lambda"]
# SGT-vs-baseline comparison is run within each (dataset, model) cell.
GROUP_KEYS = ["dataset", "model_type"]
ALPHA = 0.05

# The multi-head study (run-head-sweep) lives under this dataset name. It is
# analysed separately (per head count, not SGT-vs-baseline) and is excluded from
# the main summaries so its rows never get averaged across head counts.
HEAD_DATASET = "head-heavy-tailed"
HEAD_CONFIG_KEYS = ["loss_type", "sgt_loss_p", "sgt_loss_q", "sgt_loss_lambda"]


def _iqr(s: pd.Series) -> float:
    return s.quantile(0.75) - s.quantile(0.25)


def _ci(s: pd.Series) -> tuple[float, float]:
    """BCa bootstrap 95 % interval (low, high) of the mean over the replicates."""
    return bootstrap_ci(s.to_numpy(dtype=float))


def _finished(df: pd.DataFrame) -> pd.DataFrame:
    if "status" in df.columns:
        return df[df["status"] == "FINISHED"].copy()
    return df.copy()


def summarize(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-config replicate summary of ``metric`` over runs (mean/std/median/IQR/CI)."""
    finished = _finished(df).dropna(subset=[metric])
    cols = ["n", "mean", "std", "ci_low", "ci_high", "median", "iqr"]
    if finished.empty:
        return pd.DataFrame(columns=CONFIG_KEYS + cols)
    g = finished.groupby(CONFIG_KEYS, dropna=False)[metric]
    out = g.agg(n="count", mean="mean", std="std", median="median", iqr=_iqr)
    bounds = g.apply(_ci).reindex(out.index)
    out["ci_low"] = [b[0] for b in bounds]
    out["ci_high"] = [b[1] for b in bounds]
    return out[cols].reset_index().sort_values(["dataset", "mean"])


def _best_sgt_config(sgt: pd.DataFrame, metric: str):
    """Return the (p, q, lambda) of the lowest-mean SGT config, or ``None``."""
    means = sgt.groupby(SGT_CONFIG_KEYS, dropna=False)[metric].mean()
    if means.empty:
        return None
    best = means.idxmin()
    return best if isinstance(best, tuple) else (best,)


def _select(df: pd.DataFrame, keys, values) -> pd.DataFrame:
    out = df
    for key, val in zip(keys, values):
        out = out[out[key] == val]
    return out


def _paired_vectors(a: pd.DataFrame, b: pd.DataFrame, metric: str):
    """Align two replicate frames on ``random_state`` -> matched value vectors.

    Falls back to order-based pairing only when ``random_state`` is absent (e.g.
    legacy CSVs); returns ``(a_vals, b_vals, n_pairs)``.
    """
    if "random_state" in a.columns and "random_state" in b.columns:
        a_by = a.groupby("random_state")[metric].mean()
        b_by = b.groupby("random_state")[metric].mean()
        common = a_by.index.intersection(b_by.index)
        return a_by.loc[common].to_numpy(), b_by.loc[common].to_numpy(), len(common)
    n = min(len(a), len(b))
    return a[metric].to_numpy()[:n], b[metric].to_numpy()[:n], n


def _wilcoxon_p(a_vals: np.ndarray, b_vals: np.ndarray) -> float:
    """Two-sided paired Wilcoxon signed-rank p-value (NaN if undefined)."""
    if len(a_vals) < 2 or np.allclose(a_vals, b_vals):
        return np.nan
    try:
        _, p = wilcoxon(a_vals, b_vals, alternative="two-sided", zero_method="wilcox")
    except ValueError:
        return np.nan
    return float(p)


def _perm_p(a_vals: np.ndarray, b_vals: np.ndarray) -> float:
    """Exact two-sided sign-flip permutation p on the paired differences (NaN if undefined).

    Same guard as ``_wilcoxon_p``: NaN for fewer than two pairs or for vectors equal up to
    rounding, so ``significant`` can never flip on float noise.
    """
    if len(a_vals) < 2 or np.allclose(a_vals, b_vals):
        return np.nan
    return signflip_p(np.asarray(a_vals, dtype=float) - np.asarray(b_vals, dtype=float))


def compare_sgt_vs_baselines(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """For each dataset, best SGT config vs each baseline (sign-flip permutation p, Wilcoxon p)."""
    finished = _finished(df).dropna(subset=[metric])
    if finished.empty:
        return pd.DataFrame()
    rows = []
    for group_vals, dsub in finished.groupby(GROUP_KEYS):
        dataset, model_type = group_vals
        sgt = dsub[dsub["loss_type"] == "sgt"]
        baselines = dsub[dsub["loss_type"] != "sgt"]
        if sgt.empty or baselines.empty:
            continue

        best_cfg = _best_sgt_config(sgt, metric)
        if best_cfg is None:
            continue
        sgt_best = _select(sgt, SGT_CONFIG_KEYS, best_cfg)

        for loss_type, lsub in baselines.groupby("loss_type"):
            sgt_vals, base_vals, n_pairs = _paired_vectors(sgt_best, lsub, metric)
            p = _wilcoxon_p(sgt_vals, base_vals)
            pp = _perm_p(sgt_vals, base_vals)
            rows.append(
                {
                    "dataset": dataset,
                    "model_type": model_type,
                    "metric": metric,
                    "sgt_p": best_cfg[0],
                    "sgt_q": best_cfg[1] if len(best_cfg) > 1 else np.nan,
                    "sgt_lambda": best_cfg[2] if len(best_cfg) > 2 else np.nan,
                    "baseline": loss_type,
                    "sgt_mean": float(np.mean(sgt_vals)) if n_pairs else np.nan,
                    "baseline_mean": float(np.mean(base_vals)) if n_pairs else np.nan,
                    "sgt_median": float(np.median(sgt_vals)) if n_pairs else np.nan,
                    "baseline_median": float(np.median(base_vals)) if n_pairs else np.nan,
                    "median_diff": (
                        float(np.median(sgt_vals) - np.median(base_vals)) if n_pairs else np.nan
                    ),
                    "perm_p": pp,
                    "wilcoxon_p": p,
                    "significant": bool(pp < ALPHA) if np.isfinite(pp) else False,
                    "n_pairs": n_pairs,
                }
            )
    return pd.DataFrame(rows)


def best_methods(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per (dataset, model): the B15 summary table.

    Reports the overall best loss (lowest mean ``metric``), the best SGT config,
    the best classical baseline, and the permutation-test verdict of SGT vs the
    best baseline (``perm_p``; the Wilcoxon p is reported alongside). ``lower metric is better`` throughout.
    """
    finished = _finished(df).dropna(subset=[metric])
    if finished.empty:
        return pd.DataFrame()
    rows = []
    for group_vals, dsub in finished.groupby(GROUP_KEYS):
        dataset, model_type = group_vals
        config_means = dsub.groupby(CONFIG_KEYS, dropna=False)[metric].mean()
        if config_means.empty:
            continue
        overall_best_idx = config_means.idxmin()
        overall_best_loss = overall_best_idx[CONFIG_KEYS.index("loss_type")]

        sgt = dsub[dsub["loss_type"] == "sgt"]
        baselines = dsub[dsub["loss_type"] != "sgt"]

        best_cfg = _best_sgt_config(sgt, metric) if not sgt.empty else None
        sgt_best = _select(sgt, SGT_CONFIG_KEYS, best_cfg) if best_cfg is not None else sgt
        sgt_mean = sgt_best[metric].mean() if not sgt_best.empty else np.nan

        best_base_loss, best_base_mean, base_best = np.nan, np.nan, baselines.iloc[:0]
        if not baselines.empty:
            base_means = baselines.groupby("loss_type")[metric].mean()
            best_base_loss = base_means.idxmin()
            best_base_mean = float(base_means.min())
            base_best = baselines[baselines["loss_type"] == best_base_loss]

        sgt_vals, base_vals, n_pairs = (
            _paired_vectors(sgt_best, base_best, metric)
            if (not sgt_best.empty and not base_best.empty)
            else (np.array([]), np.array([]), 0)
        )
        p = _wilcoxon_p(sgt_vals, base_vals)
        pp = _perm_p(sgt_vals, base_vals)
        median_diff = float(np.median(sgt_vals) - np.median(base_vals)) if n_pairs else np.nan
        if not np.isfinite(pp):
            verdict = "n/a"
        elif pp >= ALPHA:
            verdict = "tie"
        elif median_diff < 0:
            verdict = "SGT wins"
        else:
            verdict = "baseline wins"

        rows.append(
            {
                "dataset": dataset,
                "model_type": model_type,
                "metric": metric,
                "overall_best_loss": overall_best_loss,
                "best_sgt_p": best_cfg[0] if best_cfg else np.nan,
                "best_sgt_q": best_cfg[1] if best_cfg and len(best_cfg) > 1 else np.nan,
                "best_sgt_lambda": best_cfg[2] if best_cfg and len(best_cfg) > 2 else np.nan,
                "sgt_mean": float(sgt_mean) if np.isfinite(sgt_mean) else np.nan,
                "best_baseline": best_base_loss,
                "baseline_mean": best_base_mean,
                "median_diff_sgt_minus_baseline": median_diff,
                "perm_p": pp,
                "wilcoxon_p": p,
                "verdict": verdict,
                "n_pairs": n_pairs,
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "model_type"]) if rows else pd.DataFrame()


def head_sweep_summary(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Analyse the multi-head study (A2/A12, Fig.12).

    Per (loss config, head count): replicate mean/std/95%CI/median of ``metric``,
    plus the **multi-head effect** — the seed-paired difference of each head count
    against the single-head (``num_heads==1``) baseline (negative => multi-head
    better) with the sign-flip permutation p and the paired-Wilcoxon p. Pairs on the shared per-run
    ``random_state`` (head configs are seed-paired within a run).
    """
    finished = _finished(df).dropna(subset=[metric])
    if "num_heads" not in finished.columns:
        return pd.DataFrame()
    head = finished[finished["dataset"] == HEAD_DATASET]
    if head.empty:
        return pd.DataFrame()

    rows = []
    for cfg_vals, g in head.groupby(HEAD_CONFIG_KEYS, dropna=False):
        cfg_vals = cfg_vals if isinstance(cfg_vals, tuple) else (cfg_vals,)
        single = g[g["num_heads"] == 1]
        for nh, gh in g.groupby("num_heads"):
            vals = gh[metric]
            if nh == 1 or single.empty:
                diff, p, pp = np.nan, np.nan, np.nan
            else:
                mh_vals, sg_vals, n_pairs = _paired_vectors(gh, single, metric)
                diff = float(np.median(mh_vals) - np.median(sg_vals)) if n_pairs else np.nan
                p = _wilcoxon_p(mh_vals, sg_vals)
                pp = _perm_p(mh_vals, sg_vals)
            row = dict(zip(HEAD_CONFIG_KEYS, cfg_vals))
            ci_low, ci_high = _ci(vals)
            row.update(
                num_heads=int(nh) if pd.notna(nh) else nh,
                n=int(vals.count()),
                mean=float(vals.mean()),
                ci_low=ci_low,
                ci_high=ci_high,
                median=float(vals.median()),
                vs_single_head_median_diff=diff,
                vs_single_head_perm_p=pp,
                vs_single_head_wilcoxon_p=p,
                vs_single_head_better=bool(np.isfinite(diff) and diff < 0),
            )
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(HEAD_CONFIG_KEYS + ["num_heads"])


# Pre-registered SGT anchors (p, q, lambda) compared against every baseline WITHOUT
# best-of-grid selection (so no winner's curse): the Cauchy-like, mid-norm robust,
# MSE-like and MAE-like corners of the grid.
ANCHORS = ((2.0, 2.5, 0.0), (1.5, 2.5, 0.0), (2.0, 20.0, 0.0), (1.0, 20.0, 0.0))
ANCHOR_LABELS = {
    (2.0, 2.5, 0.0): "cauchy-like",
    (1.5, 2.5, 0.0): "p1.5-robust",
    (2.0, 20.0, 0.0): "mse-like",
    (1.0, 20.0, 0.0): "mae-like",
}
LAMBDA_GROUP_KEYS = GROUP_KEYS + ["sgt_loss_p", "sgt_loss_q"]


def anchor_comparisons(df: pd.DataFrame, metric: str, anchors=ANCHORS) -> pd.DataFrame:
    """Fixed SGT anchors vs each classical baseline (permutation p, Wilcoxon p), per (dataset, model).

    Unlike ``compare_sgt_vs_baselines`` no SGT config is *selected* on the data, so
    a significant difference here is confirmatory rather than a screen.
    """
    finished = _finished(df).dropna(subset=[metric])
    if finished.empty:
        return pd.DataFrame()
    rows = []
    for group_vals, dsub in finished.groupby(GROUP_KEYS):
        dataset, model_type = group_vals
        sgt = dsub[dsub["loss_type"] == "sgt"]
        baselines = dsub[dsub["loss_type"] != "sgt"]
        for anchor in anchors:
            sgt_cfg = _select(sgt, SGT_CONFIG_KEYS, anchor)
            if sgt_cfg.empty:
                continue
            for loss_type, lsub in baselines.groupby("loss_type"):
                sgt_vals, base_vals, n_pairs = _paired_vectors(sgt_cfg, lsub, metric)
                p = _wilcoxon_p(sgt_vals, base_vals)
                pp = _perm_p(sgt_vals, base_vals)
                rows.append(
                    {
                        "dataset": dataset,
                        "model_type": model_type,
                        "metric": metric,
                        "anchor": ANCHOR_LABELS.get(tuple(anchor), str(anchor)),
                        "sgt_p": anchor[0],
                        "sgt_q": anchor[1],
                        "sgt_lambda": anchor[2],
                        "baseline": loss_type,
                        "sgt_mean": float(np.mean(sgt_vals)) if n_pairs else np.nan,
                        "baseline_mean": float(np.mean(base_vals)) if n_pairs else np.nan,
                        "mean_diff": float(np.mean(sgt_vals - base_vals)) if n_pairs else np.nan,
                        "median_diff": (
                            float(np.median(sgt_vals) - np.median(base_vals))
                            if n_pairs
                            else np.nan
                        ),
                        "perm_p": pp,
                        "wilcoxon_p": p,
                        "significant": bool(pp < ALPHA) if np.isfinite(pp) else False,
                        "n_pairs": n_pairs,
                    }
                )
    return pd.DataFrame(rows)


def lambda_effect(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Skewed (``lambda != 0``) SGT vs the symmetric config with the same (p, q).

    Seed-paired within each (dataset, model, p, q); ``mean_diff < 0`` means the
    asymmetric loss improved ``metric``. Configs without a ``lambda = 0`` anchor are
    skipped.
    """
    finished = _finished(df).dropna(subset=[metric])
    sgt = finished[finished["loss_type"] == "sgt"] if not finished.empty else finished
    if sgt.empty:
        return pd.DataFrame()
    rows = []
    for group_vals, g in sgt.groupby(LAMBDA_GROUP_KEYS):
        dataset, model_type, p_, q_ = group_vals
        symmetric = g[g["sgt_loss_lambda"] == 0.0]
        if symmetric.empty:
            continue
        for lam, gl in g[g["sgt_loss_lambda"] != 0.0].groupby("sgt_loss_lambda"):
            skew_vals, sym_vals, n_pairs = _paired_vectors(gl, symmetric, metric)
            p = _wilcoxon_p(skew_vals, sym_vals)
            pp = _perm_p(skew_vals, sym_vals)
            mean_diff = float(np.mean(skew_vals - sym_vals)) if n_pairs else np.nan
            rows.append(
                {
                    "dataset": dataset,
                    "model_type": model_type,
                    "metric": metric,
                    "sgt_p": p_,
                    "sgt_q": q_,
                    "sgt_lambda": lam,
                    "n_pairs": n_pairs,
                    "mean_symmetric": float(np.mean(sym_vals)) if n_pairs else np.nan,
                    "mean_skewed": float(np.mean(skew_vals)) if n_pairs else np.nan,
                    "mean_diff": mean_diff,
                    "median_diff": (
                        float(np.median(skew_vals) - np.median(sym_vals)) if n_pairs else np.nan
                    ),
                    "perm_p": pp,
                    "wilcoxon_p": p,
                    "significant": bool(pp < ALPHA) if np.isfinite(pp) else False,
                    "skewed_better": bool(np.isfinite(mean_diff) and mean_diff < 0),
                }
            )
    return pd.DataFrame(rows)


@app.command()
def main(
    input_path: Path = REPORTS_DIR / "experiment_results.csv",
    output_path: Path = REPORTS_DIR / "experiment_summary.csv",
    metric: str = "best_test_mase",
):
    """Aggregate replicate runs and test SGT vs baselines on ``metric``."""
    df = pd.read_csv(input_path)

    # Keep the multi-head study out of the main SGT-vs-baseline summaries (it is
    # analysed per head count instead); otherwise its rows average across heads.
    main_df = df[df.get("dataset") != HEAD_DATASET] if "dataset" in df.columns else df

    grouped = summarize(main_df, metric)
    comparisons = compare_sgt_vs_baselines(main_df, metric)
    best = best_methods(main_df, metric)
    head = head_sweep_summary(df, metric)
    anchors = anchor_comparisons(main_df, metric)
    lambdas = lambda_effect(main_df, metric)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_path, index=False)
    comp_path = output_path.with_name(f"{output_path.stem}_comparisons.csv")
    comparisons.to_csv(comp_path, index=False)
    best_path = output_path.with_name(f"{output_path.stem}_best_methods.csv")
    best.to_csv(best_path, index=False)

    typer.echo(f"Wrote {len(grouped)} config summaries to {output_path}")
    typer.echo(f"Wrote {len(comparisons)} SGT-vs-baseline comparisons to {comp_path}")
    typer.echo(f"Wrote {len(best)} best-method rows to {best_path}")
    anchors_path = output_path.with_name(f"{output_path.stem}_anchors.csv")
    anchors.to_csv(anchors_path, index=False)
    typer.echo(f"Wrote {len(anchors)} pre-registered anchor comparisons to {anchors_path}")
    lambda_path = output_path.with_name(f"{output_path.stem}_lambda_effect.csv")
    lambdas.to_csv(lambda_path, index=False)
    typer.echo(f"Wrote {len(lambdas)} lambda-effect rows to {lambda_path}")
    if not head.empty:
        head_path = output_path.with_name(f"{output_path.stem}_head_sweep.csv")
        head.to_csv(head_path, index=False)
        typer.echo(f"Wrote {len(head)} head-sweep rows to {head_path}")


if __name__ == "__main__":
    app()
