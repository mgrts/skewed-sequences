"""Aggregate replicate runs into summary statistics + SGT-vs-baseline tests.

``collect_results`` emits one row per MLflow run; this command turns those raw
rows into the numbers a comparison actually needs:

* ``summarize`` — per-config mean/std/median/IQR + a 95% CI half-width over the
  ``N_RUNS`` replicates (the error bars reviewers A4/B11 asked for).
* ``compare_sgt_vs_baselines`` — within each (dataset, model) cell, the best SGT
  config vs each classical baseline, tested with a **paired Wilcoxon signed-rank**
  test (B12).
* ``best_methods`` — one row per (dataset, model): the overall best loss and
  whether the best SGT config beats the best baseline significantly (the B15
  summary table: dataset x best method x significance).

Pairing: the grid runners draw the per-run seed ONCE per ``(run_idx, model_type)``
and reuse it across every loss type, so all losses in a run share the same data
split and the same initialization *seed* (the sklearn ``train_test_split`` and the
torch weight init are reproducible from it). The runs are NOT bit-reproducible —
``set_seed`` seeds only torch + numpy, not Python ``random`` / MPS / cudnn
(CLAUDE.md invariant #7) — so reproducibility is only via the logged
``random_state``. The shared split + init seed still makes the replicates a
legitimate matched block, so the paired Wilcoxon signed-rank test (matched on
``random_state``) is valid and far more sensitive at ``N_RUNS=10`` than an unpaired
test. The SGT config compared is the best-of-grid, so its advantage is optimistic
(winner's curse) — read the p-values as a screen, not a confirmatory result.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import typer

from skewed_sequences.config import REPORTS_DIR

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


def _ci95(s: pd.Series) -> float:
    """Half-width of a normal-approx 95% CI of the mean (``1.96 * std / sqrt(n)``)."""
    n = s.count()
    if n < 2:
        return np.nan
    return 1.96 * s.std(ddof=1) / np.sqrt(n)


def _finished(df: pd.DataFrame) -> pd.DataFrame:
    if "status" in df.columns:
        return df[df["status"] == "FINISHED"].copy()
    return df.copy()


def summarize(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-config replicate summary of ``metric`` over runs (mean/std/median/IQR/CI)."""
    finished = _finished(df).dropna(subset=[metric])
    if finished.empty:
        return pd.DataFrame(columns=CONFIG_KEYS + ["n", "mean", "std", "ci95", "median", "iqr"])
    return (
        finished.groupby(CONFIG_KEYS, dropna=False)[metric]
        .agg(n="count", mean="mean", std="std", ci95=_ci95, median="median", iqr=_iqr)
        .reset_index()
        .sort_values(["dataset", "mean"])
    )


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


def compare_sgt_vs_baselines(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """For each dataset, best SGT config vs each baseline (paired Wilcoxon)."""
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
                    "wilcoxon_p": p,
                    "significant": bool(p < ALPHA) if np.isfinite(p) else False,
                    "n_pairs": n_pairs,
                }
            )
    return pd.DataFrame(rows)


def best_methods(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per (dataset, model): the B15 summary table.

    Reports the overall best loss (lowest mean ``metric``), the best SGT config,
    the best classical baseline, and the paired-Wilcoxon verdict of SGT vs the
    best baseline. ``lower metric is better`` throughout.
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
        median_diff = float(np.median(sgt_vals) - np.median(base_vals)) if n_pairs else np.nan
        if not np.isfinite(p):
            verdict = "n/a"
        elif p >= ALPHA:
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
    better) with a paired-Wilcoxon p-value. Pairs on the shared per-run
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
                diff, p = np.nan, np.nan
            else:
                mh_vals, sg_vals, n_pairs = _paired_vectors(gh, single, metric)
                diff = float(np.median(mh_vals) - np.median(sg_vals)) if n_pairs else np.nan
                p = _wilcoxon_p(mh_vals, sg_vals)
            row = dict(zip(HEAD_CONFIG_KEYS, cfg_vals))
            row.update(
                num_heads=int(nh) if pd.notna(nh) else nh,
                n=int(vals.count()),
                mean=float(vals.mean()),
                ci95=float(_ci95(vals)) if vals.count() >= 2 else np.nan,
                median=float(vals.median()),
                vs_single_head_median_diff=diff,
                vs_single_head_wilcoxon_p=p,
                vs_single_head_better=bool(np.isfinite(diff) and diff < 0),
            )
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(HEAD_CONFIG_KEYS + ["num_heads"])


@app.command()
def main(
    input_path: Path = REPORTS_DIR / "experiment_results.csv",
    output_path: Path = REPORTS_DIR / "experiment_summary.csv",
    metric: str = "best_test_smape",
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_path, index=False)
    comp_path = output_path.with_name(f"{output_path.stem}_comparisons.csv")
    comparisons.to_csv(comp_path, index=False)
    best_path = output_path.with_name(f"{output_path.stem}_best_methods.csv")
    best.to_csv(best_path, index=False)

    typer.echo(f"Wrote {len(grouped)} config summaries to {output_path}")
    typer.echo(f"Wrote {len(comparisons)} SGT-vs-baseline comparisons to {comp_path}")
    typer.echo(f"Wrote {len(best)} best-method rows to {best_path}")
    if not head.empty:
        head_path = output_path.with_name(f"{output_path.stem}_head_sweep.csv")
        head.to_csv(head_path, index=False)
        typer.echo(f"Wrote {len(head)} head-sweep rows to {head_path}")


if __name__ == "__main__":
    app()
