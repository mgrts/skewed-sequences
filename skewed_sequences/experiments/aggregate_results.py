"""Aggregate replicate runs into summary statistics + SGT-vs-baseline tests.

``collect_results`` emits one row per MLflow run; this command turns those raw
rows into the numbers a comparison actually needs: per-config mean/std/median/IQR
over the ``N_RUNS`` replicates, and a per-dataset significance test of the best
SGT config against each classical baseline.

Note on pairing: runs are NOT seed-paired across loss types (each run draws its
own ``random_state``), so an UNPAIRED Mann-Whitney U is the honest test. The SGT
config compared is the best-of-grid, so its advantage is optimistic (winner's
curse) — read the p-values as a screen, not a confirmatory result.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import typer

from skewed_sequences.config import REPORTS_DIR

app = typer.Typer(pretty_exceptions_show_locals=False)

CONFIG_KEYS = ["dataset", "model_type", "loss_type", "sgt_loss_p", "sgt_loss_q", "sgt_loss_lambda"]
SGT_CONFIG_KEYS = ["sgt_loss_p", "sgt_loss_q", "sgt_loss_lambda"]
# SGT-vs-baseline comparison is run within each (dataset, model) cell.
GROUP_KEYS = ["dataset", "model_type"]


def _iqr(s: pd.Series) -> float:
    return s.quantile(0.75) - s.quantile(0.25)


def _finished(df: pd.DataFrame) -> pd.DataFrame:
    if "status" in df.columns:
        return df[df["status"] == "FINISHED"].copy()
    return df.copy()


def summarize(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-config replicate summary of ``metric`` over runs."""
    finished = _finished(df).dropna(subset=[metric])
    if finished.empty:
        return pd.DataFrame(columns=CONFIG_KEYS + ["n", "mean", "std", "median", "iqr"])
    return (
        finished.groupby(CONFIG_KEYS, dropna=False)[metric]
        .agg(n="count", mean="mean", std="std", median="median", iqr=_iqr)
        .reset_index()
        .sort_values(["dataset", "mean"])
    )


def compare_sgt_vs_baselines(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """For each dataset, best SGT config vs each baseline (unpaired Mann-Whitney)."""
    finished = _finished(df).dropna(subset=[metric])
    rows = []
    for group_vals, dsub in finished.groupby(GROUP_KEYS):
        dataset, model_type = group_vals
        sgt = dsub[dsub["loss_type"] == "sgt"]
        baselines = dsub[dsub["loss_type"] != "sgt"]
        if sgt.empty or baselines.empty:
            continue

        means = sgt.groupby(SGT_CONFIG_KEYS, dropna=False)[metric].mean()
        if means.empty:
            continue
        best_cfg = means.idxmin()  # (p, q, lambda); lower metric is better
        best_cfg = best_cfg if isinstance(best_cfg, tuple) else (best_cfg,)
        sgt_best = sgt
        for key, val in zip(SGT_CONFIG_KEYS, best_cfg):
            sgt_best = sgt_best[sgt_best[key] == val]
        sgt_vals = sgt_best[metric].to_numpy()

        for loss_type, lsub in baselines.groupby("loss_type"):
            base_vals = lsub[metric].to_numpy()
            if len(sgt_vals) < 2 or len(base_vals) < 2:
                p = np.nan
            else:
                try:
                    _, p = mannwhitneyu(sgt_vals, base_vals, alternative="two-sided")
                except ValueError:
                    p = np.nan
            rows.append(
                {
                    "dataset": dataset,
                    "model_type": model_type,
                    "metric": metric,
                    "sgt_p": best_cfg[0],
                    "sgt_q": best_cfg[1] if len(best_cfg) > 1 else np.nan,
                    "baseline": loss_type,
                    "sgt_median": float(np.median(sgt_vals)),
                    "baseline_median": float(np.median(base_vals)),
                    "median_diff": float(np.median(sgt_vals) - np.median(base_vals)),
                    "mannwhitney_p": p,
                    "n_sgt": len(sgt_vals),
                    "n_baseline": len(base_vals),
                }
            )
    return pd.DataFrame(rows)


@app.command()
def main(
    input_path: Path = REPORTS_DIR / "experiment_results.csv",
    output_path: Path = REPORTS_DIR / "experiment_summary.csv",
    metric: str = "best_test_rmse",
):
    """Aggregate replicate runs and test SGT vs baselines on ``metric``."""
    df = pd.read_csv(input_path)

    grouped = summarize(df, metric)
    comparisons = compare_sgt_vs_baselines(df, metric)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_path, index=False)
    comp_path = output_path.with_name(f"{output_path.stem}_comparisons.csv")
    comparisons.to_csv(comp_path, index=False)

    typer.echo(f"Wrote {len(grouped)} config summaries to {output_path}")
    typer.echo(f"Wrote {len(comparisons)} SGT-vs-baseline comparisons to {comp_path}")


if __name__ == "__main__":
    app()
