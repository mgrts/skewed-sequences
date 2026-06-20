"""Tests for skewed_sequences.experiments.aggregate_results."""

import numpy as np
import pandas as pd

from skewed_sequences.experiments.aggregate_results import (
    best_methods,
    compare_sgt_vs_baselines,
    head_sweep_summary,
    summarize,
)

N_RUNS = 10


def _row(loss, p, q, lam, rmse, run):
    return {
        "dataset": "normal",
        "model_type": "transformer",
        "loss_type": loss,
        "sgt_loss_p": p,
        "sgt_loss_q": q,
        "sgt_loss_lambda": lam,
        "random_state": run,  # seed-paired across loss types within a run
        "status": "FINISHED",
        "best_test_rmse": rmse,
    }


def _df():
    """Seed-paired replicates: per run, one row per loss config sharing random_state.

    SGT(q=10) is the overall best; SGT(q=2.5) is poor; mse/mae are baselines with
    mae the better baseline. Differences are consistent across runs so the paired
    Wilcoxon is significant at N_RUNS=10.
    """
    rows = []
    for run in range(N_RUNS):
        rows.append(_row("sgt", 2.0, 10.0, 0.0, 0.10 + 0.01 * run, run))  # good SGT
        rows.append(_row("sgt", 2.0, 2.5, 0.0, 0.50 + 0.01 * run, run))  # bad SGT
        rows.append(_row("mse", 2.0, 2.0, 0.0, 0.30 + 0.01 * run, run))  # baseline
        rows.append(_row("mae", 2.0, 2.0, 0.0, 0.20 + 0.01 * run, run))  # baseline
    return pd.DataFrame(rows)


def test_summarize_groups_by_config():
    g = summarize(_df(), "best_test_rmse")
    assert len(g) == 4  # 2 SGT configs + 2 baselines
    assert {"dataset", "loss_type", "n", "mean", "std", "ci95", "median", "iqr"} <= set(g.columns)
    assert (g["n"] == N_RUNS).all()
    assert (g["ci95"] > 0).all()


def test_summarize_filters_unfinished():
    df = _df()
    df.loc[0, "status"] = "FAILED"
    g = summarize(df, "best_test_rmse")
    assert g["n"].min() == N_RUNS - 1


def test_compare_picks_best_sgt_and_pairs_vs_baselines():
    c = compare_sgt_vs_baselines(_df(), "best_test_rmse")
    # One row per baseline (mse, mae).
    assert set(c["baseline"]) == {"mse", "mae"}
    assert (c["sgt_q"] == 10.0).all()  # best-of-grid SGT selected
    assert (c["median_diff"] < 0).all()  # SGT better than both baselines here
    assert (c["n_pairs"] == N_RUNS).all()
    # Consistent paired differences -> significant two-sided Wilcoxon at n=10.
    assert (c["wilcoxon_p"] < 0.05).all()
    assert c["significant"].all()


def test_best_methods_summary_table():
    b = best_methods(_df(), "best_test_rmse")
    assert len(b) == 1
    row = b.iloc[0]
    assert row["dataset"] == "normal"
    assert row["overall_best_loss"] == "sgt"
    assert row["best_sgt_q"] == 10.0
    assert row["best_baseline"] == "mae"  # the stronger baseline
    assert row["median_diff_sgt_minus_baseline"] < 0
    assert row["verdict"] == "SGT wins"
    assert row["n_pairs"] == N_RUNS


def test_unpaired_fallback_when_no_random_state():
    df = _df().drop(columns=["random_state"])
    c = compare_sgt_vs_baselines(df, "best_test_rmse")
    # Falls back to order-based pairing; still produces finite comparisons.
    assert set(c["baseline"]) == {"mse", "mae"}
    assert (c["n_pairs"] == N_RUNS).all()


def test_empty_input():
    cols = [
        "dataset",
        "loss_type",
        "sgt_loss_p",
        "sgt_loss_q",
        "sgt_loss_lambda",
        "random_state",
        "status",
        "best_test_rmse",
    ]
    g = summarize(pd.DataFrame(columns=cols), "best_test_rmse")
    assert len(g) == 0
    assert best_methods(pd.DataFrame(columns=cols), "best_test_rmse").empty
    assert compare_sgt_vs_baselines(pd.DataFrame(columns=cols), "best_test_rmse").empty


def _head_df():
    """Multi-head study rows: one loss, head counts {1,4}, seed-paired, 4-head better."""
    rows = []
    for run in range(N_RUNS):
        for nh, base in ((1, 0.60), (4, 0.50)):
            rows.append(
                {
                    "dataset": "head-heavy-tailed",
                    "model_type": "transformer",
                    "loss_type": "mse",
                    "sgt_loss_p": 2.0,
                    "sgt_loss_q": 2.0,
                    "sgt_loss_lambda": 0.0,
                    "num_heads": nh,
                    "random_state": run,
                    "status": "FINISHED",
                    "best_test_smape": base + 0.01 * run,
                }
            )
    return pd.DataFrame(rows)


def test_head_sweep_summary_multihead_effect():
    h = head_sweep_summary(_head_df(), "best_test_smape")
    assert set(h["num_heads"]) == {1, 4}
    row4 = h[h["num_heads"] == 4].iloc[0]
    assert row4["vs_single_head_median_diff"] < 0  # 4-head better than 1-head
    assert bool(row4["vs_single_head_better"]) is True
    assert row4["vs_single_head_wilcoxon_p"] < 0.05
    row1 = h[h["num_heads"] == 1].iloc[0]
    assert pd.isna(row1["vs_single_head_median_diff"])  # single-head is the baseline


def test_head_sweep_summary_empty_without_head_dataset():
    assert head_sweep_summary(_df(), "best_test_rmse").empty


def test_nonsignificant_when_baseline_ties():
    """When SGT and a baseline are statistically indistinguishable, p is large."""
    rows = []
    rng = np.random.default_rng(0)
    for run in range(N_RUNS):
        base = 0.20 + 0.01 * run
        rows.append(_row("sgt", 2.0, 10.0, 0.0, base + rng.normal(0, 0.001), run))
        rows.append(_row("mae", 2.0, 2.0, 0.0, base + rng.normal(0, 0.001), run))
    c = compare_sgt_vs_baselines(pd.DataFrame(rows), "best_test_rmse")
    assert (~c["significant"]).all()
