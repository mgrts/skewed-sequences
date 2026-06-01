"""Tests for skewed_sequences.experiments.aggregate_results."""

import pandas as pd

from skewed_sequences.experiments.aggregate_results import (
    compare_sgt_vs_baselines,
    summarize,
)


def _df():
    rows = []
    for run in range(5):
        rows.append(  # good SGT config
            {
                "dataset": "normal",
                "model_type": "transformer",
                "loss_type": "sgt",
                "sgt_loss_p": 2.0,
                "sgt_loss_q": 10.0,
                "sgt_loss_lambda": 0.0,
                "status": "FINISHED",
                "best_test_rmse": 0.10 + 0.01 * run,
            }
        )
        rows.append(  # bad SGT config
            {
                "dataset": "normal",
                "model_type": "transformer",
                "loss_type": "sgt",
                "sgt_loss_p": 2.0,
                "sgt_loss_q": 2.5,
                "sgt_loss_lambda": 0.0,
                "status": "FINISHED",
                "best_test_rmse": 0.50 + 0.01 * run,
            }
        )
        rows.append(  # baseline
            {
                "dataset": "normal",
                "model_type": "transformer",
                "loss_type": "mse",
                "sgt_loss_p": 2.0,
                "sgt_loss_q": 2.0,
                "sgt_loss_lambda": 0.0,
                "status": "FINISHED",
                "best_test_rmse": 0.30 + 0.01 * run,
            }
        )
    return pd.DataFrame(rows)


def test_summarize_groups_by_config():
    g = summarize(_df(), "best_test_rmse")
    assert len(g) == 3  # 2 SGT configs + 1 baseline
    assert {"dataset", "loss_type", "n", "mean", "std", "median", "iqr"} <= set(g.columns)
    assert (g["n"] == 5).all()


def test_summarize_filters_unfinished():
    df = _df()
    df.loc[0, "status"] = "FAILED"
    g = summarize(df, "best_test_rmse")
    assert g["n"].min() == 4


def test_compare_picks_best_sgt_and_tests_vs_baseline():
    c = compare_sgt_vs_baselines(_df(), "best_test_rmse")
    assert len(c) == 1
    row = c.iloc[0]
    assert row["baseline"] == "mse"
    assert row["sgt_q"] == 10.0  # the lower-rmse SGT config is selected
    assert row["median_diff"] < 0  # SGT better than the baseline here
    assert 0.0 <= row["mannwhitney_p"] <= 1.0
    assert row["n_sgt"] == 5 and row["n_baseline"] == 5


def test_empty_input():
    cols = [
        "dataset",
        "loss_type",
        "sgt_loss_p",
        "sgt_loss_q",
        "sgt_loss_lambda",
        "status",
        "best_test_rmse",
    ]
    g = summarize(pd.DataFrame(columns=cols), "best_test_rmse")
    assert len(g) == 0
