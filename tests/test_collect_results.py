"""Tests for skewed_sequences.experiments.collect_results."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from skewed_sequences.experiments.collect_results import (
    _derive_dataset,
    collect_experiment_results,
)

# ---------------------------------------------------------------------------
# _derive_dataset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "experiment_name,expected",
    [
        ("normal_run_1", "normal"),
        ("normal_run_10", "normal"),
        ("heavy-tailed_run_2", "heavy-tailed"),
        ("normal-skewed_run_5", "normal-skewed"),
        ("heavy-tailed-skewed_run_3", "heavy-tailed-skewed"),
        ("exp-normal_run_1", "exp-normal"),
        ("exp-heavy-tailed-skewed_run_3", "exp-heavy-tailed-skewed"),
        ("covid-owid_run_1", "covid-owid"),
        ("lanl_sgt_run_1", "lanl"),
        ("lanl_mse_run_5", "lanl"),
        ("lanl_mae_run_10", "lanl"),
        ("lanl_cauchy_run_1", "lanl"),
        ("lanl_huber_run_3", "lanl"),
        ("lanl_tukey_run_7", "lanl"),
        ("rvr-us-bed-occupancy_run_1", "rvr-us-bed-occupancy"),
        ("rvr-us-influenza-cases_run_7", "rvr-us-influenza-cases"),
        # Default experiment name (no _run_N suffix)
        ("Transformer-SGT-synthetic", "Transformer-SGT-synthetic"),
    ],
)
def test_derive_dataset(experiment_name, expected):
    assert _derive_dataset(experiment_name) == expected


# ---------------------------------------------------------------------------
# collect_experiment_results
# ---------------------------------------------------------------------------


def _make_experiment(experiment_id, name):
    exp = MagicMock()
    exp.experiment_id = experiment_id
    exp.name = name
    return exp


def _make_run(
    params, metrics, start_time=1700000000000, status="FINISHED", run_name="fancy-fox-123"
):
    run = MagicMock()
    run.info.run_name = run_name
    run.info.start_time = start_time
    run.info.status = status
    run.data.params = params
    run.data.metrics = metrics
    return run


def _make_page(runs, token=None):
    page = MagicMock()
    page.__iter__ = MagicMock(return_value=iter(runs))
    page.token = token
    return page


@patch("skewed_sequences.experiments.collect_results.MlflowClient")
def test_collect_single_run(mock_client_cls):
    client = mock_client_cls.return_value
    client.search_experiments.return_value = [
        _make_experiment("1", "normal_run_1"),
    ]

    run = _make_run(
        params={
            "random_state": "42",
            "model_type": "transformer",
            "context_length": "200",
            "output_length": "1",
            "stride": "1",
            "loss_type": "mse",
            "sgt_loss_lambda": "0.0",
            "sgt_loss_q": "2.0",
            "sgt_loss_sigma": "1.0",
            "sgt_loss_p": "2.0",
            "exp_transform": "False",
        },
        metrics={
            "best_train_smape": 10.5,
            "best_val_smape": 15.3,
            "best_test_smape": 14.1,
        },
    )
    client.search_runs.return_value = _make_page([run])

    df = collect_experiment_results(tracking_uri="sqlite:///test.db")

    assert len(df) == 1
    row = df.iloc[0]
    assert row["run_name"] == "fancy-fox-123"
    assert row["experiment_name"] == "normal_run_1"
    assert row["dataset"] == "normal"
    assert row["exp_transform"] == "False"
    assert row["status"] == "FINISHED"
    assert row["random_state"] == 42
    assert row["model_type"] == "transformer"
    assert row["context_length"] == 200
    assert row["output_length"] == 1
    assert row["stride"] == 1
    assert row["loss_type"] == "mse"
    assert row["sgt_loss_lambda"] == 0.0
    assert row["sgt_loss_q"] == 2.0
    assert row["sgt_loss_sigma"] == 1.0
    assert row["sgt_loss_p"] == 2.0
    assert row["best_train_smape"] == 10.5
    assert row["best_val_smape"] == 15.3
    assert row["best_test_smape"] == 14.1


@patch("skewed_sequences.experiments.collect_results.MlflowClient")
def test_collect_no_experiments(mock_client_cls):
    client = mock_client_cls.return_value
    client.search_experiments.return_value = []

    df = collect_experiment_results(tracking_uri="sqlite:///test.db")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


@patch("skewed_sequences.experiments.collect_results.MlflowClient")
def test_collect_missing_params_are_nan(mock_client_cls):
    client = mock_client_cls.return_value
    client.search_experiments.return_value = [
        _make_experiment("1", "normal_run_1"),
    ]

    run = _make_run(
        params={"random_state": "42", "loss_type": "mae"},
        metrics={},
    )
    client.search_runs.return_value = _make_page([run])

    df = collect_experiment_results(tracking_uri="sqlite:///test.db")

    assert len(df) == 1
    row = df.iloc[0]
    assert row["random_state"] == 42
    assert row["loss_type"] == "mae"
    assert row["model_type"] is None
    assert pd.isna(row["context_length"])
    assert pd.isna(row["output_length"])
    assert pd.isna(row["sgt_loss_lambda"])
    assert pd.isna(row["sgt_loss_p"])
    assert pd.isna(row["best_train_smape"])
    assert pd.isna(row["best_val_smape"])
    assert pd.isna(row["best_test_smape"])


@patch("skewed_sequences.experiments.collect_results.MlflowClient")
def test_collect_multiple_experiments(mock_client_cls):
    client = mock_client_cls.return_value
    client.search_experiments.return_value = [
        _make_experiment("1", "normal_run_1"),
        _make_experiment("2", "lanl_sgt_run_1"),
    ]

    run1 = _make_run(
        params={"random_state": "1", "loss_type": "mse"},
        metrics={"best_val_smape": 10.0},
    )
    run2 = _make_run(
        params={"random_state": "2", "loss_type": "sgt"},
        metrics={"best_val_smape": 8.0},
    )

    client.search_runs.side_effect = [
        _make_page([run1]),
        _make_page([run2]),
    ]

    df = collect_experiment_results(tracking_uri="sqlite:///test.db")

    assert len(df) == 2
    assert list(df["dataset"]) == ["normal", "lanl"]
    assert list(df["loss_type"]) == ["mse", "sgt"]


@patch("skewed_sequences.experiments.collect_results.MlflowClient")
def test_collect_pagination(mock_client_cls):
    client = mock_client_cls.return_value
    client.search_experiments.return_value = [
        _make_experiment("1", "normal_run_1"),
    ]

    run1 = _make_run(params={"random_state": "1", "loss_type": "mse"}, metrics={})
    run2 = _make_run(params={"random_state": "2", "loss_type": "mae"}, metrics={})

    page1 = _make_page([run1], token="next_page")
    page2 = _make_page([run2], token=None)
    client.search_runs.side_effect = [page1, page2]

    df = collect_experiment_results(tracking_uri="sqlite:///test.db")

    assert len(df) == 2
    assert client.search_runs.call_count == 2


@patch("skewed_sequences.experiments.collect_results.MlflowClient")
def test_created_at_is_utc_datetime(mock_client_cls):
    client = mock_client_cls.return_value
    client.search_experiments.return_value = [
        _make_experiment("1", "normal_run_1"),
    ]

    run = _make_run(
        params={"random_state": "1", "loss_type": "mse"},
        metrics={},
        start_time=1700000000000,
    )
    client.search_runs.return_value = _make_page([run])

    df = collect_experiment_results(tracking_uri="sqlite:///test.db")

    ts = df.iloc[0]["created_at"]
    assert ts.tzinfo is not None
    assert ts.year == 2023
