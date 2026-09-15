"""Tests for the --first-run split and the RVR --time-series selection (parallel sweeps)."""

from unittest.mock import patch

import pytest
import typer

from skewed_sequences.config import TRAINING_CONFIGS
from skewed_sequences.experiments.run_experiments import owid_covid_data, rvr_us_data

OWID = "skewed_sequences.experiments.run_experiments.owid_covid_data"
RVR = "skewed_sequences.experiments.run_experiments.rvr_us_data"


@patch(f"{OWID}.draw_experiment_seed", side_effect=lambda name, *_: hash(name) % 1000)
@patch(f"{OWID}.run_training_config")
def test_owid_first_run_covers_only_the_upper_half(mock_run, mock_seed):
    owid_covid_data.main(n_runs=10, first_run=6, resume=False)
    names = {c.kwargs["experiment_name"] for c in mock_run.call_args_list}
    assert names == {f"covid-owid_run_{i}" for i in range(6, 11)}
    assert mock_run.call_count == 5 * len(TRAINING_CONFIGS)


@patch(f"{OWID}.draw_experiment_seed", return_value=1)
@patch(f"{OWID}.run_training_config")
def test_owid_first_run_out_of_range(mock_run, mock_seed):
    with pytest.raises(typer.BadParameter):
        owid_covid_data.main(n_runs=10, first_run=11, resume=False)
    with pytest.raises(typer.BadParameter):
        owid_covid_data.main(n_runs=10, first_run=0, resume=False)
    mock_run.assert_not_called()


@patch(f"{RVR}.draw_experiment_seed", return_value=1)
@patch(f"{RVR}.run_training_config")
@patch(f"{RVR}.create_dataset_main")
def test_rvr_single_series_and_split(mock_ds, mock_run, mock_seed):
    rvr_us_data.main(
        n_runs=10,
        first_run=6,
        time_series=["total_admissions_all_influenza_confirmed_past_7days"],
        resume=False,
    )
    mock_ds.assert_called_once_with(
        time_series="total_admissions_all_influenza_confirmed_past_7days"
    )
    names = {c.kwargs["experiment_name"] for c in mock_run.call_args_list}
    assert names == {f"rvr-us-influenza-cases_run_{i}" for i in range(6, 11)}
    assert mock_run.call_count == 5 * len(TRAINING_CONFIGS)


@patch(f"{RVR}.draw_experiment_seed", return_value=1)
@patch(f"{RVR}.run_training_config")
@patch(f"{RVR}.create_dataset_main")
def test_rvr_default_runs_both_series(mock_ds, mock_run, mock_seed):
    rvr_us_data.main(n_runs=1, resume=False)
    assert mock_ds.call_count == 2
    names = {c.kwargs["experiment_name"] for c in mock_run.call_args_list}
    assert names == {"rvr-us-bed-occupancy_run_1", "rvr-us-influenza-cases_run_1"}


def test_rvr_unknown_series_rejected():
    with pytest.raises(typer.BadParameter, match="unknown time series"):
        rvr_us_data.main(n_runs=1, time_series=["nope"], resume=False)


def test_proj_root_env_override(tmp_path):
    """SKSEQ_PROJ_ROOT relocates every derived path (parallel slots get their own store).

    Checked in a fresh interpreter: config.py is imported once per process and its
    loguru handler swap cannot be re-run via importlib.reload.
    """
    import os
    import subprocess
    import sys

    code = (
        "from skewed_sequences.config import PROJ_ROOT, TRACKING_URI, PROCESSED_DATA_DIR; "
        "print(PROJ_ROOT); print(TRACKING_URI); print(PROCESSED_DATA_DIR)"
    )
    env = {**os.environ, "SKSEQ_PROJ_ROOT": str(tmp_path)}
    out = (
        subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
        )
        .stdout.strip()
        .splitlines()
    )
    assert out[-3] == str(tmp_path)
    assert out[-2] == f"sqlite:///{tmp_path / 'mlruns.db'}"
    assert out[-1] == str(tmp_path / "data" / "processed")
