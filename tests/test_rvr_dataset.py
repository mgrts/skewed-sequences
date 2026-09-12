"""Tests for skewed_sequences.data.rvr_us.dataset (aggregate filter + step check)."""

import numpy as np
import pandas as pd
import pytest

from skewed_sequences.data.rvr_us.dataset import is_aggregate_jurisdiction, main

BED = "average_inpatient_beds_occupied"
FLU = "total_admissions_all_influenza_confirmed_past_7days"


def _csv(tmp_path, n_days=700, step=False):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-08-01", periods=n_days, freq="D")
    frames = []
    for j in ["AZ", "CA"]:
        beds = 5000 + np.cumsum(rng.normal(0, 30, n_days))
        if step:
            beds = np.repeat(beds[::30], 30)[:n_days]  # constant within each 30-day block
        frames.append(
            pd.DataFrame(
                {
                    "jurisdiction": j,
                    "collection_date": dates.strftime("%Y-%m-%dT00:00:00.000"),
                    BED: beds,
                    FLU: rng.poisson(3, n_days).astype(float),
                }
            )
        )
    states = pd.concat(frames)
    aggregates = []
    for name in ["US", "Region 9"]:
        agg = states.groupby("collection_date", as_index=False)[[BED, FLU]].sum()
        agg.insert(0, "jurisdiction", name)
        aggregates.append(agg)
    df = pd.concat([states, *aggregates], ignore_index=True)
    path = tmp_path / "rvr.csv"
    df.to_csv(path, index=False)
    return path


@pytest.mark.parametrize(
    "name,expected",
    [("US", True), ("Region 1", True), ("Region 10", True), ("AZ", False), ("PR", False)],
)
def test_is_aggregate_jurisdiction(name, expected):
    assert is_aggregate_jurisdiction(name) is expected


def test_aggregates_dropped_by_default(tmp_path):
    out = tmp_path / "rvr.npy"
    main(input_path=_csv(tmp_path), output_path=out, sequence_length=300)
    assert np.load(out).shape == (2 * 2, 300, 1)  # 2 states x 2 chunks of 300


def test_include_aggregates(tmp_path):
    out = tmp_path / "rvr.npy"
    main(input_path=_csv(tmp_path), output_path=out, sequence_length=300, include_aggregates=True)
    assert np.load(out).shape == (4 * 2, 300, 1)


def test_second_series(tmp_path):
    out = tmp_path / "rvr.npy"
    main(input_path=_csv(tmp_path), output_path=out, sequence_length=300, time_series=FLU)
    arr = np.load(out)
    assert arr.shape == (4, 300, 1)
    assert np.allclose(arr[:, :, 0].mean(axis=1), 0.0, atol=1e-6)


def test_step_data_rejected(tmp_path):
    with pytest.raises(ValueError, match="piecewise constant"):
        main(input_path=_csv(tmp_path, step=True), output_path=tmp_path / "x.npy")
