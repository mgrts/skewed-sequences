"""Tests for skewed_sequences.data.owid_covid.load_data (download validation)."""

import pandas as pd
import pytest

from skewed_sequences.data.owid_covid.load_data import validate_daily_wide_csv


def _csv(tmp_path, n_days=1200, n_locations=12, freq="D", first_col="date"):
    dates = pd.date_range("2020-01-22", periods=n_days, freq=freq)
    df = pd.DataFrame({first_col: dates.strftime("%Y-%m-%d")})
    for i in range(n_locations):
        df[f"Loc{i}"] = float(i)
    path = tmp_path / "f.csv"
    df.to_csv(path, index=False)
    return path


def test_valid_daily_wide_file(tmp_path):
    assert validate_daily_wide_csv(_csv(tmp_path)) == 1200


def test_rejects_long_format(tmp_path):
    with pytest.raises(ValueError, match="leading 'date' column"):
        validate_daily_wide_csv(_csv(tmp_path, first_col="location"))


def test_rejects_truncated_file(tmp_path):
    with pytest.raises(ValueError, match="truncated"):
        validate_daily_wide_csv(_csv(tmp_path, n_days=200))
    with pytest.raises(ValueError, match="truncated"):
        validate_daily_wide_csv(_csv(tmp_path, n_locations=3))


def test_rejects_weekly_cadence(tmp_path):
    with pytest.raises(ValueError, match="not daily"):
        validate_daily_wide_csv(_csv(tmp_path, freq="7D"))
