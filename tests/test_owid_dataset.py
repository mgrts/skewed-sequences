"""Tests for skewed_sequences.data.owid_covid.dataset (daily JHU wide file loader)."""

import numpy as np
import pandas as pd
import pytest

from skewed_sequences.data.owid_covid.dataset import (
    AGGREGATE_LOCATIONS,
    COUNTRIES,
    main,
    select_locations,
)

REAL = ["Germany", "France", "Italy"]  # names present in COUNTRIES


def _wide_csv(tmp_path, n_days=700, weekly=False):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-22", periods=n_days, freq="D")
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d")})
    for c in REAL:
        df[c] = rng.poisson(1000, n_days).astype(float) * (
            1 + 0.5 * np.sin(np.arange(n_days) / 30)
        )
    df["Testland"] = rng.poisson(50, n_days).astype(float)  # not in COUNTRIES
    df["World"] = df[REAL + ["Testland"]].sum(axis=1)  # aggregate
    df["Sparse"] = 0.0
    df.loc[::10, "Sparse"] = 5.0  # 90% zero days
    if weekly:  # weekly reporting: total on one weekday, zeros on the other six
        for c in REAL + ["Testland"]:
            df.loc[df.index % 7 != 0, c] = 0.0
    path = tmp_path / "new_cases.csv"
    df.to_csv(path, index=False)
    return path


def test_countries_and_aggregates_are_disjoint():
    assert not (set(COUNTRIES) & AGGREGATE_LOCATIONS)


def test_default_country_list(tmp_path):
    out = tmp_path / "dataset.npy"
    main(input_path=_wide_csv(tmp_path), output_path=out, sequence_length=300)
    arr = np.load(out)
    assert arr.shape == (len(REAL) * 2, 300, 1)  # 700 days -> 2 chunks per country
    assert np.allclose(arr[:, :, 0].mean(axis=1), 0.0, atol=1e-6)  # per-sequence scaled
    assert np.allclose(arr[:, :, 0].std(axis=1), 1.0, atol=1e-3)


def test_all_locations_rule(tmp_path):
    wide = pd.read_csv(_wide_csv(tmp_path)).set_index("date")
    selected = select_locations(wide, all_locations=True, max_zero_fraction=0.15, min_days=300)
    assert set(selected) == set(REAL) | {"Testland"}
    assert "World" not in selected and "Sparse" not in selected

    out = tmp_path / "all.npy"
    main(input_path=_wide_csv(tmp_path), output_path=out, sequence_length=300, all_locations=True)
    assert np.load(out).shape == ((len(REAL) + 1) * 2, 300, 1)


def test_weekly_reported_data_is_rejected(tmp_path):
    """The weekly-rebased OWID file becomes a step function after the rolling mean."""
    with pytest.raises(ValueError, match="piecewise constant"):
        main(
            input_path=_wide_csv(tmp_path, weekly=True),
            output_path=tmp_path / "x.npy",
            sequence_length=300,
        )


def test_long_format_file_is_rejected(tmp_path):
    path = tmp_path / "long.csv"
    pd.DataFrame({"location": ["Germany"], "new_cases": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no 'date' column"):
        main(input_path=path, output_path=tmp_path / "x.npy")
