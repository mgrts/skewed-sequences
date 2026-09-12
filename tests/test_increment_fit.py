"""Tests for skewed_sequences.experiments.increment_fit."""

from unittest.mock import patch

import numpy as np
import pandas as pd

from skewed_sequences.experiments.increment_fit import REAL_DATASETS, fit_row, main

MOD = "skewed_sequences.experiments.increment_fit"


def _random_walk(n=30, t=300, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0, 0.2, size=(n, t)), axis=1)[..., np.newaxis]


def test_fit_row_labels_and_keys():
    row = fit_row("demo", _random_walk())
    assert row["dataset"] == "demo"
    assert {"lam", "q", "delta_nll", "increment_skewness"} <= set(row)
    assert abs(row["lam"]) < 0.2  # Gaussian increments: no skew


def test_real_dataset_paths_are_processed_npy():
    assert all(p.suffix == ".npy" for p in REAL_DATASETS.values())


@patch(f"{MOD}.generate_data_main")
def test_main_writes_csv_for_real_files_only(mock_gen, tmp_path):
    npy = tmp_path / "dataset.npy"
    np.save(npy, _random_walk())
    out = tmp_path / "fit.csv"
    with patch.dict(REAL_DATASETS, {"demo": npy, "missing": tmp_path / "nope.npy"}, clear=True):
        main(output_path=out, skip_synthetic=True)
    mock_gen.assert_not_called()
    df = pd.read_csv(out)
    assert list(df["dataset"]) == ["demo"]
    assert "delta_nll" in df.columns
