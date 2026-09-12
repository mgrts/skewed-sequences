"""Tests for skewed_sequences.data.rvr_us.load_data (SODA download + row-count check)."""

from unittest.mock import MagicMock, patch

import pytest

from skewed_sequences.data.rvr_us.load_data import (
    RVR_SODA_COLUMNS,
    download_rvr,
    expected_row_count,
)

MOD = "skewed_sequences.data.rvr_us.load_data"


def _response(text=None, json=None):
    r = MagicMock()
    r.text = text
    r.json.return_value = json
    r.raise_for_status.return_value = None
    return r


def _soda_csv(n_rows, columns=tuple(RVR_SODA_COLUMNS)):
    header = ",".join(columns)
    rows = [f"AZ,2020-08-{i % 28 + 1:02d}T00:00:00.000,{100 + i},{i % 3}" for i in range(n_rows)]
    return "\n".join([header, *rows]) + "\n"


@patch(f"{MOD}.requests.get")
def test_expected_row_count_uses_json_count(mock_get):
    mock_get.return_value = _response(json=[{"count": "88091"}])
    assert expected_row_count("https://x/resource/abc.csv") == 88091
    url, kwargs = mock_get.call_args.args[0], mock_get.call_args.kwargs
    assert url.endswith("/resource/abc.json") and kwargs["params"]["$select"] == "count(*)"


@patch(f"{MOD}.requests.get")
def test_download_selects_renames_and_verifies(mock_get, tmp_path):
    mock_get.side_effect = [_response(json=[{"count": "5"}]), _response(text=_soda_csv(5))]
    out = tmp_path / "rvr.csv"
    df = download_rvr("https://x/resource/abc.csv", out)

    assert list(df.columns) == list(RVR_SODA_COLUMNS.values())
    assert len(df) == 5 and out.exists()
    csv_call = mock_get.call_args_list[1]
    assert csv_call.kwargs["params"]["$select"] == ",".join(RVR_SODA_COLUMNS)
    assert csv_call.kwargs["params"]["$limit"] >= 5


@patch(f"{MOD}.requests.get")
def test_truncated_response_raises(mock_get, tmp_path):
    mock_get.side_effect = [_response(json=[{"count": "10"}]), _response(text=_soda_csv(5))]
    with pytest.raises(ValueError, match="truncated"):
        download_rvr("https://x/resource/abc.csv", tmp_path / "rvr.csv")
    assert not (tmp_path / "rvr.csv").exists()


@patch(f"{MOD}.requests.get")
def test_missing_column_raises(mock_get, tmp_path):
    cols = [c for c in RVR_SODA_COLUMNS if c != "total_admissions_all_influenza"]
    text = "\n".join([",".join(cols), "AZ,2020-08-01T00:00:00.000,100"]) + "\n"
    mock_get.side_effect = [_response(json=[{"count": "1"}]), _response(text=text)]
    with pytest.raises(ValueError, match="lacks columns"):
        download_rvr("https://x/resource/abc.csv", tmp_path / "rvr.csv")
