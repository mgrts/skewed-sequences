"""Download the archived CDC RVR US hospitalization timeseries via the SODA API.

The Socrata ``rows.csv?accessType=DOWNLOAD`` export streams without a
content-length, so a dropped connection produced a "successful" 3.8 MB file with
4 of 65 jurisdictions (8 training sequences) and nothing noticed. The SODA
resource endpoint (``config.RVR_DATA_URL``) supports ``$select`` (only the four
columns the loader needs, ~3 MB instead of ~150 MB), ``$order`` and ``$limit``,
and its row count can be verified against ``$select=count(*)`` before the file
is accepted.
"""

import io
from pathlib import Path

from loguru import logger
import pandas as pd
import requests
import typer

from skewed_sequences.config import EXTERNAL_DATA_DIR, RVR_DATA_URL

app = typer.Typer(pretty_exceptions_show_locals=False)

RVR_RAW_FILENAME = "rvr_us_hospitalization_daily.csv"

# SODA field name -> column name the loader (``rvr_us/dataset.py``) expects. The API
# truncates the human-readable export names; both were position-mapped and
# value-checked against the export (AZ 2020-11-09: 9419.71 / 15.0).
RVR_SODA_COLUMNS = {
    "jurisdiction": "jurisdiction",
    "collection_date": "collection_date",
    "average_inpatient_beds_1": "average_inpatient_beds_occupied",
    "total_admissions_all_influenza": "total_admissions_all_influenza_confirmed_past_7days",
}


def expected_row_count(base_url: str, timeout: float = 60) -> int:
    """Row count of the SODA resource (``$select=count(*)`` on the ``.json`` endpoint)."""
    json_url = base_url.rsplit(".", 1)[0] + ".json"
    response = requests.get(json_url, params={"$select": "count(*)"}, timeout=timeout)
    response.raise_for_status()
    return int(response.json()[0]["count"])


def download_rvr(
    base_url: str,
    output_path: Path,
    columns: dict[str, str] = RVR_SODA_COLUMNS,
    timeout: float = 600,
) -> pd.DataFrame:
    """Fetch the selected columns, rename them, verify the row count, write CSV."""
    n_expected = expected_row_count(base_url, timeout=timeout)
    logger.info(f"SODA reports {n_expected} rows; downloading {len(columns)} columns")

    response = requests.get(
        base_url,
        params={
            "$select": ",".join(columns),
            "$order": "jurisdiction,collection_date",
            "$limit": n_expected + 1000,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"SODA response lacks columns {missing}; the API field names changed? "
            f"got {df.columns.tolist()}"
        )
    df = df.rename(columns=columns)
    if len(df) != n_expected:
        raise ValueError(
            f"Downloaded {len(df)} rows but the resource has {n_expected}: truncated response."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


@app.command()
def download(
    input_url: str = typer.Option(RVR_DATA_URL, help="SODA resource URL (``.csv``)."),
    output_path: Path = typer.Option(
        EXTERNAL_DATA_DIR / RVR_RAW_FILENAME, help="Path to save the downloaded dataset."
    ),
):
    """Download the archived CDC RVR US hospitalization timeseries (row-count verified)."""
    logger.info(f"Starting download from {input_url}")
    df = download_rvr(input_url, output_path)
    logger.success(
        f"Download complete: {output_path} ({len(df)} rows, "
        f"{df['jurisdiction'].nunique()} jurisdictions)"
    )


if __name__ == "__main__":
    app()
