from pathlib import Path

from loguru import logger
import pandas as pd
import requests
from tqdm import tqdm
import typer

from skewed_sequences.config import DATA_URL, RAW_DATA_DIR

app = typer.Typer(pretty_exceptions_show_locals=False)

OWID_RAW_FILENAME = "owid_jhu_new_cases.csv"


def download_file_with_progress(url: str, dest_path: Path, chunk_size: int = 1024) -> Path:
    """
    Downloads a file from a URL with a progress bar.

    Args:
        url (str): URL of the file to download.
        dest_path (Path): Local path to save the downloaded file.
        chunk_size (int): Size of each chunk to read. Defaults to 1024 bytes.

    Returns:
        Path: The path to the saved file.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to start download: {e}")
        raise typer.Exit(code=1)

    total_size = int(response.headers.get("content-length", 0))

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with tqdm(
        total=total_size, unit="iB", unit_scale=True, desc="Downloading"
    ) as progress_bar, open(dest_path, "wb") as file:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                file.write(chunk)
                progress_bar.update(len(chunk))

    # Only a SHORT read is a real truncation. Servers that gzip the response (e.g.
    # GitHub raw) report the compressed length in content-length while
    # iter_content yields the larger decompressed stream, so ``n > total_size`` is
    # normal and must not be treated as corruption. When content-length is absent
    # (``total_size == 0``) a truncated stream is NOT detectable here — callers must
    # validate the content (see ``validate_daily_wide_csv``).
    if total_size != 0 and progress_bar.n < total_size:
        logger.error("Download incomplete or corrupted.")
        raise typer.Exit(code=1)

    return dest_path


def validate_daily_wide_csv(path: Path, min_days: int = 1000, min_locations: int = 10) -> int:
    """Check that ``path`` is the daily OWID/JHU wide file (``date`` x locations).

    Guards against (a) a truncated download (content-length is not always sent)
    and (b) accidentally pointing at the weekly-rebased ``owid-covid-data.csv``
    (long format, no ``date`` header as first column). Returns the number of days.
    """
    header = pd.read_csv(path, nrows=0).columns.tolist()
    if not header or header[0] != "date":
        raise ValueError(
            f"{path.name}: expected the OWID/JHU wide file with a leading 'date' column, "
            f"got columns {header[:5]}..."
        )
    if len(header) - 1 < min_locations:
        raise ValueError(f"{path.name}: only {len(header) - 1} location columns; truncated?")
    dates = pd.to_datetime(pd.read_csv(path, usecols=["date"])["date"])
    n_days = len(dates)
    if n_days < min_days:
        raise ValueError(f"{path.name}: only {n_days} rows (< {min_days}); truncated download?")
    gaps = dates.sort_values().diff().dt.days.dropna()
    if gaps.median() != 1:
        raise ValueError(
            f"{path.name}: median gap between rows is {gaps.median()} days, not daily data."
        )
    return n_days


@app.command()
def download(
    input_url: str = typer.Option(DATA_URL, help="URL to download the dataset from."),
    output_path: Path = typer.Option(
        RAW_DATA_DIR / OWID_RAW_FILENAME, help="Path to save the downloaded dataset."
    ),
):
    """Download the daily OWID/JHU ``new_cases.csv`` (wide: date x location)."""
    logger.info(f"Starting download from {input_url}")

    final_path = download_file_with_progress(input_url, output_path)
    n_days = validate_daily_wide_csv(final_path)

    logger.success(f"Download complete: {final_path} ({n_days} daily rows)")


if __name__ == "__main__":
    app()
