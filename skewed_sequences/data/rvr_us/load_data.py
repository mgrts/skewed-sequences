from pathlib import Path

from loguru import logger
import typer

from skewed_sequences.config import EXTERNAL_DATA_DIR, RVR_DATA_URL
from skewed_sequences.data.owid_covid.load_data import download_file_with_progress

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def download(
    input_url: str = typer.Option(RVR_DATA_URL, help="URL to download the RVR dataset from."),
    output_path: Path = typer.Option(
        EXTERNAL_DATA_DIR / "rvr_us_data.csv", help="Path to save the downloaded dataset."
    ),
):
    """Download the archived CDC RVR US hospitalization timeseries CSV."""
    logger.info(f"Starting download from {input_url}")
    final_path = download_file_with_progress(input_url, output_path)
    logger.success(f"Download complete: {final_path}")


if __name__ == "__main__":
    app()
