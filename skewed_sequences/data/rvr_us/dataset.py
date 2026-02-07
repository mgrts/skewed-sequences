from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import typer

from skewed_sequences.config import EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR, SEQUENCE_LENGTH

app = typer.Typer(pretty_exceptions_show_locals=False)


def slice_array_to_chunks(array, chunk_size=300):
    n = len(array)
    n_slices = (n + chunk_size - 1) // chunk_size  # ceil(n / chunk_size)
    chunks = []

    for i in range(n_slices):
        start = i * chunk_size
        end = start + chunk_size
        if end > n:
            end = n
            start = max(0, n - chunk_size)
        chunks.append(array[start:end])

    return np.array(chunks)


@app.command()
def main(
    input_path: Path = EXTERNAL_DATA_DIR / "rvr_us_data.csv",
    output_path: Path = PROCESSED_DATA_DIR / "rvr_us_data.npy",
    # time_series: str = 'total_admissions_all_influenza_confirmed_past_7days',
    time_series: str = "average_inpatient_beds_occupied",
    rolling_window: int = 5,
    sequence_length: int = SEQUENCE_LENGTH,
):
    logger.info("Processing RVR US Hospitalization data")

    rvr_data = pd.read_csv(input_path)

    # Sort data by jurisdiction and date
    rvr_data["collection_date"] = pd.to_datetime(rvr_data["collection_date"])
    rvr_data = rvr_data.sort_values(["jurisdiction", "collection_date"])

    # Handle missing values by forward-filling within each jurisdiction
    rvr_data[time_series] = rvr_data.groupby("jurisdiction")[time_series].transform(
        lambda x: x.ffill().bfill()
    )

    # Apply smoothing (5-day rolling average)
    rvr_data[f"smoothed_{time_series}"] = rvr_data.groupby("jurisdiction")[time_series].transform(
        lambda x: x.rolling(window=rolling_window, min_periods=1).mean()
    )

    sequences = []
    jurisdictions = rvr_data["jurisdiction"].unique()

    for jurisdiction in jurisdictions:
        ts_data = rvr_data[rvr_data["jurisdiction"] == jurisdiction][
            f"smoothed_{time_series}"
        ].values
        chunks = slice_array_to_chunks(ts_data, sequence_length)

        for chunk in chunks:
            chunk_scaled = StandardScaler().fit_transform(chunk.reshape(-1, 1)).reshape(-1)
            sequences.append(chunk_scaled)

    sequences = np.vstack(sequences)
    sequences = sequences[..., np.newaxis]

    logger.info("Saving processed data")

    with open(output_path, "wb") as f:
        np.save(f, sequences)

    logger.success("Processing RVR dataset complete.")


if __name__ == "__main__":
    app()
