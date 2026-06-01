from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from skewed_sequences.config import EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR, SEQUENCE_LENGTH
from skewed_sequences.data._common import scale_and_stack, slice_array_to_chunks

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    input_path: Path = EXTERNAL_DATA_DIR / "health_fitness_data.csv",
    output_path: Path = PROCESSED_DATA_DIR / "health_fitness_data.npy",
    time_series: str = "hours_sleep",
    # time_series: str = 'avg_heart_rate',
    sequence_length: int = SEQUENCE_LENGTH,
    rolling_window: int = 20,
):
    logger.info("Processing Fitness Tracker data")

    data = pd.read_csv(input_path)

    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["participant_id", "date"])

    data[time_series] = data.groupby("participant_id")[time_series].transform(
        lambda x: x.ffill().bfill()
    )

    data[f"smoothed_{time_series}"] = data.groupby("participant_id")[time_series].transform(
        lambda x: x.rolling(window=rolling_window, min_periods=1).mean()
    )

    chunks = []
    participant_ids = data["participant_id"].unique()

    for participant_id in participant_ids:
        ts_data = data[data["participant_id"] == participant_id][f"smoothed_{time_series}"].values
        chunks.extend(slice_array_to_chunks(ts_data, sequence_length))

    sequences = scale_and_stack(chunks, sequence_length)

    logger.info("Saving processed data")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        np.save(f, sequences)

    logger.success("Processing complete")


if __name__ == "__main__":
    app()
