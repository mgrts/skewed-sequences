from pathlib import Path
from typing import Annotated

from loguru import logger
import numpy as np
import pandas as pd
import typer

from skewed_sequences.config import EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR, SEQUENCE_LENGTH
from skewed_sequences.data._common import (
    check_not_step_function,
    scale_and_stack,
    slice_array_to_chunks,
)
from skewed_sequences.data.rvr_us.load_data import RVR_RAW_FILENAME

app = typer.Typer(pretty_exceptions_show_locals=False)


def is_aggregate_jurisdiction(name: str) -> bool:
    """``US`` and the ten HHS ``Region N`` rows are sums of the state series."""
    name = str(name).strip()
    return name == "US" or name.startswith("Region ")


@app.command()
def main(
    input_path: Path = EXTERNAL_DATA_DIR / RVR_RAW_FILENAME,
    output_path: Path = PROCESSED_DATA_DIR / "rvr_us_data.npy",
    # time_series: str = 'total_admissions_all_influenza_confirmed_past_7days',
    time_series: str = "average_inpatient_beds_occupied",
    rolling_window: int = 5,
    sequence_length: int = SEQUENCE_LENGTH,
    include_aggregates: Annotated[
        bool,
        typer.Option(
            help="Keep the national (US) and HHS-region aggregate series. They are sums of "
            "the state series, so keeping them leaks near-duplicates across the split."
        ),
    ] = False,
    max_zero_increment_fraction: float = 0.5,
):
    logger.info("Processing RVR US Hospitalization data")

    rvr_data = pd.read_csv(input_path)

    if not include_aggregates:
        mask = rvr_data["jurisdiction"].map(is_aggregate_jurisdiction)
        logger.info(
            f"Dropping {rvr_data.loc[mask, 'jurisdiction'].nunique()} aggregate jurisdictions"
        )
        rvr_data = rvr_data[~mask]

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

    chunks = []
    jurisdictions = rvr_data["jurisdiction"].unique()

    for jurisdiction in jurisdictions:
        ts_data = rvr_data[rvr_data["jurisdiction"] == jurisdiction][
            f"smoothed_{time_series}"
        ].values
        chunks.extend(slice_array_to_chunks(ts_data, sequence_length))

    sequences = scale_and_stack(chunks, sequence_length)
    zero_frac = check_not_step_function(
        sequences, max_zero_increment_fraction, name=f"RVR {time_series}"
    )
    logger.info(
        f"{sequences.shape[0]} sequences from {len(jurisdictions)} jurisdictions "
        f"({zero_frac:.1%} zero increments)"
    )

    logger.info("Saving processed data")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        np.save(f, sequences)

    logger.success("Processing RVR dataset complete.")


if __name__ == "__main__":
    app()
