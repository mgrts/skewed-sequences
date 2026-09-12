"""OWID COVID-19 daily new cases -> ``(N, 300, 1)`` standardized sequences.

Source: the OWID/JHU daily *wide* file (``date`` x location; see ``config.DATA_URL``).
The previous long-format ``owid-covid-data.csv`` is weekly-reported for its whole
history and produced piecewise-constant sequences with a zero residual scale.
"""

from pathlib import Path
from typing import Annotated

from loguru import logger
import numpy as np
import pandas as pd
import typer

from skewed_sequences.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, SEQUENCE_LENGTH
from skewed_sequences.data._common import (
    check_not_step_function,
    scale_and_stack,
    slice_array_to_chunks,
)
from skewed_sequences.data.owid_covid.load_data import OWID_RAW_FILENAME

app = typer.Typer(pretty_exceptions_show_locals=False)

# The country selection used in the paper (EU + post-Soviet states + United States).
COUNTRIES = [
    "Austria",
    "Belgium",
    "Bulgaria",
    "Croatia",
    "Cyprus",
    "Czechia",
    "Denmark",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "Ireland",
    "Italy",
    "Latvia",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Netherlands",
    "Poland",
    "Portugal",
    "Romania",
    "Slovakia",
    "Slovenia",
    "Spain",
    "Sweden",
    "United States",
    "Russia",
    "Ukraine",
    "Belarus",
    "Kazakhstan",
    "Armenia",
    "Azerbaijan",
    "Georgia",
    "Kyrgyzstan",
    "Moldova",
    "Tajikistan",
    "Turkmenistan",
    "Uzbekistan",
]

# Aggregate columns of the wide file (regions, income groups, "World"): sums of the
# country columns, never a single reporting entity. Excluded from ``--all-locations``
# so no split contains a series that is a sum of series in another split.
AGGREGATE_LOCATIONS = frozenset(
    {
        "World",
        "Africa",
        "Asia",
        "Europe",
        "European Union",
        "North America",
        "South America",
        "Oceania",
        "High income",
        "Low income",
        "Lower middle income",
        "Upper middle income",
        "International",
    }
)


def select_locations(
    wide: pd.DataFrame,
    all_locations: bool,
    max_zero_fraction: float,
    min_days: int,
) -> list[str]:
    """Columns to turn into sequences.

    Default: the paper's ``COUNTRIES`` (missing ones are logged and skipped).
    ``all_locations``: every non-aggregate location with at least ``min_days`` rows
    and at most ``max_zero_fraction`` zero/missing days (a data-driven rule that
    is reported as the selection criterion instead of a hand-picked list).
    """
    if not all_locations:
        present = [c for c in COUNTRIES if c in wide.columns]
        missing = sorted(set(COUNTRIES) - set(present))
        if missing:
            logger.warning(f"{len(missing)} listed countries absent from the file: {missing}")
        return present
    keep = []
    for col in wide.columns:
        if col in AGGREGATE_LOCATIONS:
            continue
        series = wide[col]
        if len(series) < min_days:
            continue
        zero_fraction = float((series.fillna(0) == 0).mean())
        if zero_fraction <= max_zero_fraction:
            keep.append(col)
    return keep


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / OWID_RAW_FILENAME,
    output_path: Path = PROCESSED_DATA_DIR / "dataset.npy",
    sequence_length: int = SEQUENCE_LENGTH,
    rolling_window: int = 7,
    all_locations: Annotated[
        bool,
        typer.Option(
            help="Use every non-aggregate location passing --max-zero-fraction instead of "
            "the paper's fixed country list."
        ),
    ] = False,
    max_zero_fraction: float = 0.15,
    max_zero_increment_fraction: float = 0.5,
):
    logger.info("Processing real COVID data")

    wide = pd.read_csv(input_path)
    if "date" not in wide.columns:
        raise ValueError(
            f"{input_path.name} has no 'date' column — expected the OWID/JHU wide file "
            "(run `skseq data download-owid download`)."
        )
    wide["date"] = pd.to_datetime(wide["date"])
    wide = wide.sort_values("date").set_index("date")

    locations = select_locations(
        wide,
        all_locations=all_locations,
        max_zero_fraction=max_zero_fraction,
        min_days=sequence_length,
    )
    if not locations:
        raise ValueError("No locations selected; check the input file / selection rule.")
    logger.info(f"Using {len(locations)} locations over {len(wide)} days")

    chunks = []
    for location in locations:
        # Missing days -> 0, then a rolling mean (weekend / reporting gaps), as before.
        series = wide[location].fillna(0).rolling(window=rolling_window, min_periods=1).mean()
        chunks.extend(slice_array_to_chunks(series.to_numpy(), sequence_length))

    owid_sequences = scale_and_stack(chunks, sequence_length)
    zero_frac = check_not_step_function(
        owid_sequences, max_zero_increment_fraction, name="OWID COVID"
    )
    logger.info(
        f"{owid_sequences.shape[0]} sequences of length {sequence_length} "
        f"({zero_frac:.1%} zero increments)"
    )

    logger.info("Saving processed data")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        np.save(f, owid_sequences)

    logger.success("Processing dataset complete.")


if __name__ == "__main__":
    app()
