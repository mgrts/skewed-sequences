from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from skewed_sequences.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SEED,
    SEQUENCE_LENGTH,
)
from skewed_sequences.data._common import scale_and_stack, slice_array_to_chunks

app = typer.Typer(pretty_exceptions_show_locals=False)


def random_sample(
    sequences: np.ndarray,
    n_samples: int = 5_000,
    seed: int = 42,
) -> np.ndarray:
    """
    Randomly sample n_samples sequences without replacement.
    """
    if n_samples > len(sequences):
        raise ValueError(
            f"Requested {n_samples} samples, but dataset contains only {len(sequences)}"
        )

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(sequences), size=n_samples, replace=False)

    return sequences[indices]


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "LANL-Earthquake-Prediction" / "train.csv",
    output_path: Path = PROCESSED_DATA_DIR / "lanl_sequences.npy",
    sequence_length: int = SEQUENCE_LENGTH,
    max_rows: int | None = None,
):
    """
    Create fixed-length normalized sequences from the LANL Earthquake dataset.
    """

    logger.info("Processing LANL Earthquake dataset")

    logger.info("Reading CSV")
    df = pd.read_csv(
        input_path,
        usecols=["acoustic_data"],
        dtype={"acoustic_data": np.int16},
        nrows=max_rows,
    )

    signal = df["acoustic_data"].values.astype(np.float32)
    logger.info(f"Loaded signal length: {len(signal):,}")

    logger.info("Slicing signal into fixed-length sequences")
    chunks = slice_array_to_chunks(signal, sequence_length)

    chunks = random_sample(chunks, n_samples=5000, seed=SEED)

    logger.info(f"Number of sequences: {len(chunks):,}")

    logger.info("Scaling sequences")
    sequences = scale_and_stack(chunks, sequence_length)  # (N, T, 1)

    logger.info(f"Final dataset shape: {sequences.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving processed dataset")
    with open(output_path, "wb") as f:
        np.save(f, sequences)

    logger.success("LANL dataset processing complete.")


if __name__ == "__main__":
    app()
