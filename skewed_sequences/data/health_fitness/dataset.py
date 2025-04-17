from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.preprocessing import StandardScaler

from skewed_sequences.config import EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR

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
    input_path: Path = EXTERNAL_DATA_DIR / 'health_fitness_data.csv',
    output_path: Path = PROCESSED_DATA_DIR / 'health_fitness_data.npy',
    time_series: str = 'hours_sleep',
    # time_series: str = 'avg_heart_rate',
    sequence_length: int = 200,
    rolling_window: int = 20,
):
    logger.info('Processinng Fitness Tracker data')

    data = pd.read_csv(input_path)

    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(['participant_id', 'date'])

    data[time_series] = (
        data.groupby('participant_id')[time_series]
        .transform(lambda x: x.ffill().bfill())
    )

    data[f'smoothed_{time_series}'] = (
        data.groupby('participant_id')[time_series]
        .transform(lambda x: x.rolling(window=rolling_window, min_periods=1).mean())
    )

    sequences = []
    participant_ids = data['participant_id'].unique()

    for participant_id in participant_ids:
        ts_data = data[data['participant_id'] == participant_id][f'smoothed_{time_series}'].values

        if len(ts_data) >= sequence_length:
            chunks = slice_array_to_chunks(ts_data, sequence_length)

            for chunk in chunks:
                chunk_scaled = StandardScaler().fit_transform(chunk.reshape(-1, 1)).reshape(-1)
                sequences.append(chunk_scaled)

    sequences = np.vstack(sequences)
    sequences = sequences[..., np.newaxis]

    logger.info('Saving processed data')

    with open(output_path, 'wb') as f:
        np.save(f, sequences)

    logger.success('Processing complete')


if __name__ == '__main__':
    app()
