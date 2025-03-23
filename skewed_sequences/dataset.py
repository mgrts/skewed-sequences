from pathlib import Path

import typer
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from skewed_sequences.config import PROCESSED_DATA_DIR, EXTERNAL_DATA_DIR, SEQUENCE_LENGTH

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
    input_path: Path = EXTERNAL_DATA_DIR / 'dataset.csv',
    output_path: Path = PROCESSED_DATA_DIR / 'dataset.npy',
    sequence_length: int = SEQUENCE_LENGTH,
):
    logger.info('Processing real COVID data')

    covid_data = pd.read_csv(input_path)

    countries = [
        'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czechia', 'Denmark', 'Estonia', 'Finland', 'France',
        'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands',
        'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'United States', 'Russia',
        'Ukraine', 'Belarus', 'Kazakhstan', 'Armenia', 'Azerbaijan', 'Georgia', 'Kyrgyzstan', 'Moldova',
        'Tajikistan', 'Turkmenistan', 'Uzbekistan'
    ]

    # Filter data for the selected countries
    covid_data = covid_data[covid_data['location'].isin(countries)]

    # Select relevant columns and handle missing values
    covid_data = covid_data[['location', 'date', 'new_cases']]
    covid_data['date'] = pd.to_datetime(covid_data['date'])
    covid_data = covid_data.sort_values(['location', 'date'])
    covid_data['new_cases'] = covid_data['new_cases'].fillna(0)  # Fill missing values with 0

    # Apply smoothing to remove zeros (simple moving average)
    covid_data['new_cases'] = covid_data.groupby('location')['new_cases'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )

    # Pivot the data to have dates as rows and locations as columns
    pivot_data = covid_data.pivot(index='date', columns='location', values='new_cases')
    pivot_data = pivot_data.fillna(0)  # Ensure no missing values remain

    owid_sequences = []
    for column_name in pivot_data.columns:
        column = pivot_data[column_name]
        chunks = slice_array_to_chunks(column, sequence_length)

        for chunk in chunks:
            chunk_scaled = StandardScaler().fit_transform(chunk.reshape(-1, 1)).reshape(-1)
            owid_sequences.append(chunk_scaled)

    owid_sequences = np.vstack(owid_sequences)
    owid_sequences = owid_sequences[..., np.newaxis]

    logger.info('Saving processed data')

    with open(output_path, 'wb') as f:
        np.save(f, owid_sequences)

    logger.success('Processing dataset complete.')


if __name__ == '__main__':
    app()
