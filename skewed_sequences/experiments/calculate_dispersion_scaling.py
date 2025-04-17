from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from loguru import logger

from skewed_sequences.config import (FIGURES_DIR, PROCESSED_DATA_DIR,
                                     SYNTHETIC_DATA_CONFIGS)
from skewed_sequences.data.rvr_us.dataset import \
    main as create_rvr_dataset_main
from skewed_sequences.data.synthetic.generate_data import \
    main as generate_data_main
from skewed_sequences.metrics import compute_dispersion_scaling_series

app = typer.Typer(pretty_exceptions_show_locals=False)


def save_dispersion_csv(metric_array: np.ndarray, label: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f'{label}_dispersion_scaling.csv'

    df = pd.DataFrame(metric_array, columns=['log_n', 'log_Mn_over_M1', 'Kappa_1n', 'M1', 'Mn'])
    df.to_csv(csv_path, index_label='n')

    logger.info(f'Dispersion CSV saved to {csv_path}')
    return csv_path


def save_dispersion_plot(metric_array: np.ndarray, label: str, output_dir: Path) -> Path:
    plot_path = output_dir / f'{label}_dispersion_plot.png'

    df = pd.DataFrame(metric_array, columns=['log_n', 'log_Mn_over_M1', 'Kappa_1n', 'M1', 'Mn'])

    plt.figure()
    plt.plot(df['M1'], label='$M_1$')
    plt.plot(df['Mn'], label='$M_n$')
    plt.xlabel('n')
    plt.ylabel('Mean Absolute Deviation')
    plt.title(f'Dispersion Scaling - {label}')
    plt.legend()
    plt.grid(True)

    plt.savefig(plot_path)
    plt.close()

    logger.info(f'Dispersion plot saved to {plot_path}')
    return plot_path


def process_and_save_dispersion(data: np.ndarray, label: str, output_dir: Path, num_values: int):
    typer.echo(f'Computing dispersion scaling for {label}...')
    series = compute_dispersion_scaling_series(data, num_values)
    save_dispersion_csv(series, label, output_dir)
    save_dispersion_plot(series, label, output_dir)


@app.command()
def main(output_dir: Path = FIGURES_DIR / 'dispersion_scaling',
         sample_size: int = 1000,
         num_values: int = 100):
    output_dir = Path(output_dir)

    # --- Synthetic Datasets ---
    for config in SYNTHETIC_DATA_CONFIGS:
        lam, q, sigma = config['lam'], config['q'], config['sigma']
        experiment_name = config['experiment_name']

        typer.echo(f'Generating synthetic dataset for {experiment_name}...')
        generate_data_main(lam=lam, q=q, sigma=sigma, n_sequences=sample_size, apply_smoothing=False)
        data = np.load(PROCESSED_DATA_DIR / 'synthetic_dataset.npy')

        process_and_save_dispersion(data, experiment_name, output_dir, num_values)

    # --- RVR Datasets ---
    rvr_tasks = [
        ('average_inpatient_beds_occupied', 'rvr-us-bed-occupancy'),
        ('total_admissions_all_influenza_confirmed_past_7days', 'rvr-us-influenza-cases'),
    ]

    for time_series, label in rvr_tasks:
        typer.echo(f'Creating RVR dataset for {label}...')
        create_rvr_dataset_main(time_series=time_series)
        data = np.load(PROCESSED_DATA_DIR / 'rvr_us_data.npy')

        process_and_save_dispersion(data, label, output_dir, num_values)

    # --- OWID Dataset ---
    typer.echo('Computing dispersion scaling for OWID dataset...')
    try:
        data = np.load(PROCESSED_DATA_DIR / 'dataset.npy')
        process_and_save_dispersion(data, 'covid-owid', output_dir, num_values)
    except Exception as e:
        logger.warning(f'Failed to process OWID dataset: {e}')


if __name__ == '__main__':
    app()
