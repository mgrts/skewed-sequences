import numpy as np
import typer
from loguru import logger

from skewed_sequences.config import PROCESSED_DATA_DIR, SYNTHETIC_DATA_CONFIGS
from skewed_sequences.data.rvr_us.dataset import \
    main as create_rvr_dataset_main
from skewed_sequences.data.synthetic.generate_data import \
    main as generate_data_main
from skewed_sequences.metrics import kappa, skewness

app = typer.Typer(pretty_exceptions_show_locals=False)


def skewness_of_diff(x: np.ndarray) -> float:
    if x.ndim != 3 or x.shape[2] != 1:
        raise ValueError("Input array must have shape (n_sequences, sequence_length, 1)")

    diffs = np.diff(x[:, :, 0], axis=1)
    flat_diffs = diffs.flatten()
    return skewness(flat_diffs)


@app.command()
def main(sample_size: int = 1000, n_for_kappa: int = 10):
    synthetic_path = PROCESSED_DATA_DIR / 'synthetic_dataset.npy'
    rvr_path = PROCESSED_DATA_DIR / 'rvr_us_data.npy'
    owid_path = PROCESSED_DATA_DIR / 'dataset.npy'

    dataset_configs = SYNTHETIC_DATA_CONFIGS

    results = []

    # --- Synthetic datasets ---
    for config in dataset_configs:
        lam = config['lam']
        q = config['q']
        sigma = config.get('sigma', 1.0)
        experiment_name = config['experiment_name']

        typer.echo(f'Generating dataset for {experiment_name}...')
        generate_data_main(lam=lam, q=q, sigma=sigma, n_sequences=sample_size, apply_smoothing=False)

        data = np.load(synthetic_path)
        flat_data = data.flatten()

        typer.echo(f'Computing metrics for {experiment_name}...')
        try:
            kappa_value = kappa(flat_data, n_for_kappa)
        except Exception as e:
            logger.warning(f'Failed to compute kappa for {experiment_name}: {e}')
            kappa_value = np.nan

        try:
            skew_value = skewness(flat_data)
        except Exception as e:
            logger.warning(f'Failed to compute skewness for {experiment_name}: {e}')
            skew_value = np.nan

        try:
            skew_diff_value = skewness_of_diff(data)
        except Exception as e:
            logger.warning(f'Failed to compute diff skewness for {experiment_name}: {e}')
            skew_diff_value = np.nan

        results.append({
            'experiment': experiment_name,
            'kappa': kappa_value,
            'skewness': skew_value,
            'diff_skewness': skew_diff_value
        })

    # --- RVR datasets ---
    for time_series, label in [
        ('average_inpatient_beds_occupied', 'Transformer-RVR-US-bed-occupancy'),
        ('total_admissions_all_influenza_confirmed_past_7days', 'Transformer-RVR-US-influenza-cases')
    ]:
        typer.echo(f'Creating RVR dataset for {label}...')
        create_rvr_dataset_main(time_series=time_series)
        data = np.load(rvr_path)
        flat_data = data.flatten()

        typer.echo(f'Computing metrics for {label}...')
        try:
            kappa_value = kappa(flat_data, n_for_kappa)
        except Exception as e:
            logger.warning(f'Failed to compute kappa for {label}: {e}')
            kappa_value = np.nan

        try:
            skew_value = skewness(flat_data)
        except Exception as e:
            logger.warning(f'Failed to compute skewness for {label}: {e}')
            skew_value = np.nan

        try:
            skew_diff_value = skewness_of_diff(data)
        except Exception as e:
            logger.warning(f'Failed to compute diff skewness for {label}: {e}')
            skew_diff_value = np.nan

        results.append({
            'experiment': label,
            'kappa': kappa_value,
            'skewness': skew_value,
            'diff_skewness': skew_diff_value
        })

    # --- OWID dataset ---
    typer.echo('Computing metrics for OWID dataset...')
    try:
        data = np.load(owid_path)
        flat_data = data.flatten()

        kappa_value = kappa(flat_data, n_for_kappa)
        skew_value = skewness(flat_data)
        skew_diff_value = skewness_of_diff(data)

        results.append({
            'experiment': 'Transformer-OWID',
            'kappa': kappa_value,
            'skewness': skew_value,
            'diff_skewness': skew_diff_value
        })
    except Exception as e:
        logger.warning(f'Failed to process OWID dataset: {e}')

    # --- Output results ---
    typer.echo('\nSummary of results:')
    for r in results:
        typer.echo(f"{r['experiment']}: Kappa={r['kappa']:.4f}, Skewness={r['skewness']:.4f}, Diff Skewness={r['diff_skewness']:.4f}")


if __name__ == '__main__':
    app()
