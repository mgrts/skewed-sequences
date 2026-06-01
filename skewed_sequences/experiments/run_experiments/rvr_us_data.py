import random

import typer

from skewed_sequences.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    MODEL_TYPES,
    N_RUNS,
    NUM_EPOCHS,
    NUM_WORKERS,
    PROCESSED_DATA_DIR,
    STRIDE,
    TRAINING_CONFIGS,
)
from skewed_sequences.data.rvr_us.dataset import main as create_dataset_main
from skewed_sequences.experiments.run_experiments._runner import run_training_config

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    n_runs: int = N_RUNS,
    stride: int = STRIDE,
    batch_size: int = BATCH_SIZE,
    num_epochs: int = NUM_EPOCHS,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    num_workers: int = NUM_WORKERS,
):
    time_series_list = [
        "average_inpatient_beds_occupied",
        "total_admissions_all_influenza_confirmed_past_7days",
    ]

    training_configs = TRAINING_CONFIGS
    dataset_path = PROCESSED_DATA_DIR / "rvr_us_data.npy"

    total_experiments = len(time_series_list) * len(training_configs) * n_runs * len(MODEL_TYPES)
    experiment_counter = 0

    for time_series in time_series_list:
        if time_series == "average_inpatient_beds_occupied":
            experiment_base_name = "rvr-us-bed-occupancy"
        elif time_series == "total_admissions_all_influenza_confirmed_past_7days":
            experiment_base_name = "rvr-us-influenza-cases"
        else:
            raise ValueError(f"Invalid time series: {time_series}")

        create_dataset_main(time_series=time_series)

        for training_config in training_configs:
            for run_idx in range(1, n_runs + 1):
                for model_type in MODEL_TYPES:
                    experiment_counter += 1
                    experiment_seed = random.randint(0, 2**32 - 1)
                    experiment_name = f"{experiment_base_name}_run_{run_idx}"

                    typer.echo(
                        f"[{experiment_counter}/{total_experiments}] Starting training: "
                        f"{experiment_name} with model={model_type}, config: {training_config}, "
                        f"seed: {experiment_seed}"
                    )

                    run_training_config(
                        training_config,
                        dataset_path=dataset_path,
                        experiment_name=experiment_name,
                        seed=experiment_seed,
                        model_type=model_type,
                        stride=stride,
                        batch_size=batch_size,
                        num_epochs=num_epochs,
                        early_stopping_patience=early_stopping_patience,
                        num_workers=num_workers,
                    )

                    typer.echo(f"Completed training: {experiment_name} ({model_type})\n")


if __name__ == "__main__":
    app()
