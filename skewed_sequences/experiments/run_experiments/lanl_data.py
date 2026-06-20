from pathlib import Path
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
from skewed_sequences.experiments.run_experiments._runner import run_training_config

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    n_runs: int = N_RUNS,
    dataset_name: str = "lanl_sequences.npy",
    stride: int = STRIDE,
    batch_size: int = BATCH_SIZE,
    num_epochs: int = NUM_EPOCHS,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    num_workers: int = NUM_WORKERS,
):
    """
    Run multiple training experiments on the LANL dataset.
    """

    dataset_path: Path = PROCESSED_DATA_DIR / dataset_name
    training_configs = TRAINING_CONFIGS

    total_experiments = len(training_configs) * n_runs * len(MODEL_TYPES)
    experiment_counter = 0

    typer.echo(f"==== Using dataset: {dataset_path.name} ====")

    # Seed drawn once per (run_idx, model_type) and reused across all loss
    # configs, so replicates are seed-paired across loss types (paired Wilcoxon).
    for run_idx in range(1, n_runs + 1):
        for model_type in MODEL_TYPES:
            experiment_seed = random.randint(0, 2**32 - 1)

            for train_config in training_configs:
                loss_type = train_config["loss_type"]
                experiment_counter += 1
                experiment_name = f"lanl_{loss_type}_run_{run_idx}"

                typer.echo(
                    f"==== [{experiment_counter}/{total_experiments}] Starting training: "
                    f"{experiment_name} with model={model_type}, loss_type={loss_type}, "
                    f"seed={experiment_seed} ===="
                )

                run_training_config(
                    train_config,
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

                typer.echo(f"==== Completed training: {experiment_name} ({model_type}) ====\n")


if __name__ == "__main__":
    app()
