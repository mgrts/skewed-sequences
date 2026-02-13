from pathlib import Path
import random

import typer

from skewed_sequences.config import (
    CONTEXT_LENGTH,
    N_RUNS,
    PROCESSED_DATA_DIR,
    STRIDE,
    TRAINING_CONFIGS,
)
from skewed_sequences.modeling.train import main as train_main

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    n_runs: int = N_RUNS,
    dataset_name: str = "lanl_sequences.npy",
    batch_size: int = 32,
    num_epochs: int = 100,
    early_stopping_patience: int = 20,
    num_workers: int = 0,
):
    """
    Run multiple training experiments on the LANL dataset.
    """

    dataset_path: Path = PROCESSED_DATA_DIR / dataset_name
    training_configs = TRAINING_CONFIGS

    typer.echo(f"==== Using dataset: {dataset_path.name} ====")

    for train_config in training_configs:
        loss_type = train_config["loss_type"]

        for run_idx in range(1, n_runs + 1):
            experiment_seed = random.randint(0, 2**32 - 1)
            experiment_name = f"lanl_{loss_type}_run_{run_idx}"

            typer.echo(
                f"==== Starting training: {experiment_name} "
                f"with loss_type={loss_type}, seed={experiment_seed} ===="
            )

            if loss_type.lower() == "sgt":
                train_main(
                    dataset_path=dataset_path,
                    loss_type=loss_type,
                    sgt_loss_lambda=train_config["sgt_loss_lambda"],
                    sgt_loss_q=train_config["sgt_loss_q"],
                    sgt_loss_sigma=train_config["sgt_loss_sigma"],
                    sgt_loss_p=train_config["sgt_loss_p"],
                    output_length=train_config.get("output_length", 5),
                    context_length=CONTEXT_LENGTH,
                    stride=STRIDE,
                    experiment_name=experiment_name,
                    seed=experiment_seed,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    early_stopping_patience=early_stopping_patience,
                    num_workers=num_workers,
                )
            else:
                train_main(
                    dataset_path=dataset_path,
                    loss_type=loss_type,
                    output_length=train_config.get("output_length", 5),
                    context_length=CONTEXT_LENGTH,
                    stride=STRIDE,
                    experiment_name=experiment_name,
                    seed=experiment_seed,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    early_stopping_patience=early_stopping_patience,
                    num_workers=num_workers,
                )

            typer.echo(f"==== Completed training: {experiment_name} ====\n")


if __name__ == "__main__":
    app()
