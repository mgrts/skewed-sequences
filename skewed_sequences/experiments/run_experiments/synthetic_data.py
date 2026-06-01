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
    SYNTHETIC_DATA_CONFIGS,
    TRAINING_CONFIGS,
)
from skewed_sequences.data.synthetic.generate_data import main as generate_data_main
from skewed_sequences.experiments.run_experiments._runner import run_training_config

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    n_runs: int = N_RUNS,
    n_sequences: int = 10000,
    stride: int = STRIDE,
    batch_size: int = BATCH_SIZE,
    num_epochs: int = NUM_EPOCHS,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    num_workers: int = NUM_WORKERS,
    exp_transform: bool = False,
    exp_scale: float = 0.1,
):
    dataset_path = PROCESSED_DATA_DIR / "synthetic_dataset.npy"
    dataset_configs = SYNTHETIC_DATA_CONFIGS
    training_configs = TRAINING_CONFIGS

    total_experiments = len(dataset_configs) * len(training_configs) * n_runs * len(MODEL_TYPES)
    experiment_counter = 0

    for ds_config in dataset_configs:
        lam = ds_config["lam"]
        q = ds_config["q"]
        sigma = ds_config["sigma"]
        kernel_size = ds_config.get("kernel_size", 99)
        base_experiment_name = ds_config["experiment_name"]
        if exp_transform:
            base_experiment_name = f"exp-{base_experiment_name}"

        typer.echo(
            f"==== Generating dataset with lam={lam}, q={q}, sigma={sigma}, "
            f"kernel_size={kernel_size}, exp_transform={exp_transform} ===="
        )

        generate_data_main(
            lam=lam,
            q=q,
            sigma=sigma,
            kernel_size=kernel_size,
            n_sequences=n_sequences,
            exp_transform=exp_transform,
            exp_scale=exp_scale,
        )

        typer.echo("Dataset generation complete.\n")

        for train_config in training_configs:
            loss_type = train_config["loss_type"]

            for run_idx in range(1, n_runs + 1):
                for model_type in MODEL_TYPES:
                    experiment_counter += 1
                    experiment_seed = random.randint(0, 2**32 - 1)
                    experiment_name = f"{base_experiment_name}_run_{run_idx}"

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
                        exp_transform=exp_transform,
                    )
                    typer.echo(f"==== Completed training: {experiment_name} ({model_type}) ====\n")


if __name__ == "__main__":
    app()
