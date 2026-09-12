from typing import Annotated

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
from skewed_sequences.experiments.run_experiments._runner import (
    draw_experiment_seed,
    run_training_config,
)

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    n_runs: int = N_RUNS,
    stride: int = STRIDE,
    batch_size: int = BATCH_SIZE,
    num_epochs: int = NUM_EPOCHS,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    num_workers: int = NUM_WORKERS,
    resume: Annotated[
        bool,
        typer.Option(
            help="Reuse the seed already logged for an experiment and skip configs that "
            "already have a FINISHED run (continue a killed sweep)."
        ),
    ] = True,
):
    experiment_name_base = "covid-owid"
    dataset_path = PROCESSED_DATA_DIR / "dataset.npy"
    training_configs = TRAINING_CONFIGS

    total_experiments = len(training_configs) * n_runs * len(MODEL_TYPES)
    experiment_counter = 0

    # Seed drawn once per (run_idx, model_type) and reused across all loss
    # configs, so replicates are seed-paired across loss types (paired Wilcoxon).
    for run_idx in range(1, n_runs + 1):
        for model_type in MODEL_TYPES:
            experiment_name = f"{experiment_name_base}_run_{run_idx}"
            experiment_seed = draw_experiment_seed(experiment_name, model_type, resume)

            for training_config in training_configs:
                experiment_counter += 1

                typer.echo(
                    f"[{experiment_counter}/{total_experiments}] Starting training with "
                    f"model={model_type}, config: {training_config}, run: {run_idx}, "
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
                    resume=resume,
                )

                typer.echo(
                    f"Completed training with model={model_type}, config: {training_config}, "
                    f"run: {run_idx}\n"
                )


if __name__ == "__main__":
    app()
