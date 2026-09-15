from typing import Annotated, List, Optional

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
from skewed_sequences.experiments.run_experiments._runner import (
    draw_experiment_seed,
    run_training_config,
)

app = typer.Typer(pretty_exceptions_show_locals=False)

# The two RVR series and their experiment-name prefixes (names end in _run_<i>).
RVR_SERIES = {
    "average_inpatient_beds_occupied": "rvr-us-bed-occupancy",
    "total_admissions_all_influenza_confirmed_past_7days": "rvr-us-influenza-cases",
}


@app.command()
def main(
    n_runs: int = N_RUNS,
    stride: int = STRIDE,
    batch_size: int = BATCH_SIZE,
    num_epochs: int = NUM_EPOCHS,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    num_workers: int = NUM_WORKERS,
    # --first-run: start at this run index (1-based) so the seeds of one sweep can be
    # split across parallel processes; pairing is within a run index, so a split never
    # breaks it. Each run index draws (or, with --resume, reuses) its own seed.
    first_run: int = 1,
    time_series: Annotated[
        Optional[List[str]],
        typer.Option(
            help="RVR series column(s) to run; repeatable. Default: both series. Use one per "
            "process to run the two series in parallel (each process needs its own "
            "SKSEQ_PROJ_ROOT, because the loader writes rvr_us_data.npy)."
        ),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(
            help="Reuse the seed already logged for an experiment and skip configs that "
            "already have a FINISHED run (continue a killed sweep)."
        ),
    ] = True,
):
    time_series_list = list(time_series) if time_series else list(RVR_SERIES)
    unknown = [s for s in time_series_list if s not in RVR_SERIES]
    if unknown:
        raise typer.BadParameter(f"unknown time series {unknown}; choose from {list(RVR_SERIES)}")
    if not 1 <= first_run <= n_runs:
        raise typer.BadParameter(f"--first-run must be in [1, {n_runs}], got {first_run}")

    training_configs = TRAINING_CONFIGS
    dataset_path = PROCESSED_DATA_DIR / "rvr_us_data.npy"

    total_experiments = (
        len(time_series_list) * len(training_configs) * (n_runs - first_run + 1) * len(MODEL_TYPES)
    )
    experiment_counter = 0

    for series in time_series_list:
        experiment_base_name = RVR_SERIES[series]

        create_dataset_main(time_series=series)

        # Seed drawn once per (run_idx, model_type) and reused across all loss
        # configs, so replicates are seed-paired across loss types (paired Wilcoxon).
        for run_idx in range(first_run, n_runs + 1):
            for model_type in MODEL_TYPES:
                experiment_name = f"{experiment_base_name}_run_{run_idx}"
                experiment_seed = draw_experiment_seed(experiment_name, model_type, resume)

                for training_config in training_configs:
                    experiment_counter += 1

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
                        resume=resume,
                    )

                    typer.echo(f"Completed training: {experiment_name} ({model_type})\n")


if __name__ == "__main__":
    app()
