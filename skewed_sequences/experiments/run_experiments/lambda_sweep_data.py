"""Fine skew (lambda) sub-sweep on the skewed synthetic datasets.

The main grid's skewed configs (``lambda`` in {0.5, 0.9}) lost to their symmetric
counterparts on every dataset, and the increment-likelihood diagnostic explains
why: the optimal ``lambda`` on ``heavy-tailed-skewed`` is about 0.5 while 0.9 is
far too aggressive, and the smoothed ``normal-skewed`` increments carry no skew at
all. This runner appends ``LAMBDA_SWEEP_CONFIGS`` (``lambda`` in {0.1, 0.2, 0.3} at
the (p, q) anchors whose ``lambda = 0`` versions are already in the main grid) to
the *existing* ``<dataset>_run_<i>`` experiments. With ``resume`` (default) each
experiment's logged seed is reused, so the new runs are seed-paired with the
symmetric and classical runs already there and ``aggregate_results.lambda_effect``
can test them with the paired Wilcoxon.

Datasets are regenerated exactly as ``run-synthetic`` does (same generator, same
``SEED``, same ``config.SYNTHETIC_N_SEQUENCES`` / ``SYNTHETIC_STRIDE`` defaults). The
stride is logged per run, so an experiment whose logged stride differs from the
requested one is refused; ``n_sequences`` is not logged, so keep it at the config value.
"""

from typing import Annotated, List, Optional

import typer

from skewed_sequences.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    LAMBDA_SWEEP_CONFIGS,
    MODEL_TYPES,
    N_RUNS,
    NUM_EPOCHS,
    NUM_WORKERS,
    PROCESSED_DATA_DIR,
    SYNTHETIC_DATA_CONFIGS,
    SYNTHETIC_N_SEQUENCES,
    SYNTHETIC_STRIDE,
)
from skewed_sequences.data.synthetic.generate_data import main as generate_data_main
from skewed_sequences.experiments.run_experiments._runner import (
    draw_experiment_seed,
    logged_experiment_param,
    run_training_config,
)

app = typer.Typer(pretty_exceptions_show_locals=False)

DEFAULT_DATASETS = ("heavy-tailed-skewed",)


def lambda_sweep_loss_configs() -> list[dict]:
    return list(LAMBDA_SWEEP_CONFIGS)


@app.command()
def main(
    n_runs: int = N_RUNS,
    n_sequences: int = SYNTHETIC_N_SEQUENCES,
    stride: int = SYNTHETIC_STRIDE,
    dataset: Annotated[
        Optional[List[str]],
        typer.Option(
            help="Synthetic dataset name(s) from SYNTHETIC_DATA_CONFIGS; repeatable. "
            f"Default: {', '.join(DEFAULT_DATASETS)}."
        ),
    ] = None,
    batch_size: int = BATCH_SIZE,
    num_epochs: int = NUM_EPOCHS,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    num_workers: int = NUM_WORKERS,
    # --first-run: start at this run index (1-based) so the seeds of one sweep can be
    # split across parallel processes; pairing is within a run index, so a split never
    # breaks it. Each run index draws (or, with --resume, reuses) its own seed.
    first_run: int = 1,
    resume: Annotated[
        bool,
        typer.Option(
            help="Reuse the seed already logged for an experiment (keeps the new lambda "
            "runs seed-paired with the main sweep) and skip configs that already FINISHED."
        ),
    ] = True,
):
    datasets = list(DEFAULT_DATASETS) if not dataset else list(dataset)
    configs_by_name = {cfg["experiment_name"]: cfg for cfg in SYNTHETIC_DATA_CONFIGS}
    unknown = [name for name in datasets if name not in configs_by_name]
    if unknown:
        raise typer.BadParameter(
            f"unknown dataset(s) {unknown}; choose from {sorted(configs_by_name)}"
        )

    dataset_path = PROCESSED_DATA_DIR / "synthetic_dataset.npy"
    loss_configs = lambda_sweep_loss_configs()
    if not 1 <= first_run <= n_runs:
        raise typer.BadParameter(f"--first-run must be in [1, {n_runs}], got {first_run}")
    total = len(datasets) * (n_runs - first_run + 1) * len(MODEL_TYPES) * len(loss_configs)
    counter = 0

    for name in datasets:
        ds_config = configs_by_name[name]
        typer.echo(
            f"==== Generating dataset {name}: lam={ds_config['lam']}, q={ds_config['q']}, "
            f"sigma={ds_config['sigma']}, kernel_size={ds_config.get('kernel_size', 99)}, "
            f"n_sequences={n_sequences} ===="
        )
        generate_data_main(
            lam=ds_config["lam"],
            q=ds_config["q"],
            sigma=ds_config["sigma"],
            kernel_size=ds_config.get("kernel_size", 99),
            n_sequences=n_sequences,
        )
        typer.echo("Dataset generation complete.\n")

        for run_idx in range(first_run, n_runs + 1):
            for model_type in MODEL_TYPES:
                experiment_name = f"{name}_run_{run_idx}"
                if resume:
                    logged_stride = logged_experiment_param(experiment_name, model_type, "stride")
                    if logged_stride is not None and int(logged_stride) != stride:
                        raise typer.BadParameter(
                            f"{experiment_name} was trained with stride={logged_stride}; "
                            f"refusing to append lambda runs at stride={stride} (the paired "
                            "comparison needs identical data). Pass the same --stride."
                        )
                experiment_seed = draw_experiment_seed(experiment_name, model_type, resume)

                for cfg in loss_configs:
                    counter += 1
                    typer.echo(
                        f"==== [{counter}/{total}] {experiment_name} model={model_type} "
                        f"p={cfg['sgt_loss_p']} q={cfg['sgt_loss_q']} "
                        f"lambda={cfg['sgt_loss_lambda']} seed={experiment_seed} ===="
                    )
                    run_training_config(
                        cfg,
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
                    typer.echo(f"==== Completed: {experiment_name} ({model_type}) ====\n")


if __name__ == "__main__":
    app()
