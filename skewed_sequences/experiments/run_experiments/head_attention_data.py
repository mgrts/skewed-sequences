"""Multi-head attention study on the heavy-tailed synthetic dataset.

Backs the multi-head variance-reduction claim (reviewer A2/A12, Fig. 12). Unlike
the main sweep (fixed ``embed_dim=64``, ``num_heads=4``), this study holds the
*total embedding width* fixed (``EMBED_DIM``) and varies only the head count over
``HEAD_COUNTS``. That isolates the effect of splitting attention across heads
from model capacity — the confound in the original "4x64 vs 1x256" comparison,
where width and head count changed together.

Transformer only. The per-run seed is shared across head counts AND losses, so
every head configuration sees the same data split and initialization seed
(seed-paired; runs are not bit-reproducible — CLAUDE.md #7), giving a clean
within-run comparison of head counts for each loss.
"""

import random

import typer

from skewed_sequences.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    N_RUNS,
    NUM_EPOCHS,
    NUM_WORKERS,
    PROCESSED_DATA_DIR,
    SYNTHETIC_DATA_CONFIGS,
    TRAINING_CONFIGS,
)
from skewed_sequences.data.synthetic.generate_data import main as generate_data_main
from skewed_sequences.experiments.run_experiments._runner import run_training_config

app = typer.Typer(pretty_exceptions_show_locals=False)

# Fixed total embedding width; divisible by every head count below.
EMBED_DIM = 256
HEAD_COUNTS = (1, 2, 4, 8)


def head_sweep_loss_configs() -> list[dict]:
    """Reduced loss set for the head study.

    All classical baselines + four representative symmetric SGT points
    (``p=2``; ``q`` in {1.3, 2.5, 10, 20}), so each loss gets its own
    head-count curve without re-running the full 36-config grid.
    """
    classical = [c for c in TRAINING_CONFIGS if c["loss_type"] != "sgt"]
    sgt = [
        c
        for c in TRAINING_CONFIGS
        if c["loss_type"] == "sgt"
        and c["sgt_loss_lambda"] == 0.0
        and c["sgt_loss_p"] == 2.0
        and c["sgt_loss_q"] in (1.3, 2.5, 10.0, 20.0)
    ]
    return classical + sgt


@app.command()
def main(
    n_runs: int = N_RUNS,
    n_sequences: int = 1000,
    stride: int = 5,
    num_layers: int = 4,
    batch_size: int = BATCH_SIZE,
    num_epochs: int = NUM_EPOCHS,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    num_workers: int = NUM_WORKERS,
):
    heavy_cfg = next(c for c in SYNTHETIC_DATA_CONFIGS if c["experiment_name"] == "heavy-tailed")
    # Dedicated dataset file — must NOT reuse the canonical synthetic_dataset.npy that
    # the main synthetic sweep generates/consumes, or this study (heavy-tailed, reduced
    # n) would silently clobber it and break the main run's reproducibility.
    dataset_path = PROCESSED_DATA_DIR / "head_synthetic_dataset.npy"
    loss_configs = head_sweep_loss_configs()

    typer.echo(
        f"==== Generating heavy-tailed dataset lam={heavy_cfg['lam']}, q={heavy_cfg['q']}, "
        f"sigma={heavy_cfg['sigma']}, kernel_size={heavy_cfg.get('kernel_size', 99)}, "
        f"n_sequences={n_sequences} -> {dataset_path.name} ===="
    )
    generate_data_main(
        output_path=dataset_path,
        lam=heavy_cfg["lam"],
        q=heavy_cfg["q"],
        sigma=heavy_cfg["sigma"],
        kernel_size=heavy_cfg.get("kernel_size", 99),
        n_sequences=n_sequences,
    )
    typer.echo("Dataset generation complete.\n")

    total = n_runs * len(HEAD_COUNTS) * len(loss_configs)
    counter = 0
    for run_idx in range(1, n_runs + 1):
        # One seed per run, shared across head counts AND losses -> seed-paired.
        experiment_seed = random.randint(0, 2**32 - 1)
        experiment_name = f"head-heavy-tailed_run_{run_idx}"

        for num_heads in HEAD_COUNTS:
            for cfg in loss_configs:
                counter += 1
                typer.echo(
                    f"==== [{counter}/{total}] {experiment_name} heads={num_heads} "
                    f"embed_dim={EMBED_DIM} loss={cfg['loss_type']} seed={experiment_seed} ===="
                )
                run_training_config(
                    cfg,
                    dataset_path=dataset_path,
                    experiment_name=experiment_name,
                    seed=experiment_seed,
                    model_type="transformer",
                    stride=stride,
                    embed_dim=EMBED_DIM,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    early_stopping_patience=early_stopping_patience,
                    num_workers=num_workers,
                )
                typer.echo(
                    f"==== Completed: {experiment_name} heads={num_heads} loss={cfg['loss_type']}"
                    " ====\n"
                )


if __name__ == "__main__":
    app()
