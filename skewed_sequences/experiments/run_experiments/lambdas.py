import typer

from skewed_sequences.config import (
    BATCH_SIZE,
    CONTEXT_LENGTH,
    EARLY_STOPPING_PATIENCE,
    NUM_EPOCHS,
    NUM_WORKERS,
    OUTPUT_LENGTH,
    PROCESSED_DATA_DIR,
    SGT_LOSS_LAMBDAS,
    STRIDE,
)
from skewed_sequences.modeling.train import main as train_main


def main(
    stride: int = STRIDE,
    batch_size: int = BATCH_SIZE,
    num_epochs: int = NUM_EPOCHS,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    num_workers: int = NUM_WORKERS,
):
    """
    Runs training experiments for each value in SGT_LOSS_LAMBDAS.
    """
    total_experiments = len(SGT_LOSS_LAMBDAS)
    dataset_path = PROCESSED_DATA_DIR / "synthetic_dataset.npy"
    for experiment_counter, sgt_lambda in enumerate(SGT_LOSS_LAMBDAS, 1):
        typer.echo(
            f"[{experiment_counter}/{total_experiments}] Starting training with sgt_lambda = {sgt_lambda}"
        )
        # Call the main training function with the current sgt_lambda
        train_main(
            dataset_path=dataset_path,
            loss_type="sgt",
            sgt_loss_lambda=sgt_lambda,
            output_length=OUTPUT_LENGTH,
            context_length=CONTEXT_LENGTH,
            stride=stride,
            batch_size=batch_size,
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
            num_workers=num_workers,
        )
        typer.echo(f"Completed training with sgt_lambda = {sgt_lambda}\n")


if __name__ == "__main__":
    main()
