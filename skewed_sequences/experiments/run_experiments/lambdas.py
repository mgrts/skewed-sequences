import typer

from skewed_sequences.config import (
    CONTEXT_LENGTH,
    OUTPUT_LENGTH,
    PROCESSED_DATA_DIR,
    SGT_LOSS_LAMBDAS,
    STRIDE,
)
from skewed_sequences.modeling.train import main as train_main


def main(
    stride: int = STRIDE,
    batch_size: int = 32,
    num_epochs: int = 100,
    early_stopping_patience: int = 20,
    num_workers: int = 0,
):
    """
    Runs training experiments for each value in SGT_LOSS_LAMBDAS.
    """
    dataset_path = PROCESSED_DATA_DIR / "synthetic_dataset.npy"
    for sgt_lambda in SGT_LOSS_LAMBDAS:
        typer.echo(f"Starting training with sgt_lambda = {sgt_lambda}")
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
