import typer

from skewed_sequences.config import PROCESSED_DATA_DIR, SGT_LOSS_LAMBDAS
from skewed_sequences.modeling.train import main as train_main


def main():
    """
    Runs training experiments for each value in SGT_LOSS_LAMBDAS.
    """
    dataset_path = PROCESSED_DATA_DIR / 'synthetic_dataset.npy'
    for sgt_lambda in SGT_LOSS_LAMBDAS:
        typer.echo(f'Starting training with sgt_lambda = {sgt_lambda}')
        # Call the main training function with the current sgt_lambda
        train_main(dataset_path=dataset_path, loss_type='sgt', sgt_lambda=sgt_lambda)
        typer.echo(f'Completed training with sgt_lambda = {sgt_lambda}\n')


if __name__ == '__main__':
    main()
