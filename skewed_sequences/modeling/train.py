from pathlib import Path
import numpy as np
import torch
import mlflow
import typer
from loguru import logger

from skewed_sequences.config import PROCESSED_DATA_DIR, MODELS_DIR, TRACKING_URI, SEQUENCE_LENGTH
from skewed_sequences.modeling.loss_functions import SGTLoss
from skewed_sequences.modeling.models import TransformerWithPE
from skewed_sequences.modeling.data_processing import create_dataloaders
from skewed_sequences.modeling.utils import set_seed
from skewed_sequences.modeling.trainer import train_model
from skewed_sequences.modeling.evaluation import log_val_predictions

app = typer.Typer(pretty_exceptions_show_locals=False)


def get_loss_function(loss_type: str):
    loss_type = loss_type.lower()
    if loss_type == 'sgt':
        return SGTLoss(p=2.0, q=2.0, lambda_=0.0, sigma=1.0)
    elif loss_type == 'mse':
        return torch.nn.MSELoss()
    elif loss_type == 'mae':
        return torch.nn.L1Loss()
    else:
        raise ValueError(f'Unsupported loss type: {loss_type}')


@app.command()
def main(
    dataset_path: Path = PROCESSED_DATA_DIR / 'synthetic_dataset.npy',
    sequence_length: int = SEQUENCE_LENGTH,
    output_length: int = 60,
    embed_dim: int = 64,
    num_heads: int = 4,
    num_layers: int = 4,
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    test_split: float = 0.1,
    seed: int = 933,
    early_stopping_patience: int = 5,
    loss_type: str = 'sgt',
    experiment_name: str = 'Transformer-SGT-synthetic',
):
    set_seed(seed)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    )

    assert output_length < sequence_length, 'Output length must be less than sequence length'

    logger.info(f'Using device: {device}')
    logger.info('Loading data...')

    data = np.load(dataset_path)
    input_length = sequence_length - output_length

    train_loader, val_loader = create_dataloaders(
        data=data,
        input_len=input_length,
        output_len=output_length,
        batch_size=batch_size,
        test_split=test_split,
        seed=seed,
    )

    model = TransformerWithPE(
        in_dim=data.shape[-1],
        out_dim=data.shape[-1],
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
    ).to(device)

    criterion = get_loss_function(loss_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        output_dir = MODELS_DIR / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        model_save_path = output_dir / 'model.pt'

        mlflow.log_params({
            'input_length': input_length,
            'output_length': output_length,
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'loss_type': loss_type,
            'early_stopping_patience': early_stopping_patience,
        })

        best_val_loss, best_train_metrics, best_val_metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            model_save_path=model_save_path,
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
        )

        mlflow.log_metrics({
            'best_train_smape': best_val_metrics.get('smape'),
            'best_val_smape': best_train_metrics.get('smape'),
        })

        log_val_predictions(model, val_loader, model_path=model_save_path)

        logger.success('Training complete.')


if __name__ == '__main__':
    app()
