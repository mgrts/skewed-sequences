from pathlib import Path

from loguru import logger
import mlflow
import numpy as np
import torch
import typer

from skewed_sequences.config import (
    CONTEXT_LENGTH,
    MODELS_DIR,
    OUTPUT_LENGTH,
    PROCESSED_DATA_DIR,
    SEED,
    STRIDE,
    TRACKING_URI,
)
from skewed_sequences.modeling.data_processing import create_dataloaders
from skewed_sequences.modeling.evaluation import log_val_predictions
from skewed_sequences.modeling.loss_functions import CauchyLoss, HuberLoss, SGTLoss, TukeyLoss
from skewed_sequences.modeling.models import LSTM, TransformerWithPE
from skewed_sequences.modeling.trainer import train_model
from skewed_sequences.modeling.utils import set_seed

app = typer.Typer(pretty_exceptions_show_locals=False)


def get_loss_function(
    loss_type: str,
    sgt_loss_lambda: float = 0.0,
    sgt_loss_q: float = 2.0,
    sgt_loss_sigma: float = 1.0,
    sgt_loss_p: float = 2.0,
):
    loss_type = loss_type.lower()
    if loss_type == "sgt":
        return SGTLoss(
            eps=1e-6, sigma=sgt_loss_sigma, p=sgt_loss_p, q=sgt_loss_q, lam=sgt_loss_lambda
        )
    elif loss_type == "mse":
        return torch.nn.MSELoss()
    elif loss_type == "mae":
        return torch.nn.L1Loss()
    elif loss_type == "cauchy":
        return CauchyLoss(gamma=2.0)
    elif loss_type == "huber":
        return HuberLoss(delta=1.0)
    elif loss_type == "tukey":
        return TukeyLoss(c=4.685)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


@app.command()
def main(
    dataset_path: Path = PROCESSED_DATA_DIR / "synthetic_dataset.npy",
    model_type: str = "transformer",
    context_length: int = CONTEXT_LENGTH,
    output_length: int = OUTPUT_LENGTH,
    stride: int = STRIDE,
    embed_dim: int = 64,
    num_heads: int = 4,
    num_layers: int = 4,
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    test_split: float = 0.1,
    seed: int = SEED,
    early_stopping_patience: int = 20,
    loss_type: str = "sgt",
    sgt_loss_sigma: float = 1.0,
    sgt_loss_lambda: float = 0.0,
    sgt_loss_q: float = 2.0,
    sgt_loss_p: float = 2.0,
    experiment_name: str = "Transformer-SGT-synthetic",
):
    set_seed(seed)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    logger.info(f"Using device: {device}")
    logger.info("Loading data...")

    data = np.load(dataset_path)

    assert context_length + output_length <= data.shape[1], (
        f"context_length ({context_length}) + output_length ({output_length}) "
        f"exceeds sequence length ({data.shape[1]})"
    )

    train_loader, val_loader = create_dataloaders(
        data=data,
        context_len=context_length,
        output_len=output_length,
        batch_size=batch_size,
        test_split=test_split,
        stride=stride,
        seed=seed,
    )

    if model_type == "transformer":
        model = TransformerWithPE(
            in_dim=data.shape[-1],
            out_dim=data.shape[-1],
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        ).to(device)
    elif model_type == "lstm":
        model = LSTM(
            input_dim=data.shape[-1],
            hidden_dim=embed_dim,
            num_layers=num_layers,
            output_dim=data.shape[-1],
        ).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    criterion = get_loss_function(
        loss_type,
        sgt_loss_lambda=sgt_loss_lambda,
        sgt_loss_q=sgt_loss_q,
        sgt_loss_sigma=sgt_loss_sigma,
        sgt_loss_p=sgt_loss_p,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        output_dir = MODELS_DIR / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        model_save_path = output_dir / "model.pt"

        mlflow.log_params(
            {
                "model_type": model_type,
                "sgt_loss_lambda": sgt_loss_lambda,
                "sgt_loss_q": sgt_loss_q,
                "sgt_loss_sigma": sgt_loss_sigma,
                "sgt_loss_p": sgt_loss_p,
                "context_length": context_length,
                "output_length": output_length,
                "stride": stride,
                "embed_dim": embed_dim,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "loss_type": loss_type,
                "early_stopping_patience": early_stopping_patience,
                "num_epochs": num_epochs,
                "test_split": test_split,
                "seed": seed,
            }
        )

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

        mlflow.log_metrics(
            {
                "best_train_smape": best_train_metrics.get("smape"),
                "best_val_smape": best_val_metrics.get("smape"),
            }
        )

        log_val_predictions(model, val_loader, model_path=model_save_path)

        logger.success("Training complete.")


if __name__ == "__main__":
    app()
