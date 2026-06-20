from pathlib import Path
import tempfile

from loguru import logger
import mlflow
import numpy as np
import torch
import typer

from skewed_sequences.config import (
    BATCH_SIZE,
    CONTEXT_LENGTH,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_WORKERS,
    OUTPUT_LENGTH,
    PROCESSED_DATA_DIR,
    SEED,
    STRIDE,
    TRACKING_URI,
)
from skewed_sequences.mlflow_contract import SUMMARY_METRIC_STEMS, summary_metric_key
from skewed_sequences.modeling.data_processing import create_dataloaders
from skewed_sequences.modeling.evaluation import log_val_predictions
from skewed_sequences.modeling.loss_functions import (
    CauchyLoss,
    CharbonnierLoss,
    HuberLoss,
    SGTLoss,
    TukeyLoss,
)
from skewed_sequences.modeling.models import LSTM, TransformerWithPE
from skewed_sequences.modeling.trainer import train_model
from skewed_sequences.modeling.utils import persistence_metrics, residual_scale_estimate, set_seed

app = typer.Typer(pretty_exceptions_show_locals=False)


def get_loss_function(
    loss_type: str,
    sgt_loss_lambda: float = 0.0,
    sgt_loss_q: float = 2.0,
    sgt_loss_sigma: float = 1.0,
    sgt_loss_p: float = 2.0,
    residual_scale: float = 1.0,
):
    """Build the criterion.

    Every robust loss (SGT and the Cauchy/Huber/Tukey/Charbonnier baselines)
    scales its transition by ``residual_scale`` (a robust estimate of the residual
    spread), so all of them enter their robust regime instead of collapsing to
    MSE at this data scale. For SGT this is essential: with ``sigma`` left at the
    raw unit (>> the standardized residual scale), the SGT operates entirely in
    its small-deviation (~quadratic / MSE) regime and ``q`` (its tail-tolerance
    knob) is inert. Setting ``sigma = sgt_loss_sigma * residual_scale`` puts the
    SGT transition at the residual bulk so ``q`` actually controls tail behaviour;
    ``sgt_loss_sigma`` therefore acts as a unit multiplier (default 1.0).
    """
    loss_type = loss_type.lower()
    if loss_type == "sgt":
        return SGTLoss(
            eps=1e-6,
            sigma=sgt_loss_sigma * residual_scale,
            p=sgt_loss_p,
            q=sgt_loss_q,
            lam=sgt_loss_lambda,
        )
    elif loss_type == "mse":
        return torch.nn.MSELoss()
    elif loss_type == "mae":
        return torch.nn.L1Loss()
    elif loss_type == "cauchy":
        return CauchyLoss(gamma=(2.3849 * residual_scale) ** 2)
    elif loss_type == "huber":
        return HuberLoss(delta=1.345 * residual_scale)
    elif loss_type == "tukey":
        return TukeyLoss(c=4.685 * residual_scale)
    elif loss_type == "charbonnier":
        # eps plays the role of Huber's delta — same 95%-efficiency tuning so the
        # smooth-L1 transition brackets the residual bulk at this data scale.
        return CharbonnierLoss(eps=1.345 * residual_scale)
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
    batch_size: int = BATCH_SIZE,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    test_split: float = 0.1,
    val_split: float = 0.1,
    seed: int = SEED,
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    num_workers: int = NUM_WORKERS,
    loss_type: str = "sgt",
    sgt_loss_sigma: float = 1.0,
    sgt_loss_lambda: float = 0.0,
    sgt_loss_q: float = 2.0,
    sgt_loss_p: float = 2.0,
    exp_transform: bool = False,
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

    # Robust scale of the 1-step increments — sets the robust-loss thresholds so
    # they bracket the residual bulk/tail instead of behaving like MSE.
    residual_scale = residual_scale_estimate(data)
    logger.info(f"Residual-scale estimate (MAD-based): {residual_scale:.4f}")

    train_loader, val_loader, test_loader, val_data_raw = create_dataloaders(
        data=data,
        context_len=context_length,
        output_len=output_length,
        batch_size=batch_size,
        test_split=test_split,
        val_split=val_split,
        stride=stride,
        seed=seed,
        num_workers=num_workers,
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
        residual_scale=residual_scale,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(), tempfile.TemporaryDirectory() as tmp_dir:
        model_save_path = Path(tmp_dir) / "model.pt"

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
                "val_split": val_split,
                "exp_transform": exp_transform,
                "random_state": seed,
                "residual_scale": residual_scale,
            }
        )

        best_val_mae, best_train_metrics, best_val_metrics, best_test_metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            model_save_path=model_save_path,
            num_epochs=num_epochs,
            early_stopping_patience=early_stopping_patience,
        )

        best_metrics_by_split = {
            "train": best_train_metrics,
            "val": best_val_metrics,
            "test": best_test_metrics,
        }
        summary_metrics = {
            summary_metric_key(split, stem): metrics.get(stem)
            for split, metrics in best_metrics_by_split.items()
            for stem in SUMMARY_METRIC_STEMS
        }

        # Naive last-value (persistence) baseline + MASE on the test split, so the
        # headroom of the model over the trivial forecaster is visible.
        naive = persistence_metrics(test_loader, device)
        naive_mae = naive["mae"]
        baseline_metrics = {
            "best_test_naive_rmse": naive["rmse"],
            "best_test_naive_mae": naive_mae,
            "best_test_mase": (
                (best_test_metrics["mae"] / naive_mae) if naive_mae else float("nan")
            ),
        }
        mlflow.log_metrics({**summary_metrics, **baseline_metrics})

        log_val_predictions(
            model,
            val_data_raw=val_data_raw,
            model_path=model_save_path,
            context_len=context_length,
            output_len=output_length,
        )

        logger.success("Training complete.")


if __name__ == "__main__":
    app()
