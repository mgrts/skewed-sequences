from pathlib import Path

from loguru import logger
import mlflow
import torch

from skewed_sequences.mlflow_contract import SUMMARY_METRIC_STEMS
from skewed_sequences.modeling.utils import EarlyStopping, evaluate, train_epoch


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    model_save_path,
    num_epochs,
    early_stopping_patience,
):
    device = next(model.parameters()).device
    early_stopper = EarlyStopping(patience=early_stopping_patience)

    # Select the best epoch / early-stop on a COMMON, loss-agnostic validation
    # metric (val MAE) rather than each loss's own val_loss, so the reported
    # cross-loss comparison is apples-to-apples (every loss is selected by the
    # same rule it is later compared on).
    best_val_mae = float("inf")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        val_mae = val_metrics["mae"]

        # Log to MLflow
        epoch_metrics = {"train_loss": train_loss, "val_loss": val_loss}
        for stem in SUMMARY_METRIC_STEMS:
            epoch_metrics[f"train_{stem}"] = train_metrics[stem]
            epoch_metrics[f"val_{stem}"] = val_metrics[stem]
        mlflow.log_metrics(epoch_metrics, step=epoch)

        logger.info(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val MAE: {val_mae:.4f} | Val sMAPE: {val_metrics['smape']:.2f}"
        )

        # Save on improvement of val MAE, or whenever no checkpoint exists yet so
        # that a non-finite val metric on every epoch (exploding predictions on
        # heavy tails) still leaves a loadable checkpoint instead of crashing.
        if val_mae < best_val_mae - 1e-6 or not Path(model_save_path).exists():
            if val_mae < best_val_mae - 1e-6:
                best_val_mae = val_mae

            torch.save(model.state_dict(), model_save_path)
            mlflow.log_artifact(model_save_path)
            logger.info(f"Saved best model to {model_save_path} (val_mae={val_mae:.4f})")

        if early_stopper.step(val_mae):
            logger.warning(f"Early stopping triggered after {epoch} epochs.")
            break

    if not Path(model_save_path).exists():
        raise RuntimeError(
            "No checkpoint was ever saved — val MAE was non-finite on every epoch "
            f"(best_val_mae={best_val_mae}). Check for exploding predictions/gradients "
            "on this dataset/loss configuration."
        )

    # Reload best checkpoint and re-evaluate all splits in eval mode
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    _, best_train_metrics = evaluate(model, train_loader, criterion, device)
    _, best_val_metrics = evaluate(model, val_loader, criterion, device)
    _, best_test_metrics = evaluate(model, test_loader, criterion, device)

    return best_val_mae, best_train_metrics, best_val_metrics, best_test_metrics
