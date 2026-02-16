from loguru import logger
import mlflow
import torch

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

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        # Log to MLflow
        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_mape": train_metrics["mape"],
                "train_smape": train_metrics["smape"],
                "val_mape": val_metrics["mape"],
                "val_smape": val_metrics["smape"],
            },
            step=epoch,
        )

        logger.info(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f'Train sMAPE: {train_metrics["smape"]:.2f} | Val sMAPE: {val_metrics["smape"]:.2f}'
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss

            torch.save(model.state_dict(), model_save_path)
            mlflow.log_artifact(model_save_path)
            logger.info(f"Saved best model to {model_save_path} (val_loss={val_loss:.4f})")

        if early_stopper.step(val_loss):
            logger.warning(f"Early stopping triggered after {epoch} epochs.")
            break

    # Reload best checkpoint and re-evaluate all splits in eval mode
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    _, best_train_metrics = evaluate(model, train_loader, criterion, device)
    _, best_val_metrics = evaluate(model, val_loader, criterion, device)
    _, best_test_metrics = evaluate(model, test_loader, criterion, device)

    return best_val_loss, best_train_metrics, best_val_metrics, best_test_metrics
