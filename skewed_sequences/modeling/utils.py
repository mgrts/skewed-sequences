import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple:
    model.train()
    total_loss = 0.0
    n_samples = 0
    all_preds = []
    all_targets = []

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt)
        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()
        # Sample-weight so a smaller final batch doesn't skew the epoch loss.
        total_loss += loss.item() * src.size(0)
        n_samples += src.size(0)
        all_preds.append(output.detach())
        all_targets.append(tgt.detach())

    avg_loss = total_loss / n_samples
    metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_targets))
    metrics["loss"] = avg_loss
    return avg_loss, metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    model.eval()
    total_loss = 0.0
    n_samples = 0
    all_preds = []
    all_targets = []

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        output = model(src, tgt)
        loss = criterion(output, tgt)
        # Sample-weight so a smaller final batch doesn't skew the epoch loss.
        total_loss += loss.item() * src.size(0)
        n_samples += src.size(0)
        all_preds.append(output)
        all_targets.append(tgt)

    avg_loss = total_loss / n_samples
    metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_targets))
    metrics["loss"] = avg_loss
    return avg_loss, metrics


def compute_metrics(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-6) -> dict:
    y_pred = y_pred.detach().cpu()
    y_true = y_true.detach().cpu()

    abs_diff = torch.abs(y_pred - y_true)
    denominator = torch.clamp(torch.abs(y_true), min=eps)

    mape = (abs_diff / denominator).mean().item() * 100

    smape = (
        abs_diff / torch.clamp((torch.abs(y_true) + torch.abs(y_pred)) / 2, min=eps)
    ).mean().item() * 100

    # Scale-based metrics. These are the meaningful headline numbers for
    # zero-mean / sign-changing (standardized) targets, where the percentage
    # metrics above blow up near zero crossings and are dominated by the eps
    # clamp rather than by forecast quality.
    rmse = torch.sqrt((abs_diff**2).mean()).item()
    mae = abs_diff.mean().item()

    return {
        "mape": round(mape, 4),
        "smape": round(smape, 4),
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
    }


@torch.no_grad()
def persistence_metrics(dataloader: DataLoader, device: torch.device) -> dict:
    """Metrics for the naive last-value (persistence) forecaster.

    Predicts the last observed context step for every future step — the trivial
    baseline a learned forecaster must beat. Used to compute MASE and to expose
    how much headroom the task actually has.
    """
    all_preds = []
    all_targets = []
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        out_len = tgt.size(1)
        naive = src[:, -1:, : tgt.size(-1)].repeat(1, out_len, 1)
        all_preds.append(naive)
        all_targets.append(tgt)
    return compute_metrics(torch.cat(all_preds), torch.cat(all_targets))


def residual_scale_estimate(data: np.ndarray, eps: float = 1e-6) -> float:
    """Robust scale of the 1-step increments (MAD * 1.4826).

    Used to put the robust-loss thresholds (Huber/Tukey/Cauchy) at the data's
    residual scale instead of a fixed unit scale, so they actually enter their
    robust regime rather than collapsing to MSE.
    """
    increments = np.diff(np.asarray(data)[:, :, 0], axis=1).ravel()
    mad = np.median(np.abs(increments - np.median(increments)))
    return max(float(1.4826 * mad), eps)


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum change in the monitored metric to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = float("inf")
        self.counter = 0
        self.early_stop = False

    def step(self, current_value: float) -> bool:
        """
        Call this after each validation step.

        Args:
            current_value (float): The current validation loss or monitored metric.

        Returns:
            bool: True if training should stop.
        """
        if current_value < self.best_value - self.min_delta:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
