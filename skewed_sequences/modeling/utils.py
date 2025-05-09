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
) -> float:
    model.train()
    losses = []

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt)
        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    losses = []

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        output = model(src, tgt)
        loss = criterion(output, tgt)
        losses.append(loss.item())

    return float(np.mean(losses))


def compute_metrics(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-6) -> dict:
    y_pred = y_pred.detach().cpu()
    y_true = y_true.detach().cpu()

    abs_diff = torch.abs(y_pred - y_true)
    denominator = torch.clamp(torch.abs(y_true), min=eps)

    mape = (abs_diff / denominator).mean().item() * 100

    smape = (abs_diff / torch.clamp((torch.abs(y_true) + torch.abs(y_pred)) / 2, min=eps)).mean().item() * 100

    return {
        'mape': round(mape, 4),
        'smape': round(smape, 4),
    }


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
        self.best_value = float('inf')
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
