from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset


class SlidingWindowDataset(Dataset):
    """Lazily indexes sliding windows over pre-split sequence data.

    Each item is a (input, target) pair where input has shape
    (context_len, features) and target has shape (output_len, features).
    """

    def __init__(
        self,
        data: np.ndarray,
        context_len: int,
        output_len: int,
        stride: int = 1,
    ):
        super().__init__()
        self.data = data
        self.context_len = context_len
        self.output_len = output_len
        window_total = context_len + output_len
        n_seqs, T, _ = data.shape

        # Build a flat index of (seq_idx, window_start) tuples
        self.index: List[Tuple[int, int]] = []
        for seq_idx in range(n_seqs):
            n_windows = (T - window_total) // stride + 1
            for w in range(n_windows):
                self.index.append((seq_idx, w * stride))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        seq_idx, start = self.index[idx]
        end_input = start + self.context_len
        end_target = end_input + self.output_len
        return (
            torch.tensor(self.data[seq_idx, start:end_input, :], dtype=torch.float32),
            torch.tensor(self.data[seq_idx, end_input:end_target, :], dtype=torch.float32),
        )


def create_dataloaders(
    data: np.ndarray,
    context_len: int,
    output_len: int,
    batch_size: int,
    test_split: float,
    stride: int = 1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, np.ndarray]:
    # Split at the sequence level to prevent leakage
    train_data, val_data = train_test_split(data, test_size=test_split, random_state=seed)

    train_loader = DataLoader(
        SlidingWindowDataset(train_data, context_len, output_len, stride),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        SlidingWindowDataset(val_data, context_len, output_len, stride),
        batch_size=batch_size,
    )

    return train_loader, val_loader, val_data
