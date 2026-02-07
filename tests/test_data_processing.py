"""Tests for skewed_sequences.modeling.data_processing."""

import numpy as np
import torch

from skewed_sequences.modeling.data_processing import SequenceDataset, create_dataloaders


class TestSequenceDataset:
    def test_shapes(self):
        data = np.random.randn(20, 30, 1).astype(np.float32)
        ds = SequenceDataset(data, input_len=25, output_len=5)
        assert len(ds) == 20
        src, tgt = ds[0]
        assert src.shape == (25, 1)
        assert tgt.shape == (5, 1)
        assert src.dtype == torch.float32

    def test_no_overlap(self):
        """src and tgt should be contiguous, non-overlapping slices."""
        data = np.arange(60).reshape(2, 30, 1).astype(np.float32)
        ds = SequenceDataset(data, input_len=20, output_len=10)
        src, tgt = ds[0]
        # last element of src should be one before first element of tgt
        assert src[-1, 0].item() + 1 == tgt[0, 0].item()


class TestCreateDataloaders:
    def test_returns_two_loaders(self):
        data = np.random.randn(100, 30, 1).astype(np.float32)
        train_dl, val_dl = create_dataloaders(
            data,
            input_len=25,
            output_len=5,
            batch_size=16,
            test_split=0.2,
            seed=0,
        )
        assert len(train_dl.dataset) == 80
        assert len(val_dl.dataset) == 20

    def test_batch_shapes(self):
        data = np.random.randn(50, 30, 1).astype(np.float32)
        train_dl, _ = create_dataloaders(
            data,
            input_len=25,
            output_len=5,
            batch_size=8,
            test_split=0.1,
            seed=0,
        )
        src, tgt = next(iter(train_dl))
        assert src.shape[1] == 25
        assert tgt.shape[1] == 5
        assert src.shape[0] <= 8
