"""Tests for skewed_sequences.modeling.data_processing."""

import numpy as np
import torch

from skewed_sequences.modeling.data_processing import SlidingWindowDataset, create_dataloaders


class TestSlidingWindowDataset:
    def test_shapes(self):
        data = np.random.randn(10, 50, 2).astype(np.float32)
        ds = SlidingWindowDataset(data, context_len=20, output_len=5)
        src, tgt = ds[0]
        assert src.shape == (20, 2)
        assert tgt.shape == (5, 2)
        assert src.dtype == torch.float32

    def test_window_count_stride1(self):
        """With stride=1: n_windows = T - context_len - output_len + 1 per sequence."""
        n_seqs, T, feats = 4, 50, 1
        context_len, output_len = 20, 5
        data = np.random.randn(n_seqs, T, feats).astype(np.float32)
        ds = SlidingWindowDataset(data, context_len=context_len, output_len=output_len, stride=1)
        expected_windows_per_seq = T - context_len - output_len + 1  # 26
        assert len(ds) == n_seqs * expected_windows_per_seq

    def test_window_count_stride_gt1(self):
        """With stride>1 the window count drops accordingly."""
        n_seqs, T, feats = 3, 50, 1
        context_len, output_len, stride = 20, 5, 5
        data = np.random.randn(n_seqs, T, feats).astype(np.float32)
        ds = SlidingWindowDataset(
            data, context_len=context_len, output_len=output_len, stride=stride
        )
        expected_windows_per_seq = (T - context_len - output_len) // stride + 1  # 6
        assert len(ds) == n_seqs * expected_windows_per_seq

    def test_contiguity(self):
        """src and tgt should be contiguous, non-overlapping slices."""
        data = np.arange(100).reshape(1, 100, 1).astype(np.float32)
        ds = SlidingWindowDataset(data, context_len=20, output_len=5)
        src, tgt = ds[0]
        assert src[-1, 0].item() + 1 == tgt[0, 0].item()

    def test_different_windows_differ(self):
        """Consecutive windows with stride=1 should be offset by 1."""
        data = np.arange(100).reshape(1, 100, 1).astype(np.float32)
        ds = SlidingWindowDataset(data, context_len=20, output_len=5, stride=1)
        src0, _ = ds[0]
        src1, _ = ds[1]
        assert src1[0, 0].item() == src0[0, 0].item() + 1


class TestCreateDataloaders:
    def test_sequence_level_split(self):
        """Train/val split happens at sequence level, not window level."""
        n_seqs, T, feats = 100, 50, 1
        context_len, output_len = 20, 5
        data = np.random.randn(n_seqs, T, feats).astype(np.float32)
        train_dl, val_dl, _ = create_dataloaders(
            data,
            context_len=context_len,
            output_len=output_len,
            batch_size=16,
            test_split=0.2,
            seed=0,
        )
        windows_per_seq = T - context_len - output_len + 1  # 26
        assert len(train_dl.dataset) == 80 * windows_per_seq
        assert len(val_dl.dataset) == 20 * windows_per_seq

    def test_batch_shapes(self):
        data = np.random.randn(50, 40, 1).astype(np.float32)
        train_dl, _, _ = create_dataloaders(
            data,
            context_len=20,
            output_len=5,
            batch_size=8,
            test_split=0.1,
            seed=0,
        )
        src, tgt = next(iter(train_dl))
        assert src.shape[1] == 20
        assert tgt.shape[1] == 5
        assert src.shape[0] <= 8
