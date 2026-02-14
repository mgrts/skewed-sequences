"""Tests for skewed_sequences.modeling.evaluation."""

import numpy as np
import torch
import torch.nn as nn

from skewed_sequences.modeling.evaluation import sliding_window_predictions


class _DummyModel(nn.Module):
    """Minimal model that returns ones for any input."""

    def __init__(self, out_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(out_dim, out_dim)

    def infer(self, src: torch.Tensor, tgt_len: int) -> torch.Tensor:
        B = src.shape[0]
        out_dim = src.shape[2]
        return torch.ones(B, tgt_len, out_dim)


class TestSlidingWindowPredictions:
    def test_window_count_stride1(self):
        T, context_len, output_len = 50, 20, 5
        seq = np.random.randn(T, 1).astype(np.float32)
        model = _DummyModel()
        preds = sliding_window_predictions(
            model,
            seq,
            context_len,
            output_len,
            stride=1,
            device=torch.device("cpu"),
        )
        expected = T - context_len - output_len + 1  # 26
        assert len(preds) == expected

    def test_window_count_stride_output_len(self):
        T, context_len, output_len = 50, 20, 5
        seq = np.random.randn(T, 1).astype(np.float32)
        model = _DummyModel()
        preds = sliding_window_predictions(
            model,
            seq,
            context_len,
            output_len,
            stride=output_len,
            device=torch.device("cpu"),
        )
        expected = (T - context_len - output_len) // output_len + 1  # 6
        assert len(preds) == expected

    def test_start_indices(self):
        T, context_len, output_len = 50, 20, 5
        seq = np.random.randn(T, 1).astype(np.float32)
        model = _DummyModel()
        preds = sliding_window_predictions(
            model,
            seq,
            context_len,
            output_len,
            stride=1,
            device=torch.device("cpu"),
        )
        starts = [s for s, _ in preds]
        assert starts[0] == context_len
        assert starts[1] == context_len + 1
