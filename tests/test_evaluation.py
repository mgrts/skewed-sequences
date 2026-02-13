"""Tests for skewed_sequences.modeling.evaluation."""

import numpy as np
import torch
import torch.nn as nn

from skewed_sequences.modeling.evaluation import (
    aggregate_dense_predictions,
    sliding_window_predictions,
)


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


class TestAggregateDensePredictions:
    def test_coverage(self):
        """Prediction region should be non-NaN; context region should be NaN."""
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
        agg = aggregate_dense_predictions(preds, seq_len=T, n_features=1, output_len=output_len)

        # Context region (0..context_len-1) should be NaN
        assert np.all(np.isnan(agg["mean"][:context_len]))
        # Prediction region (context_len..T-1) should be non-NaN
        assert np.all(~np.isnan(agg["mean"][context_len:]))

    def test_min_max_bounds(self):
        """min <= mean <= max everywhere predictions exist."""
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
        agg = aggregate_dense_predictions(preds, seq_len=T, n_features=1, output_len=output_len)

        valid = ~np.isnan(agg["mean"])
        assert np.all(agg["min"][valid] <= agg["mean"][valid] + 1e-9)
        assert np.all(agg["mean"][valid] <= agg["max"][valid] + 1e-9)
