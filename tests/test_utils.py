"""Tests for skewed_sequences.modeling.utils — residual scale + metric helpers."""

import numpy as np
import pytest
import torch

from skewed_sequences.modeling.utils import compute_metrics, residual_scale_estimate


class TestResidualScaleEstimate:
    def test_recovers_increment_scale(self):
        rng = np.random.default_rng(0)
        increments = rng.normal(0.0, 0.2, size=(50, 300))
        data = np.cumsum(increments, axis=1)[..., np.newaxis]
        # 1.4826 * MAD of Gaussian increments ~ their std.
        assert residual_scale_estimate(data) == pytest.approx(0.2, rel=0.1)

    def test_raises_on_piecewise_constant_data(self):
        """Weekly-reported data through a rolling mean: >50% zero increments -> MAD 0.

        This is the OWID failure that silently produced ``residual_scale=1e-6`` and
        hundreds of degenerate runs; it must fail loudly now.
        """
        levels = np.repeat(np.arange(10.0), 30)  # 300 steps, changes every 30
        data = np.tile(levels[np.newaxis, :, np.newaxis], (5, 1, 1))
        with pytest.raises(ValueError, match="degenerate"):
            residual_scale_estimate(data)

    def test_raises_on_constant_data(self):
        with pytest.raises(ValueError, match="degenerate"):
            residual_scale_estimate(np.zeros((3, 300, 1)))

    def test_raises_on_nan(self):
        data = np.cumsum(np.random.default_rng(0).normal(size=(3, 300)), axis=1)[..., np.newaxis]
        data[0, 10, 0] = np.nan
        with pytest.raises(ValueError, match="not finite"):
            residual_scale_estimate(data)

    def test_min_scale_threshold(self):
        rng = np.random.default_rng(1)
        data = np.cumsum(rng.normal(0.0, 1e-3, size=(20, 300)), axis=1)[..., np.newaxis]
        assert residual_scale_estimate(data, min_scale=1e-5) > 0
        with pytest.raises(ValueError, match="degenerate"):
            residual_scale_estimate(data, min_scale=1e-1)


class TestComputeMetrics:
    def test_perfect_prediction(self):
        y = torch.randn(8, 1, 1) + 1.0
        m = compute_metrics(y, y)
        assert m["mae"] == 0.0 and m["rmse"] == 0.0 and m["smape"] == 0.0

    def test_known_values(self):
        y_true = torch.tensor([[[1.0]], [[2.0]]])
        y_pred = torch.tensor([[[1.5]], [[2.5]]])
        m = compute_metrics(y_pred, y_true)
        assert m["mae"] == pytest.approx(0.5)
        assert m["rmse"] == pytest.approx(0.5)
        assert m["mape"] == pytest.approx(100 * (0.5 / 1.0 + 0.5 / 2.0) / 2, rel=1e-4)
