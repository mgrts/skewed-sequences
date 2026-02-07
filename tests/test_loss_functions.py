"""Tests for skewed_sequences.modeling.loss_functions."""

import pytest
import torch

from skewed_sequences.modeling.loss_functions import (
    CauchyLoss,
    HuberLoss,
    SGTLoss,
    TukeyLoss,
)


@pytest.fixture
def tensors():
    """Create a small deterministic (y, y_pred) pair."""
    torch.manual_seed(0)
    y = torch.randn(4, 5, 1)
    y_pred = torch.randn(4, 5, 1)
    return y, y_pred


class TestSGTLoss:
    def test_output_scalar(self, tensors):
        y, y_pred = tensors
        loss_fn = SGTLoss(p=2.0, q=2.0, lam=0.0, sigma=1.0, eps=1e-6)
        loss = loss_fn(y, y_pred)
        assert loss.dim() == 0  # scalar

    def test_zero_residual(self):
        y = torch.ones(2, 3, 1)
        loss_fn = SGTLoss()
        loss = loss_fn(y, y.clone())
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_no_nan_with_default_eps(self, tensors):
        y, y_pred = tensors
        loss = SGTLoss(eps=1e-6)(y, y_pred)
        assert torch.isfinite(loss)

    def test_skewed_variant(self, tensors):
        y, y_pred = tensors
        loss = SGTLoss(lam=0.5, q=5.0, sigma=0.7)(y, y_pred)
        assert torch.isfinite(loss) and loss.item() > 0


class TestCauchyLoss:
    def test_zero_residual(self):
        x = torch.ones(3, 4)
        loss = CauchyLoss(gamma=2.0)(x, x.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive(self, tensors):
        y, y_pred = tensors
        loss = CauchyLoss()(y.squeeze(-1), y_pred.squeeze(-1))
        assert loss.item() > 0


class TestHuberLoss:
    def test_matches_mse_small_residuals(self):
        """For tiny residuals (<<delta), Huber ≈ 0.5 * MSE."""
        y = torch.zeros(10)
        y_pred = torch.full((10,), 0.001)
        huber = HuberLoss(delta=1.0)(y, y_pred)
        mse = 0.5 * torch.mean((y - y_pred) ** 2)
        assert huber.item() == pytest.approx(mse.item(), abs=1e-8)

    def test_positive(self, tensors):
        y, y_pred = tensors
        loss = HuberLoss()(y.squeeze(-1), y_pred.squeeze(-1))
        assert loss.item() > 0


class TestTukeyLoss:
    def test_clamped_large_residuals(self):
        """Residuals > c should contribute constant c²/6."""
        y = torch.zeros(10)
        y_pred = torch.full((10,), 100.0)
        loss = TukeyLoss(c=4.685)(y, y_pred)
        expected = 4.685**2 / 6
        assert loss.item() == pytest.approx(expected, rel=1e-4)

    def test_zero_residual(self):
        x = torch.ones(5)
        loss = TukeyLoss()(x, x.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-6)
