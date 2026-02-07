"""Tests for skewed_sequences.modeling.train — get_loss_function factory."""

import pytest
import torch

from skewed_sequences.modeling.train import get_loss_function


@pytest.mark.parametrize("loss_type", ["sgt", "mse", "mae", "cauchy", "huber", "tukey"])
def test_all_loss_types_instantiate(loss_type):
    fn = get_loss_function(loss_type)
    assert callable(fn)


def test_unknown_loss_raises():
    with pytest.raises(ValueError, match="Unsupported loss type"):
        get_loss_function("nonexistent")


@pytest.mark.parametrize("loss_type", ["mse", "mae", "cauchy", "huber", "tukey"])
def test_all_losses_forward(loss_type):
    fn = get_loss_function(loss_type)
    y = torch.randn(4, 5)
    y_pred = torch.randn(4, 5)
    loss = fn(y, y_pred)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_sgt_loss_forward():
    fn = get_loss_function("sgt", sgt_loss_lambda=0.0, sgt_loss_q=2.0, sgt_loss_sigma=1.0)
    y = torch.randn(4, 5, 1)
    y_pred = torch.randn(4, 5, 1)
    loss = fn(y, y_pred)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
