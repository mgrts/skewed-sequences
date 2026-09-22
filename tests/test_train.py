"""Tests for skewed_sequences.modeling.train — get_loss_function factory."""

import pytest
import torch

from skewed_sequences.modeling.train import get_loss_function


@pytest.mark.parametrize(
    "loss_type", ["sgt", "mse", "mae", "cauchy", "huber", "tukey", "charbonnier"]
)
def test_all_loss_types_instantiate(loss_type):
    fn = get_loss_function(loss_type)
    assert callable(fn)


def test_unknown_loss_raises():
    with pytest.raises(ValueError, match="Unsupported loss type"):
        get_loss_function("nonexistent")


@pytest.mark.parametrize("loss_type", ["mse", "mae", "cauchy", "huber", "tukey", "charbonnier"])
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


def test_sgt_sigma_scales_with_residual_scale():
    """SGT's sigma tracks the residual scale (like Huber/Tukey/Cauchy/Charbonnier),
    so q engages the tail regime instead of SGT collapsing to MSE at this data
    scale. sgt_loss_sigma is a unit multiplier on residual_scale."""
    fn = get_loss_function("sgt", sgt_loss_sigma=1.0, sgt_loss_q=2.5, residual_scale=0.2)
    assert fn.sigma == pytest.approx(0.2)
    fn2 = get_loss_function("sgt", sgt_loss_sigma=2.0, sgt_loss_q=2.5, residual_scale=0.2)
    assert fn2.sigma == pytest.approx(0.4)
    # Default residual_scale=1.0 keeps standalone/test behaviour unchanged.
    fn3 = get_loss_function("sgt", sgt_loss_sigma=1.0, sgt_loss_q=2.5)
    assert fn3.sigma == pytest.approx(1.0)


def test_charbonnier_scales_with_residual_scale():
    fn = get_loss_function("charbonnier", residual_scale=2.0)
    assert fn.eps == pytest.approx(1.345 * 2.0)


def test_select_device_env_override(monkeypatch):
    from skewed_sequences.modeling.train import select_device

    monkeypatch.setenv("SKSEQ_DEVICE", "cpu")
    assert select_device() == "cpu"
    monkeypatch.delenv("SKSEQ_DEVICE")
    assert select_device() in {"cpu", "cuda", "mps"}


def test_compile_mode_from_env(monkeypatch):
    from skewed_sequences.modeling.train import compile_mode_from_env

    monkeypatch.delenv("SKSEQ_COMPILE", raising=False)
    assert compile_mode_from_env() == "none"
    for raw, expected in [
        ("1", "default"),
        ("default", "default"),
        ("reduce-overhead", "reduce-overhead"),
        ("0", "none"),
    ]:
        monkeypatch.setenv("SKSEQ_COMPILE", raw)
        assert compile_mode_from_env() == expected


def test_maybe_compile_forward_keeps_state_dict_and_infer(monkeypatch):
    """Compiling the bound forward must not rename checkpoint keys or touch ``infer``."""
    from unittest.mock import patch

    from skewed_sequences.modeling.models import TransformerWithPE
    from skewed_sequences.modeling.train import maybe_compile_forward

    model = TransformerWithPE(in_dim=1, out_dim=1, embed_dim=8, num_heads=2, num_layers=1)
    keys_before = list(model.state_dict())
    infer_before = model.infer
    sentinel = object()
    with patch(
        "skewed_sequences.modeling.train.torch.compile", return_value=sentinel
    ) as mock_compile:
        assert maybe_compile_forward(model, "none") is model
        mock_compile.assert_not_called()
        maybe_compile_forward(model, "reduce-overhead")
        mock_compile.assert_called_once()
        assert mock_compile.call_args.kwargs == {"mode": "reduce-overhead"}
    assert model.forward is sentinel  # instance attribute overrides the class method
    assert list(model.state_dict()) == keys_before
    assert model.infer == infer_before
