"""Tests for skewed_sequences.modeling.models — Transformer and LSTM."""

import pytest
import torch

from skewed_sequences.modeling.models import LSTM, TransformerWithPE


@pytest.fixture(params=["transformer", "lstm"])
def model_and_name(request):
    """Parametrized fixture returning (model, name) for both architectures."""
    in_dim, out_dim, embed_dim, num_heads, num_layers = 1, 1, 16, 2, 1
    if request.param == "transformer":
        m = TransformerWithPE(
            in_dim=in_dim,
            out_dim=out_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )
    else:
        m = LSTM(
            input_dim=in_dim,
            hidden_dim=embed_dim,
            num_layers=num_layers,
            output_dim=out_dim,
        )
    m.eval()
    return m, request.param


class TestForward:
    def test_output_shape(self, model_and_name):
        model, _ = model_and_name
        B, src_len, tgt_len, dim = 2, 20, 5, 1
        src = torch.randn(B, src_len, dim)
        tgt = torch.randn(B, tgt_len, dim)
        out = model(src, tgt)
        assert out.shape == (B, tgt_len, dim)

    def test_no_nan(self, model_and_name):
        model, _ = model_and_name
        src = torch.randn(2, 20, 1)
        tgt = torch.randn(2, 5, 1)
        out = model(src, tgt)
        assert torch.isfinite(out).all()


class TestInfer:
    def test_output_shape(self, model_and_name):
        model, _ = model_and_name
        B, src_len, tgt_len = 2, 20, 5
        src = torch.randn(B, src_len, 1)
        out = model.infer(src, tgt_len=tgt_len)
        assert out.shape == (B, tgt_len, 1)

    def test_single_step(self, model_and_name):
        model, _ = model_and_name
        src = torch.randn(1, 30, 1)
        out = model.infer(src, tgt_len=1)
        assert out.shape == (1, 1, 1)
        assert torch.isfinite(out).all()


class TestNoTargetLeak:
    """The shifted teacher-forcing protocol must never feed the value being
    scored into the decoder, so the prediction cannot depend on it."""

    def test_single_step_prediction_independent_of_target(self, model_and_name):
        model, _ = model_and_name
        torch.manual_seed(0)
        src = torch.randn(2, 20, 1)
        tgt = torch.randn(2, 1, 1)
        out_a = model(src, tgt)
        out_b = model(src, tgt + 100.0)  # perturb the only target
        # OUTPUT_LENGTH=1: the decoder input is the seed alone, so the output is
        # invariant to the target. (This assertion fails on the leaky version.)
        assert torch.allclose(out_a, out_b, atol=1e-5)

    def test_last_target_never_fed_to_decoder(self, model_and_name):
        model, _ = model_and_name
        torch.manual_seed(0)
        src = torch.randn(2, 20, 1)
        tgt = torch.randn(2, 5, 1)
        out_a = model(src, tgt)
        tgt2 = tgt.clone()
        tgt2[:, -1] += 100.0  # the last target is never used as a decoder input
        out_b = model(src, tgt2)
        assert torch.allclose(out_a, out_b, atol=1e-5)

    def test_forward_matches_infer_single_step(self, model_and_name):
        """For OUTPUT_LENGTH=1, teacher-forced forward and autoregressive infer
        seed from the same value, so they must produce the identical prediction."""
        model, _ = model_and_name
        torch.manual_seed(0)
        src = torch.randn(2, 20, 1)
        tgt = torch.randn(2, 1, 1)
        out_forward = model(src, tgt)
        out_infer = model.infer(src, tgt_len=1)
        assert torch.allclose(out_forward, out_infer, atol=1e-5)
