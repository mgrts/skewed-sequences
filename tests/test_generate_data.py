"""Tests for skewed_sequences.data.synthetic.generate_data."""

import numpy as np
import pytest

from skewed_sequences.data.synthetic.generate_data import (
    SkewedGeneralizedT,
    combined_cosine_gaussian_kernel,
    gaussian_kernel,
    safe_normalize,
    smooth_sequence,
)


class TestKernels:
    def test_gaussian_sums_to_one(self):
        k = gaussian_kernel(size=99, sigma=10.0)
        assert k.sum() == pytest.approx(1.0, abs=1e-10)

    def test_combined_sums_to_one(self):
        k = combined_cosine_gaussian_kernel(size=99, sigma=10.0, period=30.0)
        assert k.sum() == pytest.approx(1.0, abs=1e-10)

    def test_safe_normalize_raises_on_zero(self):
        with pytest.raises(ValueError, match="Kernel sum is zero"):
            safe_normalize(np.zeros(5))


class TestSmoothSequence:
    def test_output_length(self):
        seq = np.random.randn(300)
        out = smooth_sequence(seq, "gaussian", kernel_size=15, sigma=3.0, period=30)
        assert len(out) == 300


class TestSkewedGeneralizedT:
    def test_pdf_nonneg(self):
        sgt = SkewedGeneralizedT(mu=0, sigma=1, lam=0, p=2, q=2)
        x = np.linspace(-5, 5, 100)
        assert (sgt.pdf(x) >= 0).all()

    def test_pdf_integrates_near_one(self):
        sgt = SkewedGeneralizedT(mu=0, sigma=1, lam=0, p=2, q=100)
        x = np.linspace(-10, 10, 10_000)
        dx = x[1] - x[0]
        integral = np.sum(sgt.pdf(x)) * dx
        assert integral == pytest.approx(1.0, abs=0.05)

    def test_rvs_shape(self):
        sgt = SkewedGeneralizedT(mu=0, sigma=1, lam=0, p=2, q=100)
        samples = sgt.rvs(size=50)
        assert samples.shape == (50,)

    def test_generate_sequences_shape(self):
        sgt = SkewedGeneralizedT(mu=0, sigma=1, lam=0, p=2, q=100)
        data = sgt.generate_sequences(n_sequences=3, sequence_length=10, n_features=1)
        assert data.shape == (3, 10, 1)

    def test_lam_out_of_range(self):
        with pytest.raises(AssertionError):
            SkewedGeneralizedT(lam=1.5)
