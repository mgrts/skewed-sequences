"""Tests for skewed_sequences.metrics — statistical helpers."""

import numpy as np
import pytest

from skewed_sequences.metrics import (
    compute_dispersion_scaling_series,
    hill_estimator,
    kappa,
    mean_abs_deviation,
    skewness,
)


class TestMeanAbsDeviation:
    def test_constant_array(self):
        assert mean_abs_deviation(np.array([5.0, 5.0, 5.0])) == 0.0

    def test_symmetric(self):
        x = np.array([-1.0, 0.0, 1.0])
        assert mean_abs_deviation(x) == pytest.approx(2.0 / 3.0)

    def test_single_element(self):
        assert mean_abs_deviation(np.array([42.0])) == 0.0


class TestKappa:
    def test_raises_on_n_leq_1(self):
        with pytest.raises(ValueError, match="n must be greater than 1"):
            kappa(np.random.randn(500), n=1)

    def test_gaussian_kappa_near_zero(self):
        np.random.seed(42)
        x = np.random.randn(10_000)
        k = kappa(x, n=10)
        # For Gaussian data (finite variance, alpha=2): M_n ~ n^{1/2} * M_1
        # so kappa = 2 - log(n)/log(n^{1/2}) ≈ 0
        assert abs(k) < 0.5


class TestSkewness:
    def test_symmetric_near_zero(self):
        np.random.seed(0)
        x = np.random.randn(10_000)
        assert abs(skewness(x)) < 0.1

    def test_positive_skew(self):
        np.random.seed(0)
        x = np.random.exponential(1.0, size=10_000)
        assert skewness(x) > 0.5

    def test_raises_on_constant(self):
        with pytest.raises(ValueError, match="Standard deviation is zero"):
            skewness(np.array([1.0, 1.0, 1.0]))


class TestDispersionScalingSeries:
    def test_output_shape(self):
        np.random.seed(42)
        x = np.random.randn(5000)
        result = compute_dispersion_scaling_series(x, num_values=5)
        assert result.shape == (5, 5)


class TestHillEstimator:
    def test_recovers_pareto_tail_index(self):
        """np.random.pareto(a) has tail index a; Hill should recover it."""
        np.random.seed(0)
        x = np.random.pareto(3.0, size=100_000) + 1.0
        alpha = hill_estimator(x)
        assert alpha == pytest.approx(3.0, abs=0.5)

    def test_gaussian_has_large_alpha(self):
        """Light (Gaussian) tails -> large estimated tail index."""
        np.random.seed(1)
        x = np.random.randn(50_000)
        assert hill_estimator(x) > 3.0

    def test_heavier_tail_gives_smaller_alpha(self):
        np.random.seed(2)
        light = np.random.pareto(5.0, size=50_000) + 1.0
        heavy = np.random.pareto(1.5, size=50_000) + 1.0
        assert hill_estimator(heavy) < hill_estimator(light)

    def test_explicit_k(self):
        np.random.seed(3)
        x = np.random.pareto(2.0, size=20_000) + 1.0
        alpha = hill_estimator(x, k=2000)
        assert alpha == pytest.approx(2.0, abs=0.5)

    def test_raises_on_too_few_values(self):
        with pytest.raises(ValueError, match="at least 3"):
            hill_estimator(np.array([1.0, 2.0]))

    def test_raises_on_bad_k(self):
        with pytest.raises(ValueError, match="k must satisfy"):
            hill_estimator(np.array([1.0, 2.0, 3.0, 4.0]), k=10)
