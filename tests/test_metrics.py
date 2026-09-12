"""Tests for skewed_sequences.metrics — statistical helpers."""

import numpy as np
import pytest

from skewed_sequences.metrics import (
    compute_dispersion_scaling_series,
    excess_kurtosis,
    fit_sgt,
    hill_estimator,
    kappa,
    mean_abs_deviation,
    sgt_increment_fit,
    sgt_min_q,
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


class TestSGTFit:
    @staticmethod
    def _sample(lam, q, n=20000, seed=1):
        from skewed_sequences.data.synthetic.generate_data import SkewedGeneralizedT

        np.random.seed(seed)
        return SkewedGeneralizedT(mu=0.0, sigma=1.0, lam=lam, p=2.0, q=q).rvs(size=n)

    def test_min_q_respects_validity_domain(self):
        for p in (1.0, 1.5, 2.0):
            assert sgt_min_q(p) ** p > 2.0 / p

    def test_recovers_positive_skew(self):
        fit = fit_sgt(self._sample(lam=0.5, q=5.0), p=2.0, sigma=1.0)
        assert 0.35 < fit["lam"] < 0.65
        assert 3.0 < fit["q"] < 9.0
        assert abs(fit["mu"]) < 0.1 and fit["n"] == 20000

    def test_symmetric_sample_gives_small_lambda(self):
        fit = fit_sgt(self._sample(lam=0.0, q=5.0), p=2.0, sigma=1.0)
        assert abs(fit["lam"]) < 0.1
        assert fit["lam_at_bound"] is False and fit["q_at_bound"] is False

    def test_fix_lambda(self):
        fit = fit_sgt(self._sample(lam=0.5, q=5.0), p=2.0, sigma=1.0, fix_lambda=0.0)
        assert fit["lam"] == 0.0

    def test_subsampling_is_deterministic(self):
        x = self._sample(lam=0.3, q=4.0, n=5000)
        a = fit_sgt(x, sigma=1.0, max_samples=2000)
        b = fit_sgt(x, sigma=1.0, max_samples=2000)
        assert a == b and a["n"] == 2000

    def test_increment_fit_keys_and_skew_direction(self):
        rng = np.random.default_rng(0)
        inc = rng.exponential(1.0, size=(40, 300)) - 1.0  # right-skewed increments
        data = np.cumsum(inc, axis=1)[..., np.newaxis]
        out = sgt_increment_fit(data)
        expected = {
            "p",
            "sigma",
            "n_increments",
            "increment_skewness",
            "increment_excess_kurtosis",
            "lam",
            "q",
            "mu",
            "nll",
            "q_symmetric",
            "nll_symmetric",
            "delta_nll",
        }
        assert expected <= set(out)
        assert out["n_increments"] == 40 * 299
        assert out["increment_skewness"] > 1.0 and out["lam"] > 0.1
        assert out["delta_nll"] < 0  # the skewed fit beats the symmetric one
        assert out["lam_at_bound"] is True  # exponential increments exceed the SGT skew range
        assert isinstance(out["q_at_bound"], bool)

    def test_delta_nll_never_positive(self):
        """The symmetric fit seeds the skewed fit, so delta_nll <= 0 up to tolerance."""
        rng = np.random.default_rng(3)
        data = np.cumsum(rng.normal(0, 0.2, size=(20, 300)), axis=1)[..., np.newaxis]
        out = sgt_increment_fit(data)
        assert out["delta_nll"] <= 1e-9
        assert out["lam_at_bound"] is False

    def test_increment_fit_rejects_step_data(self):
        data = np.tile(np.repeat(np.arange(10.0), 30)[np.newaxis, :, np.newaxis], (3, 1, 1))
        with pytest.raises(ValueError, match="zero MAD"):
            sgt_increment_fit(data)

    def test_excess_kurtosis(self):
        np.random.seed(0)
        assert abs(excess_kurtosis(np.random.randn(100_000))) < 0.1
        with pytest.raises(ValueError):
            excess_kurtosis(np.ones(5))
