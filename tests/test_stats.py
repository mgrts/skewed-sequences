"""Tests for skewed_sequences.stats (bootstrap intervals, sign-flip test, Holm)."""

import numpy as np
import pytest

from skewed_sequences import stats


class TestBootstrapCI:
    def test_contains_mean_and_is_reproducible(self):
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        low, high = stats.bootstrap_ci(x)
        assert low < x.mean() < high
        assert (low, high) == stats.bootstrap_ci(x)  # fixed seed
        assert (low, high) != stats.bootstrap_ci(x, seed=1)

    def test_symmetric_sample_gives_roughly_symmetric_interval(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        low, high = stats.bootstrap_ci(x)
        assert low == pytest.approx(1.5, abs=0.3) and high == pytest.approx(3.5, abs=0.3)

    def test_degenerate_inputs(self):
        assert stats.bootstrap_ci([1.0]) == (pytest.approx(np.nan, nan_ok=True),) * 2
        assert stats.bootstrap_ci([2.0, 2.0, 2.0]) == (2.0, 2.0)
        assert stats.bootstrap_ci([np.nan, 1.0]) == (pytest.approx(np.nan, nan_ok=True),) * 2

    def test_few_distinct_values_still_finite(self):
        low, high = stats.bootstrap_ci([1, 1, 1, 1, 1, 1, 1, 1, 1, 2])
        assert np.isfinite(low) and np.isfinite(high) and low <= 1.1 <= high

    def test_skewed_sample_gives_asymmetric_interval(self):
        x = np.array([-0.5, 1, 1.2, 0.9, 1.1, 0.7, 1.3, 0.8, 1.0, 1.05])
        low, high = stats.bootstrap_ci(x)
        m = x.mean()
        assert m - low > high - m  # the outlier pulls the lower bound down


class TestSignFlipP:
    def test_hand_checked_examples(self):
        # [1, 2, 3, 4]: only the all-plus and all-minus patterns reach |mean| = 2.5
        assert stats.signflip_p([1, 2, 3, 4]) == pytest.approx(2 / 16)
        # five equal differences: 2 of 32 patterns
        assert stats.signflip_p([1, 1, 1, 1, 1]) == pytest.approx(2 / 32)
        # a zero difference doubles every pattern: (0, 1, 2) -> 4 of 8
        assert stats.signflip_p([0, 1, 2]) == pytest.approx(0.5)
        # a balanced vector is never significant
        assert stats.signflip_p([-1, 1, -1, 1]) == pytest.approx(1.0)

    def test_floor_at_n10_is_two_over_1024(self):
        d = np.full(10, 0.3)
        assert stats.signflip_p(d) == pytest.approx(2 / 1024)

    def test_tolerance_is_scale_free(self):
        assert stats.signflip_p(np.full(8, 1e-13)) == pytest.approx(2 / 256)
        assert stats.signflip_p(np.full(8, 1e13)) == pytest.approx(2 / 256)

    def test_nan_for_degenerate(self):
        assert np.isnan(stats.signflip_p([0.0, 0.0, 0.0]))
        assert np.isnan(stats.signflip_p([1.0]))

    def test_monte_carlo_branch_matches_exact(self):
        rng = np.random.default_rng(3)
        d = rng.normal(0.4, 1.0, size=12)
        exact = stats.signflip_p(d)
        mc = stats.signflip_p(d, max_exact_n=8, n_mc=50_000)
        assert mc == pytest.approx(exact, abs=0.01)


class TestHolm:
    def test_step_down(self):
        adj = stats.holm([0.002, 0.03, 0.2])
        assert adj == pytest.approx([0.006, 0.06, 0.2])

    def test_monotone_capped_and_order_preserving(self):
        raw = [0.04, 0.01, 0.5, 0.03]
        adj = stats.holm(raw)
        # sorted raw: 0.01*4, 0.03*3, 0.04*2, 0.5*1 -> 0.04, 0.09, 0.09, 0.5
        assert adj == pytest.approx([0.09, 0.04, 0.5, 0.09])
        assert stats.holm([0.9, 0.95])[0] == pytest.approx(1.0)

    def test_nan_passthrough(self):
        adj = stats.holm([0.01, np.nan, 0.02])
        assert np.isnan(adj[1]) and adj[0] == pytest.approx(0.02) and adj[2] == pytest.approx(0.02)
        assert np.isnan(stats.holm([np.nan])).all()
