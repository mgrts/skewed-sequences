"""Tests for skewed_sequences.data._common (shared loader helpers)."""

import numpy as np
import pytest

from skewed_sequences.data._common import (
    check_not_step_function,
    scale_and_stack,
    slice_array_to_chunks,
    zero_increment_fraction,
)


class TestSliceArrayToChunks:
    def test_non_overlapping_drops_remainder(self):
        chunks = slice_array_to_chunks(np.arange(700), 300)
        assert chunks.shape == (2, 300)
        # Disjoint: second chunk starts exactly where the first ends (no overlap).
        assert chunks[0][-1] == 299
        assert chunks[1][0] == 300

    def test_exact_multiple(self):
        chunks = slice_array_to_chunks(np.arange(600), 300)
        assert chunks.shape == (2, 300)

    def test_shorter_than_chunk_returns_empty(self):
        assert len(slice_array_to_chunks(np.arange(150), 300)) == 0


class TestScaleAndStack:
    def test_shape_and_per_chunk_standardization(self):
        chunks = [np.arange(300, dtype=float), np.arange(300, dtype=float) * 2.0]
        out = scale_and_stack(chunks, 300)
        assert out.shape == (2, 300, 1)
        assert np.allclose(out[:, :, 0].mean(axis=1), 0.0, atol=1e-6)
        assert np.allclose(out[:, :, 0].std(axis=1), 1.0, atol=1e-3)

    def test_drops_short_chunks(self):
        chunks = [np.arange(300, dtype=float), np.arange(150, dtype=float)]
        out = scale_and_stack(chunks, 300)
        assert out.shape == (1, 300, 1)


class TestStepFunctionCheck:
    def test_zero_increment_fraction(self):
        data = np.array([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3]], dtype=float)[..., np.newaxis]
        # 9 increments, 3 nonzero -> 6/9 zero
        assert zero_increment_fraction(data) == pytest.approx(6 / 9)

    def test_smooth_data_passes_and_returns_fraction(self):
        t = np.linspace(0, 10, 300)
        data = np.stack([np.sin(t), np.cos(t)])[..., np.newaxis]
        assert check_not_step_function(data) == 0.0

    def test_weekly_step_data_raises(self):
        weekly = np.repeat(np.arange(50.0), 7)[:300]  # constant within each week
        data = np.tile(weekly[np.newaxis, :, np.newaxis], (3, 1, 1))
        with pytest.raises(ValueError, match="piecewise constant"):
            check_not_step_function(data, name="weekly")

    def test_threshold_is_configurable(self):
        weekly = np.repeat(np.arange(50.0), 7)[:300]
        data = weekly[np.newaxis, :, np.newaxis]
        assert check_not_step_function(data, max_zero_increment_fraction=0.95) > 0.8
