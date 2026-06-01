"""Tests for skewed_sequences.data._common (shared loader helpers)."""

import numpy as np

from skewed_sequences.data._common import scale_and_stack, slice_array_to_chunks


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
