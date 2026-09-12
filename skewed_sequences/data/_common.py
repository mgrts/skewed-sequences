"""Shared helpers for the per-entity time-series loaders.

The owid / rvr / health / lanl loaders all do the same thing to each entity's
1-D series: slice it into non-overlapping fixed-length chunks, per-chunk
StandardScaler-normalize, drop short chunks, stack, and add a feature axis. That
logic lived as four near-identical copies; it lives here once.

Imported only by the (lazily-loaded) data loaders, so the sklearn import here
does not affect the lazy-CLI invariants.
"""

from typing import Iterable

import numpy as np
from sklearn.preprocessing import StandardScaler


def slice_array_to_chunks(array, chunk_size: int) -> np.ndarray:
    """Slice a 1-D array into non-overlapping fixed-length chunks.

    The trailing remainder (shorter than ``chunk_size``) is dropped rather than
    left-aligned, which would overlap the previous chunk and leak identical
    timesteps across the train/test split.
    """
    n = len(array)
    chunks = [
        array[start : start + chunk_size] for start in range(0, n - chunk_size + 1, chunk_size)
    ]
    return np.array(chunks)


def scale_and_stack(chunks: Iterable, sequence_length: int) -> np.ndarray:
    """Per-chunk standardize, drop short chunks, and stack into ``(N, T, 1)``.

    Each chunk is standardized independently (per-sequence StandardScaler, before
    any split) so no cross-sequence statistics leak (CLAUDE.md invariant #11).
    """
    sequences = []
    for chunk in chunks:
        chunk = np.asarray(chunk)
        if len(chunk) < sequence_length:
            continue  # drop short trailing chunk; avoids a ragged np.vstack
        scaled = StandardScaler().fit_transform(chunk.reshape(-1, 1)).reshape(-1)
        sequences.append(scaled)
    stacked = np.vstack(sequences)
    return stacked[..., np.newaxis]


def zero_increment_fraction(sequences: np.ndarray) -> float:
    """Fraction of one-step increments that are *exactly* zero over ``(N, T, ...)`` data."""
    arr = np.asarray(sequences)
    increments = np.diff(arr[:, :, 0], axis=1)
    return float((increments == 0).mean()) if increments.size else 0.0


def check_not_step_function(
    sequences: np.ndarray,
    max_zero_increment_fraction: float = 0.5,
    name: str = "dataset",
) -> float:
    """Raise if more than ``max_zero_increment_fraction`` of the increments are exactly 0.

    A majority of exactly-zero increments means the series are piecewise constant —
    the fingerprint of weekly-reported data (one value per week, zeros or forward
    fills in between) run through a rolling mean. Such data has a zero median
    absolute deviation of increments, so ``residual_scale_estimate`` collapses and
    every scale-dependent loss (SGT / Cauchy / Huber / Tukey / Charbonnier) is
    degenerate; the persistence baseline is also near-perfect, so MASE is
    meaningless. Fail here, at load time, instead of training hundreds of garbage
    runs (the June-2026 OWID sweep did exactly that). Returns the fraction.
    """
    frac = zero_increment_fraction(sequences)
    if frac > max_zero_increment_fraction:
        raise ValueError(
            f"{name}: {frac:.1%} of one-step increments are exactly zero "
            f"(> {max_zero_increment_fraction:.0%}); the series are piecewise constant. "
            "This is the signature of weekly-reported source data (or a rolling mean over "
            "it). Fix the source/loader — the residual scale of such data is zero and "
            "every robust loss degenerates."
        )
    return frac
