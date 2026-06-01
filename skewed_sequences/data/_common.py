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
