from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from skewed_sequences.style import COLORS, apply_style


def visualize_sliding_window_prediction(
    sequence: np.ndarray,
    dense_agg: Optional[dict],
    tiled_predictions: Optional[List[Tuple[int, np.ndarray]]],
    context_len: int,
    output_len: int,
    zoom_context: int = 20,
) -> plt.Figure:
    """Visualize dense and tiled sliding-window predictions over a full sequence.

    Args:
        sequence: Ground-truth values, shape ``(T,)`` or ``(T, 1)``.
        dense_agg: Dict with keys ``"mean"``, ``"min"``, ``"max"`` — arrays
            of shape ``(T,)`` produced by :func:`aggregate_dense_predictions`.
            ``None`` to skip the dense overlay.
        tiled_predictions: List of ``(start_idx, pred_array)`` tuples from
            stride=output_len inference.  ``None`` to skip.
        context_len: Number of context time-steps (drawn as a vertical line).
        output_len: Prediction horizon length.
        zoom_context: How many context steps to show before the prediction
            region in the optional zoom panel.

    Returns:
        A :class:`matplotlib.figure.Figure`.
    """
    apply_style()

    seq = sequence.squeeze()
    T = len(seq)

    pred_start = context_len
    pred_len = T - context_len
    need_zoom = pred_len < 0.15 * T

    if need_zoom:
        fig, (ax_full, ax_zoom) = plt.subplots(
            1,
            2,
            figsize=(22, 7),
            gridspec_kw={"width_ratios": [2, 1]},
        )
    else:
        fig, ax_full = plt.subplots(figsize=(18, 7))
        ax_zoom = None

    x = np.arange(T)

    # Ground truth
    ax_full.plot(x, seq, color=COLORS["ground_truth"], linewidth=1.0, label="Ground truth")

    # Dense: mean line + min/max envelope
    if dense_agg is not None:
        mean = dense_agg["mean"]
        mn = dense_agg["min"]
        mx = dense_agg["max"]
        valid = ~np.isnan(mean)
        ax_full.plot(
            x[valid],
            mean[valid],
            color=COLORS["dense_mean"],
            linewidth=1.2,
            label="Dense mean",
        )
        ax_full.fill_between(
            x[valid],
            mn[valid],
            mx[valid],
            color=COLORS["dense_band"],
            alpha=0.25,
            label="Dense min–max",
        )

    # Tiled blocks
    if tiled_predictions is not None:
        for i, (start, pred) in enumerate(tiled_predictions):
            pred_1d = pred.squeeze()
            xs = np.arange(start, start + len(pred_1d))
            label = "Tiled prediction" if i == 0 else None
            ax_full.plot(xs, pred_1d, color=COLORS["tiled"], linewidth=1.0, label=label)

    # Context boundary
    ax_full.axvline(
        pred_start,
        color=COLORS["context_boundary"],
        linestyle="--",
        linewidth=0.8,
        label="Context boundary",
    )

    ax_full.set_title("Sliding-window prediction")
    ax_full.set_xlabel("Timestep")
    ax_full.set_ylabel("Value")
    ax_full.legend(fontsize=8)

    # Zoom panel
    if ax_zoom is not None:
        zoom_start = max(0, pred_start - zoom_context)
        ax_zoom.plot(
            x[zoom_start:],
            seq[zoom_start:],
            color=COLORS["ground_truth"],
            linewidth=1.0,
            label="Ground truth",
        )
        if dense_agg is not None:
            valid_z = ~np.isnan(mean[zoom_start:])
            xz = x[zoom_start:]
            ax_zoom.plot(
                xz[valid_z],
                mean[zoom_start:][valid_z],
                color=COLORS["dense_mean"],
                linewidth=1.2,
                label="Dense mean",
            )
            ax_zoom.fill_between(
                xz[valid_z],
                mn[zoom_start:][valid_z],
                mx[zoom_start:][valid_z],
                color=COLORS["dense_band"],
                alpha=0.25,
            )
        if tiled_predictions is not None:
            for i, (start, pred) in enumerate(tiled_predictions):
                pred_1d = pred.squeeze()
                xs = np.arange(start, start + len(pred_1d))
                mask = xs >= zoom_start
                if mask.any():
                    ax_zoom.plot(
                        xs[mask],
                        pred_1d[mask],
                        color=COLORS["tiled"],
                        linewidth=1.0,
                    )
        ax_zoom.axvline(
            pred_start,
            color=COLORS["context_boundary"],
            linestyle="--",
            linewidth=0.8,
        )
        ax_zoom.set_title(
            f"Zoom: last {zoom_context} context + prediction region",
        )
        ax_zoom.set_xlabel("Timestep")
        ax_zoom.legend(fontsize=8)

        ax_full.axvspan(zoom_start, T, alpha=0.07, color="orange")

    fig.tight_layout()
    return fig
