import matplotlib.pyplot as plt
import numpy as np

from skewed_sequences.visualization.style import COLORS, apply_style


def visualize_sliding_window_prediction(
    sequence: np.ndarray,
    predictions: np.ndarray,
    context_len: int,
) -> plt.Figure:
    """Visualize ground truth vs single-step autoregressive predictions.

    Args:
        sequence: Ground-truth values, shape ``(T,)`` or ``(T, 1)``.
        predictions: 1-D array of predicted values, shape ``(T,)``.
            Positions before *context_len* should be ``NaN``.
        context_len: Number of context time-steps (drawn as a vertical line).

    Returns:
        A :class:`matplotlib.figure.Figure`.
    """
    apply_style()

    seq = sequence.squeeze()
    T = len(seq)

    fig, ax = plt.subplots(figsize=(18, 7))
    x = np.arange(T)

    # Ground truth
    ax.plot(x, seq, color=COLORS["ground_truth"], linewidth=1.0, label="Ground truth")

    # Predictions
    valid = ~np.isnan(predictions)
    ax.plot(
        x[valid],
        predictions[valid],
        color=COLORS["prediction"],
        linewidth=1.0,
        label="Prediction",
    )

    # Context boundary
    ax.axvline(
        context_len,
        color=COLORS["context_boundary"],
        linestyle="--",
        linewidth=0.8,
        label="Context boundary",
    )

    ax.set_title("Single-step autoregressive prediction")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Value")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig
