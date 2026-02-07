import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_time_series(time_series):
    """
    Plots the given time series.

    Parameters:
        time_series (np.ndarray): The time series to plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(time_series, label="Non-Stationary Time Series with Varying Variance")
    plt.title("Non-Stationary Time Series Generation")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_prediction(
    src: torch.Tensor,
    tgt: torch.Tensor,
    pred: torch.Tensor,
    pred_infer: torch.Tensor,
    idx=0,
    zoom_context: int = 20,
):
    """Visualizes a given sample including predictions.

    When the prediction horizon is short relative to the input (tgt_len < 10%
    of src_len), an additional zoomed subplot is added showing only the last
    ``zoom_context`` input steps plus the prediction region.

    Args:
        src: source sequence [bs, src_seq_len, num_features]
        tgt: target sequence [bs, tgt_seq_len, num_features]
        pred: prediction of the model [bs, tgt_seq_len, num_features]
        pred_infer: prediction obtained by running inference
            [bs, tgt_seq_len, num_features]
        idx: batch index to visualize
        zoom_context: number of input timesteps to show before prediction
            in the zoomed subplot
    """
    src_len = src.shape[1]
    tgt_len = tgt.shape[1]
    x = np.arange(src_len + tgt_len)

    need_zoom = tgt_len < 0.1 * src_len

    if need_zoom:
        fig, (ax_full, ax_zoom) = plt.subplots(
            1,
            2,
            figsize=(24, 8),
            gridspec_kw={"width_ratios": [2, 1]},
        )
    else:
        fig, ax_full = plt.subplots(figsize=(20, 10))

    # --- Full view ---
    ax_full.plot(x[:src_len], src[idx].cpu().detach(), "bo-", markersize=1, label="src")
    ax_full.plot(x[src_len:], tgt[idx].cpu().detach(), "go-", markersize=2, label="tgt")
    ax_full.plot(x[src_len:], pred[idx].cpu().detach(), "ro-", markersize=2, label="pred")
    ax_full.plot(
        x[src_len:], pred_infer[idx].cpu().detach(), "yo-", markersize=2, label="pred_infer"
    )
    ax_full.set_title("Full sequence")
    ax_full.legend()

    if need_zoom:
        # --- Zoomed view on prediction region ---
        zoom_start = max(0, src_len - zoom_context)
        ax_zoom.plot(
            x[zoom_start:src_len],
            src[idx, zoom_start:src_len].cpu().detach(),
            "bo-",
            label="src (context)",
        )
        ax_zoom.plot(x[src_len:], tgt[idx].cpu().detach(), "go-", label="tgt")
        ax_zoom.plot(x[src_len:], pred[idx].cpu().detach(), "ro-", label="pred")
        ax_zoom.plot(x[src_len:], pred_infer[idx].cpu().detach(), "yo-", label="pred_infer")
        ax_zoom.set_title(
            f"Prediction region (last {zoom_context} input + {tgt_len} output steps)"
        )
        ax_zoom.legend()

        # Add a shaded region on the full plot to indicate the zoomed area
        ax_full.axvspan(zoom_start, src_len + tgt_len, alpha=0.1, color="orange")

    fig.tight_layout()
    return fig
