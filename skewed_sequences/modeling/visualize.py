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
    plt.plot(time_series, label='Non-Stationary Time Series with Varying Variance')
    plt.title('Non-Stationary Time Series Generation')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_prediction(
    src: torch.Tensor,
    tgt: torch.Tensor,
    pred: torch.Tensor,
    pred_infer: torch.Tensor,
    idx=0,
):
    """Visualizes a given sample including predictions.

    Args:
        src: source sequence [bs, src_seq_len, num_features]
        tgt: target sequence [bs, tgt_seq_len, num_features]
        pred: prediction of the model [bs, tgt_seq_len, num_features]
        pred_infer: prediction obtained by running inference
            [bs, tgt_seq_len, num_features]
        idx: batch index to visualize
    """
    x = np.arange(src.shape[1] + tgt.shape[1])
    src_len = src.shape[1]

    fig = plt.figure(figsize=(20, 10))

    plt.plot(x[:src_len], src[idx].cpu().detach(), 'bo-', label='src')
    plt.plot(x[src_len:], tgt[idx].cpu().detach(), 'go-', label='tgt')
    plt.plot(x[src_len:], pred[idx].cpu().detach(), 'ro-', label='pred')
    plt.plot(x[src_len:], pred_infer[idx].cpu().detach(), 'yo-', label='pred_infer')

    plt.legend()

    return fig
