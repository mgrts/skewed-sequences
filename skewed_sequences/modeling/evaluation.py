from typing import List, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch

from skewed_sequences.visualization.predictions import visualize_sliding_window_prediction


def sliding_window_predictions(
    model: torch.nn.Module,
    sequence: np.ndarray,
    context_len: int,
    output_len: int,
    stride: int,
    device: torch.device,
) -> List[Tuple[int, np.ndarray]]:
    """Run autoregressive inference on every sliding window of *sequence*.

    Args:
        model: Trained model with an ``infer(src, tgt_len)`` method.
        sequence: 1-D or 2-D array of shape ``(T,)`` or ``(T, n_features)``.
        context_len: Number of input time-steps per window.
        output_len: Number of predicted time-steps per window.
        stride: Step size between consecutive windows.
        device: Torch device for inference.

    Returns:
        List of ``(start_idx, prediction)`` tuples where *start_idx* is the
        absolute position of the first predicted time-step and *prediction*
        is a numpy array of shape ``(output_len,)`` or ``(output_len, n_features)``.
    """
    if sequence.ndim == 1:
        sequence = sequence[:, None]

    T = sequence.shape[0]
    window_total = context_len + output_len
    predictions: List[Tuple[int, np.ndarray]] = []

    for start in range(0, T - window_total + 1, stride):
        ctx = sequence[start : start + context_len]
        src = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model.infer(src, tgt_len=output_len)
        pred_np = pred.squeeze(0).cpu().numpy()
        predictions.append((start + context_len, pred_np))

    return predictions


def log_val_predictions(
    model: torch.nn.Module,
    val_data_raw: np.ndarray,
    model_path,
    context_len: int,
    output_len: int,
    num_vis_examples: int = 3,
) -> None:
    """Generate single-step prediction figures and log them to MLflow.

    For each of the first *num_vis_examples* validation sequences, runs
    single-step autoregressive inference (stride=1, output_len=1), collects
    predictions into a 1-D array, and logs a ground-truth vs prediction figure.
    """
    device = next(model.parameters()).device
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    n_seqs = val_data_raw.shape[0]

    for i in range(min(num_vis_examples, n_seqs)):
        seq = val_data_raw[i]  # (T, n_features) or (T,)

        preds = sliding_window_predictions(
            model,
            seq,
            context_len,
            output_len,
            stride=1,
            device=device,
        )

        # Collect single-step predictions into a 1-D array
        seq_1d = seq.squeeze()
        T = len(seq_1d)
        prediction_line = np.full(T, np.nan)
        for start_idx, pred in preds:
            prediction_line[start_idx] = pred.squeeze()[0] if pred.ndim > 1 else pred[0]

        fig = visualize_sliding_window_prediction(
            sequence=seq,
            predictions=prediction_line,
            context_len=context_len,
        )
        mlflow.log_figure(fig, f"prediction_{i}.png")
        plt.close(fig)
