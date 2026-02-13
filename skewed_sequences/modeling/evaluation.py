from typing import List, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch

from skewed_sequences.modeling.visualize import visualize_sliding_window_prediction


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


def aggregate_dense_predictions(
    predictions: List[Tuple[int, np.ndarray]],
    seq_len: int,
    n_features: int,
    output_len: int,
) -> dict:
    """Accumulate overlapping predictions into mean / min / max arrays.

    Args:
        predictions: Output of :func:`sliding_window_predictions` with stride=1.
        seq_len: Total length of the original sequence.
        n_features: Number of features per time-step.
        output_len: Prediction horizon (used only for documentation clarity).

    Returns:
        Dict with keys ``"mean"``, ``"min"``, ``"max"`` — each a numpy array
        of shape ``(seq_len,)`` (squeezed from the first feature dimension).
        Positions without predictions are ``NaN``.
    """
    accumulator = np.full((seq_len, n_features), 0.0, dtype=np.float64)
    minimum = np.full((seq_len, n_features), np.inf, dtype=np.float64)
    maximum = np.full((seq_len, n_features), -np.inf, dtype=np.float64)
    counts = np.zeros(seq_len, dtype=np.int64)

    for start_idx, pred in predictions:
        pred_2d = pred if pred.ndim == 2 else pred[:, None]
        end_idx = start_idx + pred_2d.shape[0]
        accumulator[start_idx:end_idx] += pred_2d
        minimum[start_idx:end_idx] = np.minimum(minimum[start_idx:end_idx], pred_2d)
        maximum[start_idx:end_idx] = np.maximum(maximum[start_idx:end_idx], pred_2d)
        counts[start_idx:end_idx] += 1

    has_data = counts > 0
    mean = np.full(seq_len, np.nan)
    mn = np.full(seq_len, np.nan)
    mx = np.full(seq_len, np.nan)

    mean[has_data] = accumulator[has_data, 0] / counts[has_data]
    mn[has_data] = minimum[has_data, 0]
    mx[has_data] = maximum[has_data, 0]

    return {"mean": mean, "min": mn, "max": mx}


def log_val_predictions(
    model: torch.nn.Module,
    val_data_raw: np.ndarray,
    model_path,
    context_len: int,
    output_len: int,
    num_vis_examples: int = 3,
) -> None:
    """Generate sliding-window prediction figures and log them to MLflow.

    For each of the first *num_vis_examples* validation sequences, runs two
    inference passes (dense stride=1 and tiled stride=output_len), produces
    a figure via :func:`visualize_sliding_window_prediction`, and logs it.
    """
    device = next(model.parameters()).device
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    n_seqs = val_data_raw.shape[0]
    n_features = val_data_raw.shape[2] if val_data_raw.ndim == 3 else 1

    for i in range(min(num_vis_examples, n_seqs)):
        seq = val_data_raw[i]  # (T, n_features) or (T,)

        # Dense predictions (stride=1)
        dense_preds = sliding_window_predictions(
            model,
            seq,
            context_len,
            output_len,
            stride=1,
            device=device,
        )
        dense_agg = aggregate_dense_predictions(
            dense_preds,
            seq_len=seq.shape[0],
            n_features=n_features,
            output_len=output_len,
        )

        # Tiled predictions (stride=output_len)
        tiled_preds = sliding_window_predictions(
            model,
            seq,
            context_len,
            output_len,
            stride=output_len,
            device=device,
        )

        fig = visualize_sliding_window_prediction(
            sequence=seq,
            dense_agg=dense_agg,
            tiled_predictions=tiled_preds,
            context_len=context_len,
            output_len=output_len,
        )
        mlflow.log_figure(fig, f"prediction_{i}.png")
        plt.close(fig)
