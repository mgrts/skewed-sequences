import matplotlib.pyplot as plt
import mlflow
import torch

from skewed_sequences.modeling.visualize import visualize_prediction


def log_val_predictions(model, val_loader, model_path, num_vis_examples: int = 5):
    """
    Log visualizations of predictions for num_vis_examples from the validation set.
    """
    device = next(model.parameters()).device
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        vis_count = 0
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            pred = model(src, tgt)
            pred_infer = model.infer(src, tgt_len=tgt.shape[1])

            batch_size = src.shape[0]
            for i in range(batch_size):
                if vis_count >= num_vis_examples:
                    break

                fig = visualize_prediction(src, tgt, pred, pred_infer, idx=i)
                mlflow.log_figure(fig, f"prediction_{vis_count}.png")
                plt.close(fig)

                vis_count += 1

            if vis_count >= num_vis_examples:
                break
