"""Single source of truth for the MLflow param/metric key names.

These keys are an untyped contract: ``modeling/train.py`` logs them and
``experiments/collect_results.py`` reads them back with literal ``.get()`` calls
(CLAUDE.md invariant #6). A silent rename on either side produces NaN CSV columns
with no failing test, so the names live here once and both sides import them. A
producer/consumer test (``tests/test_mlflow_contract.py``) pins that the keys
``train.main`` actually logs match this module.

Lightweight by design (no heavy imports): safe to import from ``train.py`` and
``collect_results.py`` without violating the lazy-import invariants.
"""

# Per-split summary metrics logged once at the end of training (best checkpoint).
SPLITS = ("train", "val", "test")
SUMMARY_METRIC_STEMS = ("smape", "mape", "rmse", "mae")


def summary_metric_key(split: str, stem: str) -> str:
    """e.g. ('test', 'rmse') -> 'best_test_rmse'."""
    return f"best_{split}_{stem}"


SUMMARY_METRIC_KEYS = tuple(
    summary_metric_key(split, stem) for split in SPLITS for stem in SUMMARY_METRIC_STEMS
)

# Naive last-value (persistence) baseline + MASE, logged once per run on the test
# split so headroom above the trivial forecaster is visible.
BASELINE_METRIC_KEYS = ("best_test_naive_rmse", "best_test_naive_mae", "best_test_mase")

# Everything train.main logs via mlflow.log_metrics at the end of a run.
ALL_SUMMARY_METRIC_KEYS = SUMMARY_METRIC_KEYS + BASELINE_METRIC_KEYS

# Per-epoch metrics logged by trainer.py during training.
PER_EPOCH_METRIC_STEMS = ("loss", *SUMMARY_METRIC_STEMS)

# The exact set of params logged by train.main. The seed is logged as
# ``random_state`` (NOT ``seed``), per invariant #6.
PARAM_KEYS = frozenset(
    {
        "model_type",
        "sgt_loss_lambda",
        "sgt_loss_q",
        "sgt_loss_sigma",
        "sgt_loss_p",
        "context_length",
        "output_length",
        "stride",
        "embed_dim",
        "num_heads",
        "num_layers",
        "batch_size",
        "learning_rate",
        "loss_type",
        "early_stopping_patience",
        "num_epochs",
        "test_split",
        "val_split",
        "exp_transform",
        "random_state",
        "residual_scale",
    }
)
