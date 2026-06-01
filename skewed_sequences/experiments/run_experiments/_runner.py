"""Shared scaffolding for the grid runners.

Every grid runner (synthetic / lanl / owid / rvr) loops over ``TRAINING_CONFIGS``
and calls ``train.main`` once per (config, run), expanding the SGT loss params
only for ``loss_type == "sgt"``. That call lived as a duplicated if/else block in
each runner; it lives here once.
"""

from pathlib import Path

from skewed_sequences.config import CONTEXT_LENGTH, OUTPUT_LENGTH, STRIDE
from skewed_sequences.modeling.train import main as train_main


def run_training_config(
    train_config: dict,
    *,
    dataset_path: Path,
    experiment_name: str,
    seed: int,
    stride: int = STRIDE,
    context_length: int = CONTEXT_LENGTH,
    **extra_train_kwargs,
) -> None:
    """Run ``train.main`` for a single grid config.

    Loop literals (batch_size/num_epochs/early_stopping_patience/num_workers) are
    NOT re-listed here — they default in ``train.main`` from ``config.py``. Pass
    overrides (and runner-specific kwargs like ``exp_transform``) via
    ``extra_train_kwargs``.
    """
    loss_type = train_config["loss_type"]

    kwargs = dict(
        dataset_path=dataset_path,
        loss_type=loss_type,
        # Explicit output_length key is required on every config; OUTPUT_LENGTH is
        # the (correct) fallback so a dropped key can never silently change the
        # horizon (CLAUDE.md invariant #3).
        output_length=train_config.get("output_length", OUTPUT_LENGTH),
        context_length=context_length,
        stride=stride,
        experiment_name=experiment_name,
        seed=seed,
        **extra_train_kwargs,
    )
    if loss_type.lower() == "sgt":
        kwargs.update(
            sgt_loss_lambda=train_config["sgt_loss_lambda"],
            sgt_loss_q=train_config["sgt_loss_q"],
            sgt_loss_sigma=train_config["sgt_loss_sigma"],
            sgt_loss_p=train_config["sgt_loss_p"],
        )

    train_main(**kwargs)
