"""Tests for the shared grid-runner helper."""

from unittest.mock import patch

from skewed_sequences.config import OUTPUT_LENGTH
from skewed_sequences.experiments.run_experiments._runner import run_training_config


@patch("skewed_sequences.experiments.run_experiments._runner.train_main")
def test_sgt_config_expands_loss_params(mock_train):
    cfg = {
        "loss_type": "sgt",
        "sgt_loss_p": 1.5,
        "sgt_loss_q": 10.0,
        "sgt_loss_sigma": 1.0,
        "sgt_loss_lambda": 0.0,
        "output_length": 1,
    }
    run_training_config(
        cfg,
        dataset_path="d.npy",
        experiment_name="normal_run_1",
        seed=7,
        stride=1,
        batch_size=8,
        exp_transform=True,
    )
    kw = mock_train.call_args.kwargs
    assert kw["loss_type"] == "sgt"
    assert (kw["sgt_loss_p"], kw["sgt_loss_q"], kw["sgt_loss_lambda"]) == (1.5, 10.0, 0.0)
    assert kw["output_length"] == 1
    assert kw["experiment_name"] == "normal_run_1" and kw["seed"] == 7
    assert kw["batch_size"] == 8 and kw["exp_transform"] is True


@patch("skewed_sequences.experiments.run_experiments._runner.train_main")
def test_classical_config_omits_sgt_params(mock_train):
    run_training_config(
        {"loss_type": "mse", "output_length": 1},
        dataset_path="d.npy",
        experiment_name="lanl_mse_run_2",
        seed=3,
    )
    kw = mock_train.call_args.kwargs
    assert kw["loss_type"] == "mse"
    assert "sgt_loss_q" not in kw and "sgt_loss_p" not in kw


@patch("skewed_sequences.experiments.run_experiments._runner.train_main")
def test_missing_output_length_falls_back_to_constant(mock_train):
    run_training_config(
        {"loss_type": "mae"}, dataset_path="d.npy", experiment_name="x_run_1", seed=1
    )
    assert mock_train.call_args.kwargs["output_length"] == OUTPUT_LENGTH
