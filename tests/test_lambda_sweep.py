"""Tests for the fine lambda sub-sweep runner."""

from unittest.mock import patch

import pytest
import typer

from skewed_sequences.config import LAMBDA_SWEEP_CONFIGS, TRAINING_CONFIGS
from skewed_sequences.experiments.run_experiments.lambda_sweep_data import (
    DEFAULT_DATASETS,
    lambda_sweep_loss_configs,
    main,
)

MOD = "skewed_sequences.experiments.run_experiments.lambda_sweep_data"


def test_loss_configs():
    cfgs = lambda_sweep_loss_configs()
    assert cfgs == LAMBDA_SWEEP_CONFIGS and len(cfgs) == 12
    assert all(c["loss_type"] == "sgt" and c["sgt_loss_lambda"] > 0 for c in cfgs)
    assert {c["sgt_loss_lambda"] for c in cfgs} == {0.1, 0.2, 0.3}
    assert {(c["sgt_loss_p"], c["sgt_loss_q"]) for c in cfgs} == {
        (2.0, 2.5),
        (2.0, 10.0),
        (1.5, 2.5),
        (1.5, 10.0),
    }


def test_every_anchor_has_a_symmetric_twin_in_the_main_grid():
    """The paired lambda-vs-0 test needs the lambda=0 config of each (p, q)."""
    symmetric = {
        (c["sgt_loss_p"], c["sgt_loss_q"])
        for c in TRAINING_CONFIGS
        if c["loss_type"] == "sgt" and c["sgt_loss_lambda"] == 0.0
    }
    assert {(c["sgt_loss_p"], c["sgt_loss_q"]) for c in LAMBDA_SWEEP_CONFIGS} <= symmetric


@patch(f"{MOD}.logged_experiment_param", return_value="5")
@patch(f"{MOD}.draw_experiment_seed", return_value=99)
@patch(f"{MOD}.run_training_config")
@patch(f"{MOD}.generate_data_main")
def test_main_appends_to_existing_experiments(mock_gen, mock_run, mock_seed, mock_param):
    main(n_runs=2, n_sequences=10, num_epochs=1, stride=5, resume=True)
    mock_param.assert_any_call(f"{DEFAULT_DATASETS[0]}_run_1", "transformer", "stride")

    assert mock_gen.call_count == 1  # one dataset (the default)
    assert mock_run.call_count == 2 * len(LAMBDA_SWEEP_CONFIGS)
    names = {c.kwargs["experiment_name"] for c in mock_run.call_args_list}
    assert names == {f"{DEFAULT_DATASETS[0]}_run_1", f"{DEFAULT_DATASETS[0]}_run_2"}
    assert all(
        c.kwargs["seed"] == 99 and c.kwargs["resume"] is True for c in mock_run.call_args_list
    )
    mock_seed.assert_any_call(f"{DEFAULT_DATASETS[0]}_run_1", "transformer", True)


@patch(f"{MOD}.draw_experiment_seed", return_value=1)
@patch(f"{MOD}.run_training_config")
@patch(f"{MOD}.generate_data_main")
def test_dataset_option(mock_gen, mock_run, mock_seed):
    main(n_runs=1, n_sequences=10, dataset=["normal-skewed", "heavy-tailed-skewed"], resume=False)
    assert mock_gen.call_count == 2
    names = {c.kwargs["experiment_name"] for c in mock_run.call_args_list}
    assert names == {"normal-skewed_run_1", "heavy-tailed-skewed_run_1"}


def test_unknown_dataset_raises():
    with pytest.raises(typer.BadParameter):
        main(n_runs=1, dataset=["nope"], resume=False)


@patch(f"{MOD}.logged_experiment_param", return_value="1")
@patch(f"{MOD}.run_training_config")
@patch(f"{MOD}.generate_data_main")
def test_refuses_stride_mismatch_with_logged_experiment(mock_gen, mock_run, mock_param):
    """The main sweep logged stride=1; appending lambda runs at stride=5 is refused."""
    with pytest.raises(typer.BadParameter, match="stride=1"):
        main(n_runs=1, n_sequences=10, stride=5, resume=True)
    mock_run.assert_not_called()


def test_defaults_match_the_synthetic_sweep():
    """The lambda runner must regenerate the same data as run-synthetic / run-head-sweep."""
    import inspect

    from skewed_sequences.config import SYNTHETIC_N_SEQUENCES, SYNTHETIC_STRIDE
    from skewed_sequences.experiments.run_experiments import head_attention_data, synthetic_data

    for mod in (synthetic_data, head_attention_data):
        params = inspect.signature(mod.main).parameters
        assert params["n_sequences"].default == SYNTHETIC_N_SEQUENCES
        assert params["stride"].default == SYNTHETIC_STRIDE
    params = inspect.signature(main).parameters
    assert params["n_sequences"].default == SYNTHETIC_N_SEQUENCES
    assert params["stride"].default == SYNTHETIC_STRIDE
