"""Tests for the shared grid-runner helper."""

from unittest.mock import patch

import pytest

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


# ---------------------------------------------------------------------------
# resume support: seed reuse + skip-finished (MLflow is mocked; never the real DB)
# ---------------------------------------------------------------------------

import random  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

from skewed_sequences.experiments.run_experiments._runner import (  # noqa: E402
    _param_filter,
    draw_experiment_seed,
    has_finished_run,
    logged_experiment_param,
    logged_experiment_seed,
)

RUNNER = "skewed_sequences.experiments.run_experiments._runner"
SGT_CFG = {
    "loss_type": "sgt",
    "sgt_loss_p": 2.0,
    "sgt_loss_q": 10.0,
    "sgt_loss_sigma": 1.0,
    "sgt_loss_lambda": 0.0,
    "output_length": 1,
}


def _mock_client(experiment_exists=True, runs=()):
    client = MagicMock()
    client.get_experiment_by_name.return_value = (
        MagicMock(experiment_id="7") if experiment_exists else None
    )
    client.search_runs.return_value = list(runs)
    return client


def _run_with_params(**params):
    run = MagicMock()
    run.data.params = {k: str(v) for k, v in params.items()}
    return run


def test_param_filter_matches_mlflow_stringification():
    f = _param_filter(
        {"loss_type": "sgt", "sgt_loss_q": 2.5, "sgt_loss_lambda": 0.0, "random_state": 42}
    )
    assert f == (
        "params.loss_type = 'sgt' and params.sgt_loss_q = '2.5' and "
        "params.sgt_loss_lambda = '0.0' and params.random_state = '42' and "
        "attributes.status = 'FINISHED'"
    )
    assert _param_filter({"a": 1}, status=None) == "params.a = '1'"


@patch(f"{RUNNER}._client")
def test_logged_seed_found_in_any_listed_experiment(mock_client):
    client = _mock_client(runs=[_run_with_params(random_state=123456, model_type="transformer")])
    client.get_experiment_by_name.side_effect = [None, MagicMock(experiment_id="9")]
    mock_client.return_value = client
    assert logged_experiment_seed(["lanl_mse_run_1", "lanl_sgt_run_1"], "transformer") == 123456
    assert (
        "params.model_type = 'transformer'" in client.search_runs.call_args.kwargs["filter_string"]
    )


@patch(f"{RUNNER}._client")
def test_draw_seed_reuses_logged_seed_on_resume(mock_client):
    mock_client.return_value = _mock_client(
        runs=[_run_with_params(random_state=123456, model_type="transformer")]
    )
    assert draw_experiment_seed("normal_run_1", "transformer", resume=True) == 123456


@patch(f"{RUNNER}._client")
def test_draw_seed_fresh_when_experiment_missing(mock_client):
    mock_client.return_value = _mock_client(experiment_exists=False)
    seed = draw_experiment_seed("normal_run_1", "transformer", resume=True, rng=random.Random(0))
    assert seed == random.Random(0).randint(0, 2**32 - 1)


@patch(f"{RUNNER}._client")
def test_draw_seed_without_resume_never_queries_mlflow(mock_client):
    seed = draw_experiment_seed("normal_run_1", "transformer", resume=False)
    assert 0 <= seed < 2**32
    mock_client.assert_not_called()


@patch(f"{RUNNER}._client")
def test_has_finished_run(mock_client):
    mock_client.return_value = _mock_client(runs=[MagicMock()])
    assert has_finished_run("normal_run_1", {"loss_type": "mse", "random_state": 1}) is True
    filter_string = mock_client.return_value.search_runs.call_args.kwargs["filter_string"]
    assert "attributes.status = 'FINISHED'" in filter_string
    mock_client.return_value = _mock_client(experiment_exists=False)
    assert has_finished_run("normal_run_1", {"loss_type": "mse"}) is False


@patch(f"{RUNNER}.has_finished_run", return_value=True)
@patch(f"{RUNNER}.train_main")
def test_resume_skips_finished_config_and_matches_on_the_right_keys(mock_train, mock_has):
    trained = run_training_config(
        SGT_CFG,
        dataset_path="d.npy",
        experiment_name="head-heavy-tailed_run_1",
        seed=7,
        stride=5,
        resume=True,
        model_type="transformer",
        embed_dim=256,
        num_heads=8,
        batch_size=32,
        num_epochs=1,
        early_stopping_patience=20,
        num_workers=4,
    )
    assert trained is False
    mock_train.assert_not_called()
    experiment_name, match = mock_has.call_args.args
    assert experiment_name == "head-heavy-tailed_run_1"
    assert match["random_state"] == 7 and match["stride"] == 5 and match["output_length"] == 1
    assert (match["sgt_loss_p"], match["sgt_loss_q"], match["sgt_loss_lambda"]) == (2.0, 10.0, 0.0)
    assert match["model_type"] == "transformer" and match["embed_dim"] == 256
    assert match["num_heads"] == 8
    # The training budget IS identity: a 1-epoch smoke run must not shadow a sweep run.
    assert (match["batch_size"], match["num_epochs"], match["early_stopping_patience"]) == (
        32,
        1,
        20,
    )
    assert "num_workers" not in match  # not logged by train.main -> must not be matched


@patch(f"{RUNNER}.has_finished_run", return_value=False)
@patch(f"{RUNNER}.train_main")
def test_resume_trains_when_not_finished(mock_train, mock_has):
    trained = run_training_config(
        {"loss_type": "mse", "output_length": 1},
        dataset_path="d.npy",
        experiment_name="normal_run_1",
        seed=3,
        resume=True,
    )
    assert trained is True
    mock_train.assert_called_once()
    assert "sgt_loss_q" not in mock_has.call_args.args[1]


@patch(f"{RUNNER}.has_finished_run")
@patch(f"{RUNNER}.train_main")
def test_no_resume_skips_the_lookup(mock_train, mock_has):
    assert run_training_config(SGT_CFG, dataset_path="d.npy", experiment_name="x", seed=1) is True
    mock_has.assert_not_called()
    mock_train.assert_called_once()


@patch(f"{RUNNER}._client")
def test_logged_param_scans_all_runs_and_refuses_mixed_values(mock_client):
    mock_client.return_value = _mock_client(
        runs=[
            _run_with_params(random_state=1, model_type="transformer", stride=5),
            _run_with_params(random_state=1, model_type="transformer", stride=5),
        ]
    )
    assert logged_experiment_param("normal_run_1", "transformer", "stride") == "5"
    assert logged_experiment_seed("normal_run_1", "transformer") == 1
    kwargs = mock_client.return_value.search_runs.call_args.kwargs
    assert kwargs["max_results"] > 1  # every run is inspected, not just the newest

    mock_client.return_value = _mock_client(
        runs=[
            _run_with_params(random_state=1, model_type="transformer"),
            _run_with_params(random_state=2, model_type="transformer"),
        ]
    )
    with pytest.raises(RuntimeError, match="2 distinct values"):
        draw_experiment_seed("normal_run_1", "transformer", resume=True)


@patch(f"{RUNNER}._client")
def test_logged_param_none_when_key_absent(mock_client):
    mock_client.return_value = _mock_client(runs=[_run_with_params(model_type="transformer")])
    assert logged_experiment_param("normal_run_1", "transformer", "stride") is None
