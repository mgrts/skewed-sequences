"""Producer/consumer test for the MLflow key contract.

train.main is the producer (logs params + summary metrics); collect_results is
the consumer (reads them). Both import the key names from
``skewed_sequences.mlflow_contract``; this test pins that the keys train.main
actually logs match that module, so a drift fails CI instead of silently
producing NaN CSV columns (CLAUDE.md invariant #6).
"""

from unittest.mock import MagicMock, patch

import numpy as np

from skewed_sequences.mlflow_contract import ALL_SUMMARY_METRIC_KEYS, PARAM_KEYS


@patch("skewed_sequences.modeling.train.log_val_predictions")
@patch("skewed_sequences.modeling.train.persistence_metrics")
@patch("skewed_sequences.modeling.train.train_model")
@patch("skewed_sequences.modeling.train.create_dataloaders")
@patch("skewed_sequences.modeling.train.mlflow")
@patch("skewed_sequences.modeling.train.np.load")
def test_train_logs_exactly_the_contract_keys(
    mock_load, mock_mlflow, mock_create_dl, mock_train_model, mock_persistence, mock_log_vis
):
    from skewed_sequences.modeling.train import main

    mock_load.return_value = np.zeros((4, 300, 1), dtype="float32")
    mock_create_dl.return_value = (MagicMock(), MagicMock(), MagicMock(), np.zeros((1, 300, 1)))
    metrics = {"smape": 1.0, "mape": 2.0, "rmse": 0.3, "mae": 0.2}
    mock_train_model.return_value = (0.5, metrics, metrics, metrics)
    mock_persistence.return_value = {"smape": 5.0, "mape": 6.0, "rmse": 0.6, "mae": 0.5}

    captured = {"params": {}, "metrics": {}}
    mock_mlflow.log_params.side_effect = lambda d: captured["params"].update(d)
    mock_mlflow.log_metrics.side_effect = lambda d, **kw: captured["metrics"].update(d)

    main(num_epochs=1)

    assert set(captured["params"]) == set(
        PARAM_KEYS
    ), "train.main param keys drifted from mlflow_contract.PARAM_KEYS"
    assert set(captured["metrics"]) == set(
        ALL_SUMMARY_METRIC_KEYS
    ), "train.main summary-metric keys drifted from mlflow_contract.ALL_SUMMARY_METRIC_KEYS"
