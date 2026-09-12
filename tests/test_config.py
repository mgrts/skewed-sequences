"""Tests for skewed_sequences.config — paths and constants."""

from skewed_sequences.config import (
    CONTEXT_LENGTH,
    DATA_DIR,
    FIGURES_DIR,
    LAMBDA_SWEEP_CONFIGS,
    LAMBDA_SWEEP_LAMBDAS,
    N_RUNS,
    OUTPUT_LENGTH,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    SEED,
    SEQUENCE_LENGTH,
    STRIDE,
    SYNTHETIC_DATA_CONFIGS,
    SYNTHETIC_N_SEQUENCES,
    SYNTHETIC_STRIDE,
    TRAINING_CONFIGS,
)


def test_proj_root_exists():
    assert PROJ_ROOT.is_dir()


def test_path_hierarchy():
    assert DATA_DIR == PROJ_ROOT / "data"
    assert PROCESSED_DATA_DIR == DATA_DIR / "processed"
    assert FIGURES_DIR == PROJ_ROOT / "reports" / "figures"


def test_constants():
    assert isinstance(SEED, int) and SEED > 0
    assert SEQUENCE_LENGTH == 300
    assert CONTEXT_LENGTH == 200
    assert STRIDE == 1
    assert OUTPUT_LENGTH == 1
    assert N_RUNS >= 1
    assert SYNTHETIC_N_SEQUENCES == 1000
    assert SYNTHETIC_STRIDE == 5


def test_synthetic_data_configs():
    assert len(SYNTHETIC_DATA_CONFIGS) == 4
    for cfg in SYNTHETIC_DATA_CONFIGS:
        assert "lam" in cfg and "q" in cfg and "sigma" in cfg and "experiment_name" in cfg
        assert -1 < cfg["lam"] < 1
        assert cfg["q"] > 0
        assert cfg["sigma"] > 0


def test_training_configs():
    assert len(TRAINING_CONFIGS) == 36
    for cfg in TRAINING_CONFIGS:
        assert "loss_type" in cfg
        assert "output_length" in cfg
        assert cfg["output_length"] == OUTPUT_LENGTH
        if cfg["loss_type"] == "sgt":
            assert "sgt_loss_p" in cfg
            assert "sgt_loss_q" in cfg
            assert "sgt_loss_sigma" in cfg
            assert "sgt_loss_lambda" in cfg
    loss_types = {c["loss_type"] for c in TRAINING_CONFIGS}
    assert loss_types == {"sgt", "mse", "mae", "cauchy", "huber", "tukey", "charbonnier"}

    # The SGT skew sweep must be present (the asymmetric loss is actually exercised).
    sgt_lambdas = {c["sgt_loss_lambda"] for c in TRAINING_CONFIGS if c["loss_type"] == "sgt"}
    assert sgt_lambdas != {0.0}, "expected at least one nonzero-lambda (skewed) SGT config"
    # Every SGT config must satisfy the validity domain q**p > 2/p.
    for c in TRAINING_CONFIGS:
        if c["loss_type"] == "sgt":
            assert c["sgt_loss_q"] ** c["sgt_loss_p"] > 2.0 / c["sgt_loss_p"]


def test_lambda_sweep_configs():
    assert LAMBDA_SWEEP_LAMBDAS == (0.1, 0.2, 0.3)
    assert len(LAMBDA_SWEEP_CONFIGS) == 12
    for cfg in LAMBDA_SWEEP_CONFIGS:
        assert cfg["loss_type"] == "sgt"
        assert cfg["output_length"] == OUTPUT_LENGTH
        assert 0 < cfg["sgt_loss_lambda"] < 0.5  # below the main grid's {0.5, 0.9}
        assert cfg["sgt_loss_q"] ** cfg["sgt_loss_p"] > 2.0 / cfg["sgt_loss_p"]
    # No overlap with the main grid (these are appended to it).
    main_keys = {
        (c["sgt_loss_p"], c["sgt_loss_q"], c["sgt_loss_lambda"])
        for c in TRAINING_CONFIGS
        if c["loss_type"] == "sgt"
    }
    sweep_keys = {
        (c["sgt_loss_p"], c["sgt_loss_q"], c["sgt_loss_lambda"]) for c in LAMBDA_SWEEP_CONFIGS
    }
    assert not (main_keys & sweep_keys)
