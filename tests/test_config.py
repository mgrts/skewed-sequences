"""Tests for skewed_sequences.config — paths and constants."""

from skewed_sequences.config import (
    CONTEXT_LENGTH,
    DATA_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    N_RUNS,
    OUTPUT_LENGTH,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    SEED,
    SEQUENCE_LENGTH,
    STRIDE,
    SYNTHETIC_DATA_CONFIGS,
    TRAINING_CONFIGS,
)


def test_proj_root_exists():
    assert PROJ_ROOT.is_dir()


def test_path_hierarchy():
    assert DATA_DIR == PROJ_ROOT / "data"
    assert PROCESSED_DATA_DIR == DATA_DIR / "processed"
    assert MODELS_DIR == PROJ_ROOT / "models"
    assert FIGURES_DIR == PROJ_ROOT / "reports" / "figures"


def test_constants():
    assert isinstance(SEED, int) and SEED > 0
    assert SEQUENCE_LENGTH == 300
    assert CONTEXT_LENGTH == 200
    assert STRIDE == 1
    assert OUTPUT_LENGTH == 3
    assert N_RUNS >= 1


def test_synthetic_data_configs():
    assert len(SYNTHETIC_DATA_CONFIGS) == 4
    for cfg in SYNTHETIC_DATA_CONFIGS:
        assert "lam" in cfg and "q" in cfg and "sigma" in cfg and "experiment_name" in cfg
        assert -1 < cfg["lam"] < 1
        assert cfg["q"] > 0
        assert cfg["sigma"] > 0


def test_training_configs():
    assert len(TRAINING_CONFIGS) == 16
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
    assert loss_types == {"sgt", "mse", "mae", "cauchy", "huber", "tukey"}
