"""Tests for the multi-head attention study runner."""

from unittest.mock import patch

from skewed_sequences.experiments.run_experiments.head_attention_data import (
    EMBED_DIM,
    HEAD_COUNTS,
    head_sweep_loss_configs,
    main,
)


def test_loss_configs_count_and_types():
    cfgs = head_sweep_loss_configs()
    assert len(cfgs) == 10  # 6 classical baselines + 4 representative SGT points
    loss_types = {c["loss_type"] for c in cfgs}
    assert {"mse", "mae", "cauchy", "huber", "tukey", "charbonnier"} <= loss_types
    sgt = [c for c in cfgs if c["loss_type"] == "sgt"]
    assert len(sgt) == 4
    assert all(c["sgt_loss_p"] == 2.0 and c["sgt_loss_lambda"] == 0.0 for c in sgt)
    assert {c["sgt_loss_q"] for c in sgt} == {1.3, 2.5, 10.0, 20.0}


def test_embed_dim_divisible_by_every_head_count():
    # nn.Transformer requires d_model % nhead == 0.
    assert all(EMBED_DIM % h == 0 for h in HEAD_COUNTS)


@patch("skewed_sequences.experiments.run_experiments.head_attention_data.run_training_config")
@patch("skewed_sequences.experiments.run_experiments.head_attention_data.generate_data_main")
def test_main_sweeps_heads_with_fixed_width_and_shared_seed(mock_gen, mock_run):
    main(n_runs=1, n_sequences=10, num_epochs=1)

    assert mock_gen.call_count == 1  # heavy-tailed dataset generated once
    assert mock_run.call_count == len(HEAD_COUNTS) * len(head_sweep_loss_configs())

    heads_seen = set()
    for call in mock_run.call_args_list:
        kw = call.kwargs
        assert kw["embed_dim"] == EMBED_DIM  # width held fixed
        assert kw["model_type"] == "transformer"
        heads_seen.add(kw["num_heads"])
    assert heads_seen == set(HEAD_COUNTS)

    # All runs within one run_idx share the same seed (seed-paired head configs).
    seeds = {call.kwargs["seed"] for call in mock_run.call_args_list}
    assert len(seeds) == 1
