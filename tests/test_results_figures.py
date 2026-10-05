"""Tests for skewed_sequences.visualization.results_figures (publication figures/tables)."""

import pandas as pd
import pytest

from skewed_sequences.visualization import results_figures as rf

N_RUNS = 4


def _rows(dataset, loss, p, q, lam, base, seed_offset=0.0, extra=None):
    rows = []
    for run in range(N_RUNS):
        rows.append(
            {
                "dataset": dataset,
                "loss_type": loss,
                "sgt_loss_p": p,
                "sgt_loss_q": q,
                "sgt_loss_lambda": lam,
                "random_state": run,
                "status": "FINISHED",
                "best_test_mase": base + 0.01 * run + seed_offset,
                "best_test_smape": 10 * (base + 0.01 * run),
                **(extra or {}),
            }
        )
    return rows


@pytest.fixture()
def results_csv(tmp_path):
    rows = []
    for ds, shift in [("normal", 0.0), ("heavy-tailed", 0.5), ("covid-owid", 1.0)]:
        for loss, base in [
            ("mse", 0.30),
            ("mae", 0.28),
            ("huber", 0.25),
            ("cauchy", 0.26),
            ("charbonnier", 0.25),
            ("tukey", 0.40 if ds != "covid-owid" else 60.0),
        ]:
            rows += _rows(ds, loss, 2.0, 2.0, 0.0, base + shift)
        for p in (1.0, 1.5, 2.0):
            for q in (2.5, 10.0, 20.0):
                rows += _rows(ds, "sgt", p, q, 0.0, 0.22 + 0.01 * p + 0.002 * q + shift)
        rows += _rows(ds, "sgt", 2.0, 10.0, 0.5, 0.35 + shift)
        if ds == "heavy-tailed":
            for lam in (0.1, 0.3):
                rows += _rows(ds, "sgt", 2.0, 10.0, lam, 0.23 + shift - 0.01)
    for nh in (1, 2, 4):
        rows += _rows(
            "head-heavy-tailed", "mse", 2.0, 2.0, 0.0, 0.9 - 0.01 * nh, extra={"num_heads": nh}
        )
        rows += _rows(
            "head-heavy-tailed", "sgt", 2.0, 2.5, 0.0, 0.88 - 0.01 * nh, extra={"num_heads": nh}
        )
    rows += _rows("covid-owid-weekly-INVALID", "mse", 2.0, 2.0, 0.0, 20.0)
    df = pd.DataFrame(rows)
    df.loc[0, "status"] = "RUNNING"
    path = tmp_path / "results.csv"
    df.to_csv(path, index=False)
    return path


def test_load_results_filters(results_csv):
    df = rf.load_results(results_csv)
    assert (df.status == "FINISHED").all()
    assert not df.dataset.str.contains("INVALID").any()


def test_loss_rows_and_best(results_csv):
    df = rf.load_results(results_csv)
    rows = rf.loss_rows(df[df.dataset == "normal"], "best_test_mase")
    assert set(rows.key) >= {"mse", "mae", "huber", "cauchy", "charbonnier", "tukey", "sgt_best"}
    assert (rows.n >= N_RUNS - 1).all()
    best = rf.best_sgt_config(df[df.dataset == "normal"], "best_test_mase")
    assert best == (1.0, 2.5, 0.0)  # lowest synthetic mean by construction


def test_paired_diff_is_seed_paired(results_csv):
    df = rf.load_results(results_csv)
    d = df[df.dataset == "heavy-tailed"]
    diff, ci, p, n = rf.paired_diff(
        d[d.loss_type == "huber"], d[d.loss_type == "mse"], "best_test_mase"
    )
    assert diff == pytest.approx(-0.05)
    assert n == N_RUNS and ci == pytest.approx(0.0, abs=1e-9)  # constant difference -> zero CI


def test_lambda_effect_table(results_csv):
    df = rf.load_results(results_csv)
    t = rf.lambda_effect_table(df, "best_test_mase")
    fine = t[(t.dataset == "heavy-tailed") & (t.lam == 0.1)].iloc[0]
    assert fine["diff"] < 0 and fine["n"] == N_RUNS
    assert (t[t.lam == 0.5]["diff"] > 0).all()
    assert not (t.dataset == "head-heavy-tailed").any()


def test_tables_shape(results_csv):
    df = rf.load_results(results_csv)
    main = rf.results_table(df, "best_test_mase")
    assert "SGT, best of grid (config)" in main.index
    assert list(main.columns) == ["Gaussian", "Heavy-tailed", "OWID COVID-19 cases"]
    assert main.loc["Tukey", "OWID COVID-19 cases"].startswith("61.")  # 60 + 1.0 shift
    summ = rf.summary_table(df, "best_test_mase")
    assert list(summ["Dataset"]) == ["Gaussian", "Heavy-tailed", "OWID COVID-19 cases"]
    assert summ.columns[-1] == "Best SGT vs best classical"


def test_all_figures_render(results_csv, tmp_path):
    out = tmp_path / "figs"
    rf.all_figures(results_csv, out, "best_test_mase")
    expected = {
        "losses_synthetic_best_test_mase.png",
        "losses_real_best_test_mase.png",
        "grid_synthetic_best_test_mase.png",
        "grid_real_best_test_mase.png",
        "lambda_effect_best_test_mase.png",
        "heads_best_test_mase.png",
        "tables_best_test_mase.md",
        "table_results_best_test_mase.csv",
        "table_summary_best_test_mase.csv",
    }
    assert expected <= {p.name for p in out.iterdir()}
    assert all((out / n).stat().st_size > 1000 for n in expected if n.endswith(".png"))


def test_heads_requires_head_dataset(results_csv, tmp_path):
    df = rf.load_results(results_csv)
    import typer

    with pytest.raises(typer.BadParameter):
        rf.figure_heads(
            df[df.dataset != "head-heavy-tailed"], "best_test_mase", tmp_path / "h.png"
        )


def test_sgt_label():
    assert rf.sgt_label(2.0, 2.5) == "SGT (p=2, q=2.5)"
    assert rf.sgt_label(1.5, 10.0, 0.1) == "SGT (p=1.5, q=10, λ=0.1)"
