"""Tests for skewed_sequences.visualization.results_figures (publication figures/tables)."""

import numpy as np
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
        rows += _rows(ds, "sgt", 2.0, 2.5, 0.5, 0.35 + shift)
        rows += _rows(ds, "sgt", 2.0, 2.5, 0.9, 0.45 + shift)
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


@pytest.fixture()
def data_dir(tmp_path):
    """Tiny (N, T, 1) arrays for two datasets: smooth Gaussian and heavy-tailed."""
    rng = np.random.default_rng(0)
    d = tmp_path / "data"
    d.mkdir()
    smooth = np.cumsum(rng.normal(size=(6, 300)), axis=1)
    smooth = (smooth - smooth.mean(axis=1, keepdims=True)) / smooth.std(axis=1, keepdims=True)
    heavy = np.cumsum(rng.standard_t(2.0, size=(6, 300)), axis=1)
    heavy = (heavy - heavy.mean(axis=1, keepdims=True)) / heavy.std(axis=1, keepdims=True)
    np.save(d / "normal.npy", smooth[:, :, None])
    np.save(d / "heavy-tailed.npy", heavy[:, :, None])
    return d


@pytest.fixture()
def curves_csv(tmp_path):
    rows = []
    for ds in ("covid-owid", "rvr-us-influenza-cases"):
        for loss, level in (("tukey", 2.0), ("huber", 0.1), ("sgt", 0.1)):
            for run in range(2):
                for step in range(1, 6):
                    rows.append(
                        dict(
                            dataset=ds,
                            run_uuid=f"{ds}-{loss}-{run}",
                            loss_type=loss,
                            key="train_mae",
                            step=step,
                            value=level / step,
                        )
                    )
    path = tmp_path / "curves.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_results_filters(results_csv):
    df = rf.load_results(results_csv)
    assert (df.status == "FINISHED").all()
    assert not df.dataset.str.contains("INVALID").any()


def test_ci_is_a_reproducible_bootstrap_interval():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    lo, hi = rf._ci(s)
    assert lo < s.mean() < hi
    assert (lo, hi) == rf._ci(s)  # fixed resampling seed
    assert rf._ci(pd.Series([2.0, 2.0])) == (2.0, 2.0)


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
    r = rf.paired_diff(d[d.loss_type == "huber"], d[d.loss_type == "mse"], "best_test_mase")
    assert r.mean == pytest.approx(-0.05)
    tie = rf.paired_diff(d[d.loss_type == "mse"], d[d.loss_type == "mse"], "best_test_mase")
    assert np.isnan(tie.p) and np.isnan(tie.wilcoxon_p)  # equal vectors: both undefined
    assert r.n == N_RUNS
    # constant difference -> the bootstrap interval is a point
    assert r.ci_low == pytest.approx(r.mean) and r.ci_high == pytest.approx(r.mean)
    # all four differences on one side -> exact sign-flip p at its floor, Wilcoxon 2/16
    assert r.p == pytest.approx(2 / 2**N_RUNS)
    assert r.wilcoxon_p == pytest.approx(0.125)


def test_anchor_effects_relative(results_csv):
    df = rf.load_results(results_csv)
    t = rf.anchor_effects(df, "best_test_mase")
    assert set(t.reference) == {"mse", "best"}
    r = t[(t.dataset == "heavy-tailed") & (t.reference == "mse") & (t.p == 2.0)].iloc[0]
    # SGT(2, 2.5) mean 0.745 vs MSE 0.80 -> about -6.9 %
    assert r.rel == pytest.approx(100 * r["diff"] / 0.815, rel=1e-6)
    assert r.rel < 0 and r.reference_loss == "mse"
    assert {"perm_p", "wilcoxon_p", "rel_ci_low", "rel_ci_high"} <= set(t.columns)
    assert r.rel_ci_low <= r.rel <= r.rel_ci_high


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
    # cells carry the mean and the bootstrap interval, not a +- half-width
    import re

    assert re.fullmatch(r"\d\.\d{3} \[\d\.\d{3}, \d\.\d{3}\]", main.loc["MSE", "Gaussian"])


def test_holm_table_adjusts_within_dataset(results_csv):
    df = rf.load_results(results_csv)
    hol = rf.holm_table(df, "best_test_mase")
    assert set(hol.dataset) == {"normal", "heavy-tailed", "covid-owid"}
    assert (hol.family_size == 12).all()  # two anchors x six baselines
    assert (hol.holm_p >= hol.perm_p - 1e-12).all() and (hol.holm_p <= 1).all()
    # every anchor-vs-baseline difference in the fixture is constant across seeds, so the
    # raw p is the exact floor 2 / 2**n (n = 3 where the fixture drops the RUNNING run) and
    # the Holm value is at most 12 times it
    assert hol.perm_p.to_numpy() == pytest.approx(2.0 / 2.0 ** hol.n.to_numpy())
    assert (hol.holm_p <= 12 * hol.perm_p + 1e-12).all()
    # no SGT runs at all -> an empty frame, and the tables command must not choke on it
    assert rf.holm_table(df[df.loss_type != "sgt"], "best_test_mase").empty


def test_all_figures_render(results_csv, data_dir, curves_csv, tmp_path):
    out = tmp_path / "figs"
    rf.all_figures(results_csv, out, "best_test_mase", data_dir=data_dir, curves_csv=curves_csv)
    expected = {
        "summary_best_test_mase.png",
        "summary_best_test_mase.csv",
        "losses_synthetic_best_test_mase.png",
        "losses_real_best_test_mase.png",
        "grid_synthetic_best_test_mase.png",
        "grid_real_best_test_mase.png",
        "lambda_effect_best_test_mase.png",
        "heads_best_test_mase.png",
        "loss_family.png",
        "datasets.png",
        "tukey_curves.png",
        "tables_best_test_mase.md",
        "table_results_best_test_mase.csv",
        "table_summary_best_test_mase.csv",
        "table_holm_best_test_mase.csv",
        "table_lambda_best_test_mase.csv",
    }
    names = {p.name for p in out.iterdir()}
    assert expected <= names
    assert all((out / n).stat().st_size > 1000 for n in expected if n.endswith(".png"))
    # every PNG has a vector twin
    assert all((out / n).with_suffix(".pdf").exists() for n in expected if n.endswith(".png"))


def test_heads_requires_head_dataset(results_csv, tmp_path):
    df = rf.load_results(results_csv)
    import typer

    with pytest.raises(typer.BadParameter):
        rf.figure_heads(
            df[df.dataset != "head-heavy-tailed"], "best_test_mase", tmp_path / "h.png"
        )


def test_datasets_requires_arrays(tmp_path):
    import typer

    with pytest.raises(typer.BadParameter):
        rf.figure_datasets(tmp_path, tmp_path / "d.png")


def test_dataset_card_stats_keys(data_dir):
    arr = np.load(data_dir / "heavy-tailed.npy")
    st = rf.dataset_card_stats(arr)
    assert {"s_hat", "kurtosis", "hill", "lam", "q", "mu", "increments"} <= set(st)
    assert st["s_hat"] > 0 and -1 < st["lam"] < 1 and st["q"] > 0
    seq = rf._typical_sequence(arr)
    assert seq.shape == (300,)


def test_baseline_losses_shapes():
    e = np.linspace(-5, 5, 11)
    L = rf._baseline_losses(e, s=1.0)
    assert set(L) == {"MSE", "MAE", "Huber", "Cauchy", "Tukey", "Charbonnier"}
    assert all(v.shape == e.shape for v in L.values())
    # Tukey is bounded by c^2/6 and the others are zero at e = 0
    assert L["Tukey"].max() == pytest.approx(4.685**2 / 6)
    assert all(v[5] == pytest.approx(0.0) for v in L.values())


def test_sgt_label():
    assert rf.sgt_label(2.0, 2.5) == "SGT (p=2, q=2.5)"
    assert rf.sgt_label(1.5, 10.0, 0.1) == "SGT (p=1.5, q=10, λ=0.1)"
