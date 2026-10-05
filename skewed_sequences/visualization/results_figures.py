"""Publication figures and tables from the collected experiment results.

Everything here reads the CSV written by ``collect-results`` (one row per MLflow run)
and regenerates the revision's result figures and tables, so a reviewer-facing number
never has to be typed by hand:

* ``losses``   — mean test metric per loss with 95 % CIs (classical baselines + the
  pre-registered SGT anchors + the best-of-grid SGT), one panel per dataset.
* ``grid``     — the SGT ``p x q`` landscape (mean test metric) as annotated heat maps.
* ``lambda-effect`` — seed-paired effect of the skew parameter vs the symmetric twin.
* ``heads``    — the fixed-width multi-head study (mean per head count, all losses).
* ``tables``   — the main results table, the B15 summary table and the lambda table
  as Markdown + CSV.
* ``all``      — everything above.

Figures are sized for the journal's column widths (3.35 in single, 7.0 in double) at
300 dpi with >= 8 pt type, legends capitalised (reviewer A9 / B7 / B11). Lower is better
for every metric; MASE = 1 is the naive last-value forecast.
"""

from pathlib import Path

from loguru import logger
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
import typer  # noqa: E402

from skewed_sequences.config import FIGURES_DIR, REPORTS_DIR  # noqa: E402

app = typer.Typer(pretty_exceptions_show_locals=False)

DEFAULT_INPUT = REPORTS_DIR / "experiment_results.csv"
DEFAULT_OUTPUT = FIGURES_DIR / "results"
METRIC = "best_test_mase"
METRIC_LABEL = {
    "best_test_mase": "Test MASE",
    "best_test_rmse": "Test RMSE",
    "best_test_mae": "Test MAE",
    "best_test_smape": "Test sMAPE (%)",
}

# Dataset order and display names (synthetic first, then real).
SYNTHETIC = ["normal", "heavy-tailed", "normal-skewed", "heavy-tailed-skewed"]
REAL = ["covid-owid", "rvr-us-bed-occupancy", "rvr-us-influenza-cases"]
DATASET_LABEL = {
    "normal": "Gaussian",
    "heavy-tailed": "Heavy-tailed",
    "normal-skewed": "Gaussian, skewed",
    "heavy-tailed-skewed": "Heavy-tailed, skewed",
    "covid-owid": "OWID COVID-19 cases",
    "rvr-us-bed-occupancy": "RVR bed occupancy",
    "rvr-us-influenza-cases": "RVR influenza admissions",
    "head-heavy-tailed": "Heavy-tailed (head study)",
}
CLASSICAL = ["mse", "mae", "huber", "cauchy", "charbonnier", "tukey"]
CLASSICAL_LABEL = {
    "mse": "MSE",
    "mae": "MAE",
    "huber": "Huber",
    "cauchy": "Cauchy",
    "charbonnier": "Charbonnier",
    "tukey": "Tukey",
}
# Pre-registered SGT anchors (p, q, lambda) — fixed before the sweep, no selection.
ANCHORS = [(2.0, 2.5, 0.0), (1.5, 2.5, 0.0), (2.0, 20.0, 0.0), (1.0, 20.0, 0.0)]
HEAD_DATASET = "head-heavy-tailed"

# Validated two-hue categorical palette (dataviz check: light surface, CVD-safe).
COLOR_SGT = "#1A5EA6"
COLOR_CLASSICAL = "#B5893A"
COLOR_INK = "#16202B"
COLOR_MUTED = "#5C6B7A"
COLOR_RULE = "#D3DAE1"
COLUMN_IN, PAGE_IN = 3.35, 7.0


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": COLOR_MUTED,
            "axes.labelcolor": COLOR_INK,
            "xtick.color": COLOR_MUTED,
            "ytick.color": COLOR_MUTED,
            "text.color": COLOR_INK,
            "grid.color": COLOR_RULE,
            "grid.linewidth": 0.6,
            "axes.grid": True,
            "axes.grid.axis": "x",
            "axes.axisbelow": True,
        }
    )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def load_results(path: Path) -> pd.DataFrame:
    """Finished runs only; retired/invalid experiments and the head study are kept
    in the frame and selected per figure."""
    df = pd.read_csv(path)
    df = df[df["status"] == "FINISHED"].copy()
    df = df[~df["dataset"].str.contains("INVALID", na=False)]
    for col in ("sgt_loss_p", "sgt_loss_q", "sgt_loss_lambda"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def sgt_label(p: float, q: float, lam: float = 0.0) -> str:
    base = f"SGT (p={p:g}, q={q:g})"
    return base if lam == 0 else base[:-1] + f", λ={lam:g})"


def _sel_sgt(d: pd.DataFrame, p: float, q: float, lam: float = 0.0) -> pd.DataFrame:
    return d[
        (d.loss_type == "sgt")
        & (d.sgt_loss_p == p)
        & (d.sgt_loss_q == q)
        & (d.sgt_loss_lambda == lam)
    ]


def _ci95(s: pd.Series) -> float:
    n = s.count()
    return float(1.96 * s.std(ddof=1) / np.sqrt(n)) if n >= 2 else np.nan


def best_sgt_config(d: pd.DataFrame, metric: str):
    """(p, q, lambda) of the lowest-mean SGT config in ``d``, or None."""
    sgt = d[d.loss_type == "sgt"]
    if sgt.empty:
        return None
    means = sgt.groupby(["sgt_loss_p", "sgt_loss_q", "sgt_loss_lambda"])[metric].mean()
    return tuple(float(x) for x in means.idxmin())


def paired_diff(a: pd.DataFrame, b: pd.DataFrame, metric: str):
    """Seed-paired mean difference a-b, its 95 % CI half-width, Wilcoxon p, n."""
    x = a.groupby("random_state")[metric].mean()
    y = b.groupby("random_state")[metric].mean()
    idx = x.index.intersection(y.index)
    if len(idx) < 2:
        return np.nan, np.nan, np.nan, len(idx)
    diff = (x.loc[idx] - y.loc[idx]).to_numpy()
    p = np.nan if np.allclose(diff, 0) else float(wilcoxon(diff).pvalue)
    return float(diff.mean()), float(1.96 * diff.std(ddof=1) / np.sqrt(len(diff))), p, len(diff)


def loss_rows(d: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Rows for the loss-comparison figure/table: classical, anchors, best-of-grid."""
    rows = []
    for loss in CLASSICAL:
        s = d[d.loss_type == loss][metric]
        rows.append(
            dict(
                key=loss,
                label=CLASSICAL_LABEL[loss],
                family="classical",
                mean=s.mean(),
                ci=_ci95(s),
                n=int(s.count()),
            )
        )
    for p, q, lam in ANCHORS:
        s = _sel_sgt(d, p, q, lam)[metric]
        if s.empty:
            continue
        rows.append(
            dict(
                key=f"sgt_{p}_{q}_{lam}",
                label=sgt_label(p, q, lam),
                family="sgt",
                mean=s.mean(),
                ci=_ci95(s),
                n=int(s.count()),
            )
        )
    best = best_sgt_config(d, metric)
    if best is not None:
        s = _sel_sgt(d, *best)[metric]
        rows.append(
            dict(
                key="sgt_best",
                label=f"SGT, best of grid {sgt_label(*best)[4:]}",
                family="sgt",
                mean=s.mean(),
                ci=_ci95(s),
                n=int(s.count()),
                config=sgt_label(*best)[4:],
            )
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _loss_panel(ax, rows: pd.DataFrame, metric: str, title: str) -> None:
    rows = rows.iloc[::-1].reset_index(drop=True)  # first row at the top
    finite = rows[np.isfinite(rows["mean"])]
    core = finite[finite["key"] != "tukey"]
    upper = (core["mean"] + core["ci"]).max()
    lower = (core["mean"] - core["ci"]).min()
    span = max(upper - lower, 1e-6)
    xmax = upper + 0.18 * span
    xmin = max(0.0, lower - 0.12 * span)
    for i, r in rows.iterrows():
        color = COLOR_SGT if r["family"] == "sgt" else COLOR_CLASSICAL
        if not np.isfinite(r["mean"]):
            continue
        if r["mean"] > xmax:  # off-scale (Tukey collapse): clip and annotate
            ax.plot([xmax - 0.1 * span, xmax], [i, i], color=color, lw=1.6, solid_capstyle="butt")
            ax.annotate(
                f"{r['mean']:.1f} →",
                (xmax, i),
                xytext=(-2, 0),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=6.5,
                color=COLOR_MUTED,
            )
            continue
        ax.plot(
            [r["mean"] - r["ci"], r["mean"] + r["ci"]],
            [i, i],
            color=color,
            lw=1.4,
            solid_capstyle="round",
        )
        ax.plot(r["mean"], i, "o", ms=4.2, color=color, mec="white", mew=0.7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows["label"])
    for tick, fam in zip(ax.get_yticklabels(), rows["family"]):
        tick.set_color(COLOR_INK if fam == "sgt" else COLOR_MUTED)
    ax.set_xlim(xmin, xmax)
    ax.set_title(title, loc="left")
    ax.set_xlabel(METRIC_LABEL.get(metric, metric))
    ax.grid(axis="y", visible=False)
    if metric == "best_test_mase" and xmin < 1.0 < xmax:
        ax.axvline(1.0, color=COLOR_MUTED, lw=0.8, ls=(0, (3, 2)))
        ax.annotate(
            "naive",
            (1.0, len(rows) - 0.6),
            xytext=(2, 0),
            textcoords="offset points",
            fontsize=6.5,
            color=COLOR_MUTED,
        )


def _legend(fig, loc=(0.5, 0.0)):
    handles = [
        plt.Line2D([], [], marker="o", color=COLOR_SGT, lw=1.4, ms=4, label="SGT loss"),
        plt.Line2D(
            [], [], marker="o", color=COLOR_CLASSICAL, lw=1.4, ms=4, label="Classical losses"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=loc,
        title="Mean over 10 seed-paired runs, whiskers = 95 % CI",
        title_fontsize=7,
    )


def figure_losses(
    df: pd.DataFrame, datasets: list, metric: str, out: Path, ncols: int, title: str | None = None
) -> Path:
    present = [d for d in datasets if (df.dataset == d).any()]
    if not present:
        raise typer.BadParameter(f"none of {datasets} present in the results")
    nrows = int(np.ceil(len(present) / ncols))
    width = PAGE_IN if ncols > 1 else COLUMN_IN
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(width, 2.45 * nrows + 0.45), squeeze=False, sharey="row"
    )
    for ax, ds in zip(axes.flat, present):
        _loss_panel(ax, loss_rows(df[df.dataset == ds], metric), metric, DATASET_LABEL.get(ds, ds))
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    for ax in list(axes.flat)[len(present) :]:
        ax.axis("off")
    bottom = 0.14 if nrows == 1 else 0.07  # room for the legend under one row of panels
    fig.tight_layout(h_pad=1.4, w_pad=1.2, rect=(0, bottom, 1, 1 if title is None else 0.96))
    if title:
        fig.suptitle(title, fontsize=9)
    _legend(fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    logger.success(f"saved {out}")
    return out


def figure_grid(df: pd.DataFrame, datasets: list, metric: str, out: Path, ncols: int) -> Path:
    present = [d for d in datasets if (df.dataset == d).any()]
    nrows = int(np.ceil(len(present) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(PAGE_IN, 2.9 * nrows), squeeze=False)
    for ax, ds in zip(axes.flat, present):
        d = df[(df.dataset == ds) & (df.loss_type == "sgt") & (df.sgt_loss_lambda == 0)]
        piv = d.pivot_table(
            index="sgt_loss_q", columns="sgt_loss_p", values=metric, aggfunc="mean"
        )
        piv = piv.sort_index(ascending=False)
        vals = piv.to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        vmin, vmax = np.nanpercentile(finite, 2), np.nanpercentile(finite, 98)
        im = ax.imshow(vals, cmap="Blues_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels([f"{c:g}" for c in piv.columns])
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels([f"{r:g}" for r in piv.index])
        ax.set_xlabel("p")
        ax.set_ylabel("q")
        ax.grid(False)
        ax.tick_params(length=0)
        best = np.unravel_index(np.nanargmin(vals), vals.shape)
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                if not np.isfinite(v):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=6.5, color=COLOR_MUTED)
                    continue
                light = (v - vmin) / max(vmax - vmin, 1e-9) > 0.45
                ax.text(
                    j,
                    i,
                    f"{v:.3f}" if metric != "best_test_smape" else f"{v:.1f}",
                    ha="center",
                    va="center",
                    fontsize=6.3,
                    color=COLOR_INK if light else "white",
                    fontweight="bold" if (i, j) == best else "normal",
                )
        cls = df[(df.dataset == ds) & (df.loss_type != "sgt")].groupby("loss_type")[metric].mean()
        fmt = "{:.2f}" if metric != "best_test_smape" else "{:.1f}"
        pairs = [
            f"{CLASSICAL_LABEL[k]} {fmt.format(cls[k])}"
            for k in ("mse", "mae", "huber", "cauchy")
            if k in cls
        ]
        ref = "  ".join(pairs[:2]) + "\n" + "  ".join(pairs[2:])
        ax.set_title(f"{DATASET_LABEL.get(ds, ds)}\n{ref}", loc="left", fontsize=7)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6.5)
        cb.set_label(METRIC_LABEL.get(metric, metric), fontsize=6.5)
    for ax in list(axes.flat)[len(present) :]:
        ax.axis("off")
    fig.tight_layout(h_pad=1.6, w_pad=1.6)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    logger.success(f"saved {out}")
    return out


def lambda_effect_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Seed-paired effect of every lambda != 0 vs its symmetric twin, per dataset/(p,q)."""
    rows = []
    sgt = df[(df.loss_type == "sgt") & (df.dataset != HEAD_DATASET)]
    for (ds, p, q), g in sgt.groupby(["dataset", "sgt_loss_p", "sgt_loss_q"]):
        sym = g[g.sgt_loss_lambda == 0]
        if sym.empty:
            continue
        for lam, gl in g[g.sgt_loss_lambda != 0].groupby("sgt_loss_lambda"):
            diff, ci, pval, n = paired_diff(gl, sym, metric)
            rows.append(
                dict(
                    dataset=ds,
                    p=p,
                    q=q,
                    lam=lam,
                    n=n,
                    mean_symmetric=sym[metric].mean(),
                    mean_skewed=gl[metric].mean(),
                    diff=diff,
                    ci95=ci,
                    wilcoxon_p=pval,
                )
            )
    return pd.DataFrame(rows)


def figure_lambda(df: pd.DataFrame, metric: str, out: Path) -> Path:
    t = lambda_effect_table(df, metric)
    if t.empty:
        raise typer.BadParameter("no skewed (lambda != 0) SGT runs in the results")
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_IN, 2.6))
    # Left: fine grid on the skewed heavy-tailed data (all (p, q) with a lambda sweep).
    ax = axes[0]
    fine = t[(t.dataset == "heavy-tailed-skewed")]
    styles = {
        (2.0, 2.5): ("-", "o"),
        (2.0, 10.0): ("-", "s"),
        (1.5, 2.5): ("--", "o"),
        (1.5, 10.0): ("--", "s"),
    }
    for (p, q), g in fine.groupby(["p", "q"]):
        g = g.sort_values("lam")
        ls, mk = styles.get((p, q), ("-", "o"))
        ax.errorbar(
            g["lam"],
            g["diff"],
            yerr=g["ci95"],
            fmt=mk + ls,
            ms=3.5,
            lw=1.1,
            capsize=2,
            color=COLOR_SGT if p == 2.0 else COLOR_CLASSICAL,
            label=f"p={p:g}, q={q:g}",
        )
    ax.axhline(0, color=COLOR_MUTED, lw=0.8)
    ax.set_xlabel("Skew parameter λ")
    ax.set_ylabel(f"Δ {METRIC_LABEL.get(metric, metric)} vs λ = 0")
    ax.set_title("Heavy-tailed, skewed data", loc="left")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    # Right: the main-grid lambda points on every synthetic dataset at (2, 10).
    ax = axes[1]
    main = t[(t.p == 2.0) & (t.q == 10.0) & (t.dataset.isin(SYNTHETIC)) & (t.lam.isin([0.5, 0.9]))]
    lams = sorted(main.lam.unique())
    present_ds = [d for d in SYNTHETIC if (main.dataset == d).any()]
    width = 0.8 / max(len(present_ds), 1)
    for k, ds in enumerate(present_ds):
        g = main[main.dataset == ds].sort_values("lam")
        x = np.array([lams.index(v) for v in g["lam"]]) + (k - (len(present_ds) - 1) / 2) * width
        ax.bar(
            x,
            g["diff"],
            width=width,
            yerr=g["ci95"],
            capsize=1.5,
            color=COLOR_SGT if "skewed" in ds else COLOR_CLASSICAL,
            alpha=0.9 - 0.3 * (k % 2),
            label=DATASET_LABEL[ds],
            error_kw=dict(lw=0.8),
        )
    ax.set_xticks(range(len(lams)))
    ax.set_xticklabels([f"λ = {v:g}" for v in lams])
    ax.axhline(0, color=COLOR_MUTED, lw=0.8)
    ax.set_ylabel(f"Δ {METRIC_LABEL.get(metric, metric)} vs λ = 0")
    ax.set_title("Main-grid skew values, all synthetic data (p = 2, q = 10)", loc="left")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout(w_pad=2.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    logger.success(f"saved {out}")
    return out


def figure_heads(df: pd.DataFrame, metric: str, out: Path) -> Path:
    h = df[df.dataset == HEAD_DATASET].copy()
    if h.empty:
        raise typer.BadParameter("no head-study runs (dataset 'head-heavy-tailed') in the results")
    h["cfg"] = h.apply(
        lambda r: CLASSICAL_LABEL.get(r.loss_type, None)
        or sgt_label(r.sgt_loss_p, r.sgt_loss_q, r.sgt_loss_lambda),
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.5))
    heads = sorted(h.num_heads.unique())
    for cfg, g in h.groupby("cfg"):
        m = g.groupby("num_heads")[metric].mean().reindex(heads)
        ax.plot(
            heads,
            m.values,
            color=COLOR_SGT if cfg.startswith("SGT") else COLOR_CLASSICAL,
            lw=0.8,
            alpha=0.45,
        )
    agg = h.groupby("num_heads")[metric].agg(["mean", _ci95]).reindex(heads)
    ax.errorbar(
        heads,
        agg["mean"],
        yerr=agg["_ci95"],
        fmt="o-",
        color=COLOR_INK,
        lw=1.6,
        ms=4.5,
        capsize=2.5,
        label="All losses (mean ± 95 % CI)",
        zorder=5,
    )
    single = h[h.num_heads == heads[0]]
    for nh in heads[1:]:
        diff, ci, pval, n = paired_diff(h[h.num_heads == nh], single, metric)
        ax.annotate(
            f"p = {pval:.2f}" if np.isfinite(pval) else "",
            (nh, agg.loc[nh, "mean"]),
            xytext=(7, 7),
            textcoords="offset points",
            ha="left",
            fontsize=6.5,
            color=COLOR_MUTED,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(heads)
    ax.set_xticklabels([str(x) for x in heads])
    ax.minorticks_off()
    ax.set_xlabel("Attention heads (embedding width fixed at 256)")
    ax.set_ylabel(METRIC_LABEL.get(metric, metric))
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    handles = [
        plt.Line2D(
            [], [], color=COLOR_INK, marker="o", lw=1.6, ms=4, label="All losses, mean ± 95 % CI"
        ),
        plt.Line2D([], [], color=COLOR_SGT, lw=0.8, alpha=0.6, label="SGT configurations"),
        plt.Line2D([], [], color=COLOR_CLASSICAL, lw=0.8, alpha=0.6, label="Classical losses"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    logger.success(f"saved {out}")
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def results_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Main results table: rows = losses (classical + anchors + best SGT), cols = datasets."""
    frames = []
    for ds in SYNTHETIC + REAL:
        d = df[df.dataset == ds]
        if d.empty:
            continue
        r = loss_rows(d, metric)
        r["dataset"] = DATASET_LABEL.get(ds, ds)
        frames.append(r)
    t = pd.concat(frames, ignore_index=True)
    t["cell"] = t.apply(
        lambda r: (
            f"{r['mean']:.3f} ± {r['ci']:.3f}"
            if r["mean"] < 10
            else f"{r['mean']:.1f} ± {r['ci']:.1f}"
        ),
        axis=1,
    )
    is_best = t["key"] == "sgt_best"
    t.loc[is_best, "cell"] = t.loc[is_best, "cell"] + " " + t.loc[is_best, "config"]
    t.loc[is_best, "label"] = "SGT, best of grid (config)"
    wide = t.pivot_table(index="label", columns="dataset", values="cell", aggfunc="first")
    order = [CLASSICAL_LABEL[c] for c in CLASSICAL] + [sgt_label(*a) for a in ANCHORS]
    order += ["SGT, best of grid (config)"]
    wide = wide.reindex([o for o in order if o in wide.index])
    cols = [DATASET_LABEL[d] for d in SYNTHETIC + REAL if DATASET_LABEL[d] in wide.columns]
    return wide[cols]


def summary_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Reviewer-B15 table: per dataset, the best loss overall, the best classical, the
    pre-registered Cauchy-like anchor vs MSE and vs the best classical (paired p)."""
    rows = []
    for ds in SYNTHETIC + REAL:
        d = df[df.dataset == ds]
        if d.empty:
            continue
        cls = d[d.loss_type != "sgt"].groupby("loss_type")[metric].mean()
        best_cls = cls.idxmin()
        cfg_means = d.groupby(
            ["loss_type", "sgt_loss_p", "sgt_loss_q", "sgt_loss_lambda"], dropna=False
        )[metric].mean()
        best_any = cfg_means.idxmin()
        best_any_label = CLASSICAL_LABEL.get(best_any[0]) or sgt_label(
            best_any[1], best_any[2], best_any[3]
        )
        anchor = _sel_sgt(d, 2.0, 2.5, 0.0)
        d_mse, _, p_mse, _ = paired_diff(anchor, d[d.loss_type == "mse"], metric)
        d_cls, _, p_cls, _ = paired_diff(anchor, d[d.loss_type == best_cls], metric)
        best_sgt = best_sgt_config(d, metric)
        d_b, _, p_b, _ = paired_diff(_sel_sgt(d, *best_sgt), d[d.loss_type == best_cls], metric)
        rows.append(
            {
                "Dataset": DATASET_LABEL.get(ds, ds),
                "Best overall": best_any_label,
                "Best classical": CLASSICAL_LABEL[best_cls],
                "SGT (2, 2.5) vs MSE": f"{d_mse:+.3f} (p = {p_mse:.3f})",
                "SGT (2, 2.5) vs best classical": f"{d_cls:+.3f} (p = {p_cls:.3f})",
                "Best SGT vs best classical": f"{d_b:+.3f} (p = {p_b:.3f})",
            }
        )
    return pd.DataFrame(rows)


def _to_markdown(t: pd.DataFrame, index: bool = True) -> str:
    cols = ([t.index.name or ""] if index else []) + [str(c) for c in t.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in t.iterrows():
        vals = ([str(idx)] if index else []) + ["" if pd.isna(v) else str(v) for v in row.tolist()]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def losses(
    input_path: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT, metric: str = METRIC
):
    """Per-loss mean ± 95 % CI, one panel per dataset (synthetic 2x2, real 1x3)."""
    _style()
    df = load_results(input_path)
    figure_losses(df, SYNTHETIC, metric, output_dir / f"losses_synthetic_{metric}.png", ncols=2)
    figure_losses(df, REAL, metric, output_dir / f"losses_real_{metric}.png", ncols=3)


@app.command()
def grid(
    input_path: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT, metric: str = METRIC
):
    """SGT p x q landscape heat maps (symmetric configs)."""
    _style()
    df = load_results(input_path)
    figure_grid(df, SYNTHETIC, metric, output_dir / f"grid_synthetic_{metric}.png", ncols=2)
    figure_grid(df, REAL, metric, output_dir / f"grid_real_{metric}.png", ncols=3)


@app.command("lambda-effect")
def lambda_effect(
    input_path: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT, metric: str = METRIC
):
    """Seed-paired effect of the skew parameter."""
    _style()
    df = load_results(input_path)
    figure_lambda(df, metric, output_dir / f"lambda_effect_{metric}.png")
    lambda_effect_table(df, metric).to_csv(output_dir / f"lambda_effect_{metric}.csv", index=False)


@app.command()
def heads(
    input_path: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT, metric: str = METRIC
):
    """Fixed-width multi-head study."""
    _style()
    df = load_results(input_path)
    figure_heads(df, metric, output_dir / f"heads_{metric}.png")


@app.command()
def tables(
    input_path: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT, metric: str = METRIC
):
    """Main results table, B15 summary table and lambda table (Markdown + CSV)."""
    df = load_results(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    main = results_table(df, metric)
    main.to_csv(output_dir / f"table_results_{metric}.csv")
    summ = summary_table(df, metric)
    summ.to_csv(output_dir / f"table_summary_{metric}.csv", index=False)
    lam = lambda_effect_table(df, metric)
    lam.round(4).to_csv(output_dir / f"table_lambda_{metric}.csv", index=False)
    md = (
        f"## Results ({METRIC_LABEL.get(metric, metric)}, mean ± 95 % CI over 10 seed-paired runs)\n\n"
        + _to_markdown(main.rename_axis("Loss"))
        + "\n## Summary (reviewer B15)\n\n"
        + _to_markdown(summ, index=False)
        + "\n## Skew parameter (seed-paired vs λ = 0)\n\n"
        + _to_markdown(lam.round(4), index=False)
    )
    (output_dir / f"tables_{metric}.md").write_text(md)
    typer.echo(md)
    logger.success(f"tables written to {output_dir}")


@app.command("all")
def all_figures(
    input_path: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT, metric: str = METRIC
):
    """Every figure and table."""
    losses(input_path, output_dir, metric)
    grid(input_path, output_dir, metric)
    lambda_effect(input_path, output_dir, metric)
    heads(input_path, output_dir, metric)
    tables(input_path, output_dir, metric)


if __name__ == "__main__":
    app()
