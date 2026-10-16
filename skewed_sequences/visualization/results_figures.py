"""Publication figures and tables from the collected experiment results.

Everything here reads the CSV written by ``collect-results`` (one row per MLflow run)
and regenerates the paper's result figures and tables, so a reader-facing number never
has to be typed by hand:

* ``summary``  — the headline figure: seed-paired change in test MASE of the two
  tail-tolerant SGT anchors against MSE and against the best classical loss, per dataset.
* ``losses``   — mean test metric per loss with 95 % bootstrap CIs (classical baselines + the
  SGT anchors fixed before the sweep + the best-of-grid SGT), one panel per dataset.
* ``grid``     — the SGT ``p x q`` landscape (mean test metric) as annotated heat maps.
* ``lambda-effect`` — seed-paired effect of the skew parameter vs the symmetric twin.
* ``heads``    — the fixed-width multi-head study (mean per head count, all losses).
* ``loss-family`` — the SGT loss and gradient shapes next to the robust baselines
  (needs no results; pure functions of the loss definitions).
* ``datasets`` — one card per dataset: an example sequence and the distribution of its
  one-step increments with the fitted SGT density (needs the ``.npy`` arrays).
* ``tukey-curves`` — per-epoch training error on the sparse series (needs the
  per-epoch metrics exported from MLflow).
* ``tables``   — the main results table, the summary table, the lambda table and the
  Holm-adjusted anchor-versus-baseline table as Markdown + CSV.
* ``all``      — everything above for which the inputs exist.

Figures are sized for a 7.0 in text width (3.35 in for a single column) and written as
PDF (vector, fonts embedded) and 300 dpi PNG. Every interval is a BCa bootstrap interval
of the mean (10,000 resamples, fixed seed) and every headline p-value an exact sign-flip
permutation test on the seed-paired differences (``skewed_sequences.stats``); the paired
Wilcoxon signed-rank p is kept next to it. Lower is better for every metric; MASE = 1 is
the naive last-value forecast.
"""

from pathlib import Path
from typing import NamedTuple

from loguru import logger
import matplotlib

matplotlib.use("Agg")
from matplotlib import gridspec  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kurtosis as _kurtosis  # noqa: E402
from scipy.stats import norm as _norm  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
import typer  # noqa: E402

from skewed_sequences.config import FIGURES_DIR, REPORTS_DIR  # noqa: E402
from skewed_sequences.stats import bootstrap_ci, holm, signflip_p  # noqa: E402

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
# SGT anchors (p, q, lambda) fixed before the sweep — no selection bias.
ANCHORS = [(2.0, 2.5, 0.0), (1.5, 2.5, 0.0), (2.0, 20.0, 0.0), (1.0, 20.0, 0.0)]
HEAD_DATASET = "head-heavy-tailed"

# Two-hue categorical palette (validated: light surface, CVD-safe) + neutrals.
COLOR_SGT = "#1A5EA6"
COLOR_SGT_LIGHT = "#7FA7D1"
COLOR_CLASSICAL = "#B5893A"
COLOR_CLASSICAL_LIGHT = "#D9BE86"
COLOR_INK = "#16202B"
COLOR_MUTED = "#5C6B7A"
COLOR_FAINT = "#9AA6B2"
COLOR_RULE = "#E1E6EB"
COLOR_BAND = "#EEF2F6"
CMAP_BLUE = LinearSegmentedColormap.from_list(
    "paper_blues", ["#0B3C74", "#1A5EA6", "#5E8FC6", "#A9C4E2", "#E3ECF6"]
)
COLUMN_IN, PAGE_IN = 3.35, 7.0
FONT_STACK = ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": FONT_STACK,
            "font.size": 8,
            "axes.titlesize": 8.8,
            "axes.titleweight": "medium",
            "axes.labelsize": 8,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": COLOR_FAINT,
            "axes.linewidth": 0.6,
            "axes.labelcolor": COLOR_INK,
            "xtick.color": COLOR_MUTED,
            "ytick.color": COLOR_MUTED,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "text.color": COLOR_INK,
            "grid.color": COLOR_RULE,
            "grid.linewidth": 0.5,
            "axes.grid": True,
            "axes.grid.axis": "x",
            "axes.axisbelow": True,
            "legend.frameon": False,
            "lines.solid_capstyle": "round",
        }
    )


def _save(fig, out: Path) -> Path:
    """Write ``out`` (PNG) and the same figure as PDF next to it."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    if out.suffix.lower() == ".png":
        fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    logger.success(f"saved {out}")
    return out


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


def _ci(s) -> tuple[float, float]:
    """BCa bootstrap 95 % interval (low, high) of the mean over the replicates."""
    return bootstrap_ci(np.asarray(s, dtype=float))


class PairedDiff(NamedTuple):
    mean: float
    ci_low: float
    ci_high: float
    p: float  # exact two-sided sign-flip permutation p
    wilcoxon_p: float
    n: int


def best_sgt_config(d: pd.DataFrame, metric: str):
    """(p, q, lambda) of the lowest-mean SGT config in ``d``, or None."""
    sgt = d[d.loss_type == "sgt"]
    if sgt.empty:
        return None
    means = sgt.groupby(["sgt_loss_p", "sgt_loss_q", "sgt_loss_lambda"])[metric].mean()
    return tuple(float(x) for x in means.idxmin())


def paired_diff(a: pd.DataFrame, b: pd.DataFrame, metric: str) -> PairedDiff:
    """Seed-paired mean difference a-b with its bootstrap 95 % interval, the exact sign-flip
    permutation p, the Wilcoxon signed-rank p and the number of pairs."""
    x = a.groupby("random_state")[metric].mean()
    y = b.groupby("random_state")[metric].mean()
    idx = x.index.intersection(y.index)
    if len(idx) < 2:
        return PairedDiff(np.nan, np.nan, np.nan, np.nan, np.nan, len(idx))
    diff = (x.loc[idx] - y.loc[idx]).to_numpy()
    tie = np.allclose(diff, 0)  # equal up to rounding: both tests undefined
    wp = np.nan if tie else float(wilcoxon(diff).pvalue)
    pp = np.nan if tie else signflip_p(diff)
    low, high = bootstrap_ci(diff)
    return PairedDiff(float(diff.mean()), low, high, pp, wp, len(diff))


def loss_rows(d: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Rows for the loss-comparison figure/table: classical, anchors, best-of-grid."""
    rows = []
    for loss in CLASSICAL:
        s = d[d.loss_type == loss][metric]
        lo, hi = _ci(s)
        rows.append(
            dict(
                key=loss,
                label=CLASSICAL_LABEL[loss],
                family="classical",
                mean=s.mean(),
                ci_low=lo,
                ci_high=hi,
                n=int(s.count()),
            )
        )
    for p, q, lam in ANCHORS:
        s = _sel_sgt(d, p, q, lam)[metric]
        if s.empty:
            continue
        lo, hi = _ci(s)
        rows.append(
            dict(
                key=f"sgt_{p}_{q}_{lam}",
                label=sgt_label(p, q, lam),
                family="sgt",
                mean=s.mean(),
                ci_low=lo,
                ci_high=hi,
                n=int(s.count()),
            )
        )
    best = best_sgt_config(d, metric)
    if best is not None:
        s = _sel_sgt(d, *best)[metric]
        lo, hi = _ci(s)
        rows.append(
            dict(
                key="sgt_best",
                label=f"SGT, best of grid {sgt_label(*best)[4:]}",
                family="sgt",
                mean=s.mean(),
                ci_low=lo,
                ci_high=hi,
                n=int(s.count()),
                config=sgt_label(*best)[4:],
            )
        )
    return pd.DataFrame(rows)


def anchor_effects(df: pd.DataFrame, metric: str, anchors=None) -> pd.DataFrame:
    """Seed-paired relative change (%) of each anchor vs MSE and vs the best classical
    loss on every dataset: the numbers behind the headline figure."""
    anchors = anchors or ANCHORS[:2]
    rows = []
    for ds in SYNTHETIC + REAL:
        d = df[df.dataset == ds]
        if d.empty:
            continue
        cls = d[d.loss_type != "sgt"].groupby("loss_type")[metric].mean()
        best_cls = cls.idxmin()
        for p, q, lam in anchors:
            a = _sel_sgt(d, p, q, lam)
            if a.empty:
                continue
            for ref_key, ref_loss in (("mse", "mse"), ("best", best_cls)):
                b = d[d.loss_type == ref_loss]
                r = paired_diff(a, b, metric)
                scale = b[metric].mean()
                rows.append(
                    dict(
                        dataset=ds,
                        anchor=sgt_label(p, q, lam),
                        p=p,
                        q=q,
                        reference=ref_key,
                        reference_loss=ref_loss,
                        diff=r.mean,
                        ci_low=r.ci_low,
                        ci_high=r.ci_high,
                        rel=100 * r.mean / scale,
                        rel_ci_low=100 * r.ci_low / scale,
                        rel_ci_high=100 * r.ci_high / scale,
                        perm_p=r.p,
                        wilcoxon_p=r.wilcoxon_p,
                        n=r.n,
                    )
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures: results
# ---------------------------------------------------------------------------


def figure_summary(df: pd.DataFrame, metric: str, out: Path) -> Path:
    """Headline figure: relative change of the tail-tolerant anchors vs MSE (left) and
    vs the best classical loss (right), per dataset, with bootstrap 95 % CIs. Filled
    markers are significant at p < 0.05 (exact sign-flip permutation test over the seed
    pairs)."""
    t = anchor_effects(df, metric)
    if t.empty:
        raise typer.BadParameter("no anchor runs in the results")
    datasets = [d for d in SYNTHETIC + REAL if (t.dataset == d).any()]
    ypos = {}
    y = 0.0
    for ds in datasets:
        if ds == REAL[0] and any(x in datasets for x in SYNTHETIC):
            y += 0.7  # gap between synthetic and real
        ypos[ds] = y
        y += 1.0
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_IN, 0.42 * len(datasets) + 1.3), sharey=True)
    anchors = list(dict.fromkeys(t.anchor))
    colors = {anchors[0]: COLOR_SGT, anchors[-1]: COLOR_SGT_LIGHT}
    offsets = {anchors[0]: -0.17, anchors[-1]: 0.17} if len(anchors) > 1 else {anchors[0]: 0}
    for ax, ref, title in zip(
        axes, ("mse", "best"), ("Against MSE", "Against the best classical loss")
    ):
        sub = t[t.reference == ref]
        lim = np.nanmax(np.abs(np.r_[sub.rel_ci_low, sub.rel_ci_high])) * 1.08
        for ds in datasets:
            yy = ypos[ds]
            ax.axhspan(yy - 0.45, yy + 0.45, color=COLOR_BAND, lw=0, zorder=0)
            for anc in anchors:
                r = sub[(sub.dataset == ds) & (sub.anchor == anc)]
                if r.empty:
                    continue
                r = r.iloc[0]
                yk = yy + offsets[anc]
                sig = np.isfinite(r.perm_p) and r.perm_p < 0.05
                ax.plot(
                    [r.rel_ci_low, r.rel_ci_high],
                    [yk, yk],
                    color=colors[anc],
                    lw=1.3,
                    zorder=2,
                )
                ax.plot(
                    r.rel,
                    yk,
                    "o",
                    ms=4.6,
                    color=colors[anc] if sig else "white",
                    mec=colors[anc],
                    mew=1.1,
                    zorder=3,
                )
            if ref == "best":
                name = CLASSICAL_LABEL[sub[sub.dataset == ds].iloc[0].reference_loss]
                ax.annotate(
                    name,
                    (lim, yy),
                    xytext=(-2, 0),
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    fontsize=6.6,
                    color=COLOR_MUTED,
                    zorder=4,
                )
        ax.axvline(0, color=COLOR_MUTED, lw=0.8, zorder=1)
        ax.set_xlim(-lim, lim)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Change in test MASE relative to the baseline (%)")
        ax.grid(axis="y", visible=False)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}"))
    axes[0].set_yticks([ypos[d] for d in datasets])
    axes[0].set_yticklabels([DATASET_LABEL[d] for d in datasets])
    axes[0].invert_yaxis()
    axes[0].annotate(
        "lower is better for SGT",
        (0, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(-4, 3),
        textcoords="offset points",
        ha="right",
        fontsize=6.6,
        color=COLOR_MUTED,
    )
    handles = [
        plt.Line2D([], [], marker="o", color=colors[a], lw=1.3, ms=4.2, label=a) for a in anchors
    ] + [
        plt.Line2D([], [], marker="o", color=COLOR_MUTED, lw=0, ms=4.2, label="p < 0.05"),
        plt.Line2D(
            [], [], marker="o", color="white", mec=COLOR_MUTED, lw=0, ms=4.2, label="p ≥ 0.05"
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.01),
        title="Mean over 10 seed-paired runs, whiskers = 95 % bootstrap CI",
        title_fontsize=6.8,
    )
    fig.tight_layout(w_pad=1.6, rect=(0, 0.1, 1, 1))
    return _save(fig, out)


def _loss_panel(ax, rows: pd.DataFrame, metric: str, title: str) -> None:
    rows = rows.iloc[::-1].reset_index(drop=True)  # first row at the top
    finite = rows[np.isfinite(rows["mean"])]
    core = finite[finite["key"] != "tukey"]
    upper = core["ci_high"].max()
    lower = core["ci_low"].min()
    span = max(upper - lower, 1e-6)
    xmax = upper + 0.18 * span
    xmin = max(0.0, lower - 0.12 * span)
    # Reference band: the best classical loss and its CI.
    cls = finite[finite["family"] == "classical"]
    if not cls.empty:
        b = cls.loc[cls["mean"].idxmin()]
        ax.axvspan(b["ci_low"], b["ci_high"], color=COLOR_BAND, lw=0, zorder=0)
        ax.axvline(b["mean"], color=COLOR_CLASSICAL_LIGHT, lw=0.8, zorder=1)
    for i, r in rows.iterrows():
        color = COLOR_SGT if r["family"] == "sgt" else COLOR_CLASSICAL
        if not np.isfinite(r["mean"]):
            continue
        if r["mean"] > xmax:  # off-scale (Tukey collapse): clip and annotate
            ax.plot([xmax - 0.1 * span, xmax], [i, i], color=color, lw=1.4, solid_capstyle="butt")
            ax.annotate(
                f"{r['mean']:.1f} (off scale)",
                (xmax, i),
                xytext=(-2, 0),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=6.4,
                color=COLOR_MUTED,
            )
            continue
        ax.plot(
            [r["ci_low"], r["ci_high"]],
            [i, i],
            color=color,
            lw=1.3,
            zorder=2,
        )
        marker = "*" if r["key"] == "sgt_best" else "o"
        ax.plot(
            r["mean"],
            i,
            marker,
            ms=6.4 if marker == "*" else 4.2,
            color=color,
            mec="white",
            mew=0.6,
            zorder=3,
        )
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
            "persistence",
            (1.0, len(rows) - 0.6),
            xytext=(2, 0),
            textcoords="offset points",
            fontsize=6.4,
            color=COLOR_MUTED,
        )


def _legend(fig, loc=(0.5, 0.0)):
    handles = [
        plt.Line2D([], [], marker="o", color=COLOR_SGT, lw=1.3, ms=4, label="SGT loss"),
        plt.Line2D(
            [], [], marker="*", color=COLOR_SGT, lw=0, ms=6.4, label="SGT, best of the grid"
        ),
        plt.Line2D(
            [], [], marker="o", color=COLOR_CLASSICAL, lw=1.3, ms=4, label="Classical losses"
        ),
        matplotlib.patches.Patch(color=COLOR_BAND, label="Best classical loss, 95 % bootstrap CI"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=loc,
        title="Mean over 10 seed-paired runs, whiskers = 95 % bootstrap CI",
        title_fontsize=6.8,
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
        nrows, ncols, figsize=(width, 2.45 * nrows + 0.5), squeeze=False, sharey="row"
    )
    for ax, ds in zip(axes.flat, present):
        _loss_panel(ax, loss_rows(df[df.dataset == ds], metric), metric, DATASET_LABEL.get(ds, ds))
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))
    for ax in list(axes.flat)[len(present) :]:
        ax.axis("off")
    bottom = 0.15 if nrows == 1 else 0.075  # room for the legend under one row of panels
    fig.tight_layout(h_pad=1.4, w_pad=1.2, rect=(0, bottom, 1, 1 if title is None else 0.96))
    if title:
        fig.suptitle(title, fontsize=9)
    _legend(fig)
    return _save(fig, out)


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
        im = ax.imshow(vals, cmap=CMAP_BLUE, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels([f"{c:g}" for c in piv.columns])
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels([f"{r:g}" for r in piv.index])
        ax.set_xlabel("Norm exponent p")
        ax.set_ylabel("Tail parameter q")
        ax.grid(False)
        ax.tick_params(length=0)
        for s in ("left", "bottom"):
            ax.spines[s].set_visible(False)
        best = np.unravel_index(np.nanargmin(vals), vals.shape)
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                if not np.isfinite(v):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=6.5, color=COLOR_MUTED)
                    continue
                light = (v - vmin) / max(vmax - vmin, 1e-9) > 0.5
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
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False, ec=COLOR_INK, lw=1.2
            )
        )
        cls = df[(df.dataset == ds) & (df.loss_type != "sgt")].groupby("loss_type")[metric].mean()
        fmt = "{:.3f}" if metric != "best_test_smape" else "{:.1f}"
        pairs = [
            f"{CLASSICAL_LABEL[k]} {fmt.format(cls[k])}"
            for k in ("mse", "mae", "huber", "cauchy")
            if k in cls
        ]
        ref = "   ".join(pairs[:2]) + "\n" + "   ".join(pairs[2:])
        ax.set_title(f"{DATASET_LABEL.get(ds, ds)}\n{ref}", loc="left", fontsize=7.2)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6.5, length=2)
        cb.outline.set_visible(False)
        cb.set_label(METRIC_LABEL.get(metric, metric), fontsize=6.5)
    for ax in list(axes.flat)[len(present) :]:
        ax.axis("off")
    fig.tight_layout(h_pad=1.6, w_pad=1.6)
    return _save(fig, out)


def lambda_effect_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Seed-paired effect of every lambda != 0 vs its symmetric twin, per dataset/(p,q)."""
    rows = []
    sgt = df[(df.loss_type == "sgt") & (df.dataset != HEAD_DATASET)]
    for (ds, p, q), g in sgt.groupby(["dataset", "sgt_loss_p", "sgt_loss_q"]):
        sym = g[g.sgt_loss_lambda == 0]
        if sym.empty:
            continue
        for lam, gl in g[g.sgt_loss_lambda != 0].groupby("sgt_loss_lambda"):
            r = paired_diff(gl, sym, metric)
            rows.append(
                dict(
                    dataset=ds,
                    p=p,
                    q=q,
                    lam=lam,
                    n=r.n,
                    mean_symmetric=sym[metric].mean(),
                    mean_skewed=gl[metric].mean(),
                    diff=r.mean,
                    ci_low=r.ci_low,
                    ci_high=r.ci_high,
                    perm_p=r.p,
                    wilcoxon_p=r.wilcoxon_p,
                )
            )
    return pd.DataFrame(rows)


def figure_lambda(df: pd.DataFrame, metric: str, out: Path, fitted_lambda: float = 0.17) -> Path:
    t = lambda_effect_table(df, metric)
    if t.empty:
        raise typer.BadParameter("no skewed (lambda != 0) SGT runs in the results")
    fig, axes = plt.subplots(
        1, 2, figsize=(PAGE_IN, 2.7), gridspec_kw=dict(width_ratios=[1.15, 1])
    )
    # Left: fine grid on the skewed heavy-tailed data (all (p, q) with a lambda sweep).
    ax = axes[0]
    fine = t[(t.dataset == "heavy-tailed-skewed")]
    styles = {
        (2.0, 2.5): (COLOR_SGT, "o"),
        (2.0, 10.0): (COLOR_SGT, "s"),
        (1.5, 2.5): (COLOR_SGT_LIGHT, "o"),
        (1.5, 10.0): (COLOR_SGT_LIGHT, "s"),
    }
    if fitted_lambda is not None and not fine.empty:
        ax.axvline(fitted_lambda, color=COLOR_FAINT, lw=0.8, ls=(0, (3, 2)))
        ax.annotate(
            f"fitted λ̂ = {fitted_lambda:g}",
            (fitted_lambda, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(3, -2),
            textcoords="offset points",
            va="top",
            fontsize=6.6,
            color=COLOR_MUTED,
        )
    for (p, q), g in fine.groupby(["p", "q"]):
        g = g.sort_values("lam")
        color, mk = styles.get((p, q), (COLOR_SGT, "o"))
        ax.errorbar(
            g["lam"],
            100 * g["diff"] / g["mean_symmetric"],
            yerr=100
            * np.vstack([g["diff"] - g["ci_low"], g["ci_high"] - g["diff"]])
            / g["mean_symmetric"].to_numpy(),
            fmt=mk + "-",
            ms=3.6,
            lw=1.1,
            capsize=0,
            elinewidth=0.9,
            color=color,
            mec="white",
            mew=0.5,
            label=f"p = {p:g}, q = {q:g}",
        )
    ax.axhline(0, color=COLOR_MUTED, lw=0.8)
    ax.set_xlabel("Skew parameter λ of the loss")
    ax.set_ylabel("Change in test MASE vs λ = 0 (%)")
    ax.set_title("Heavy-tailed, skewed data (increments skewed)", loc="left")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    if not fine.empty:
        ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(0.0, 0.93), handlelength=1.6)
    # Right: the main-grid lambda values on every dataset at (2, 2.5): all harmed.
    ax = axes[1]
    main = t[(t.p == 2.0) & (t.q == 2.5) & (t.lam.isin([0.5, 0.9]))]
    datasets = [d for d in SYNTHETIC + REAL if (main.dataset == d).any()]
    ypos = {d: i for i, d in enumerate(datasets)}
    for lam, color, off in ((0.5, COLOR_SGT_LIGHT, -0.17), (0.9, COLOR_SGT, 0.17)):
        g = main[main.lam == lam]
        for _, r in g.iterrows():
            rel = 100 * r["diff"] / r["mean_symmetric"]
            lo = 100 * r["ci_low"] / r["mean_symmetric"]
            hi = 100 * r["ci_high"] / r["mean_symmetric"]
            yk = ypos[r.dataset] + off
            ax.plot([lo, hi], [yk, yk], color=color, lw=1.3)
            ax.plot(rel, yk, "o", ms=4, color=color, mec="white", mew=0.5)
    for d, yy in ypos.items():
        ax.axhspan(yy - 0.45, yy + 0.45, color=COLOR_BAND, lw=0, zorder=0)
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels([DATASET_LABEL[d] for d in datasets])
    ax.invert_yaxis()
    ax.axvline(0, color=COLOR_MUTED, lw=0.8)
    ax.set_xlabel("Change in test MASE vs λ = 0 (%)")
    ax.set_title("λ = 0.5 and 0.9 at p = 2, q = 2.5 (every dataset)", loc="left")
    ax.grid(axis="y", visible=False)
    ax.set_xscale("symlog", linthresh=10)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}"))
    handles = [
        plt.Line2D([], [], marker="o", color=COLOR_SGT_LIGHT, lw=1.3, ms=4, label="λ = 0.5"),
        plt.Line2D([], [], marker="o", color=COLOR_SGT, lw=1.3, ms=4, label="λ = 0.9"),
    ]
    ax.legend(handles=handles, loc="upper left")
    fig.tight_layout(w_pad=2.0)
    return _save(fig, out)


def figure_heads(df: pd.DataFrame, metric: str, out: Path) -> Path:
    h = df[df.dataset == HEAD_DATASET].copy()
    if h.empty:
        raise typer.BadParameter("no head-study runs (dataset 'head-heavy-tailed') in the results")
    h["cfg"] = h.apply(
        lambda r: CLASSICAL_LABEL.get(r.loss_type, None)
        or sgt_label(r.sgt_loss_p, r.sgt_loss_q, r.sgt_loss_lambda),
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.55))
    heads = sorted(h.num_heads.unique())
    for cfg, g in h.groupby("cfg"):
        m = g.groupby("num_heads")[metric].mean().reindex(heads)
        ax.plot(
            heads,
            m.values,
            color=COLOR_SGT if cfg.startswith("SGT") else COLOR_CLASSICAL,
            lw=0.8,
            alpha=0.5,
        )
    agg = h.groupby("num_heads")[metric].agg(["mean"]).reindex(heads)
    bounds = h.groupby("num_heads")[metric].apply(_ci).reindex(heads)
    agg["ci_low"] = [b[0] for b in bounds]
    agg["ci_high"] = [b[1] for b in bounds]
    ax.fill_between(
        heads,
        agg["ci_low"],
        agg["ci_high"],
        color=COLOR_INK,
        alpha=0.08,
        lw=0,
    )
    ax.plot(
        heads, agg["mean"], "o-", color=COLOR_INK, lw=1.6, ms=4.5, mec="white", mew=0.6, zorder=5
    )
    single = h[h.num_heads == heads[0]]
    for nh in heads[1:]:
        r = paired_diff(h[h.num_heads == nh], single, metric)
        ax.annotate(
            f"p = {r.p:.2f}" if np.isfinite(r.p) else "",
            (nh, agg.loc[nh, "mean"]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
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
            [],
            [],
            color=COLOR_INK,
            marker="o",
            lw=1.6,
            ms=4,
            label="All losses, mean, 95 % bootstrap CI",
        ),
        plt.Line2D([], [], color=COLOR_SGT, lw=0.8, alpha=0.7, label="SGT configurations"),
        plt.Line2D([], [], color=COLOR_CLASSICAL, lw=0.8, alpha=0.7, label="Classical losses"),
    ]
    ax.legend(handles=handles, loc="upper right")
    fig.tight_layout()
    return _save(fig, out)


# ---------------------------------------------------------------------------
# Figures: datasets, loss family, training curves
# ---------------------------------------------------------------------------


def _sgt_pdf(x, sigma, lam, p, q, mu=0.0):
    from skewed_sequences.data.synthetic.generate_data import SkewedGeneralizedT

    return SkewedGeneralizedT(mu=mu, sigma=sigma, lam=lam, p=p, q=q).pdf(x)


def dataset_card_stats(arr: np.ndarray) -> dict:
    """Increment statistics and the SGT increment fit used on a dataset card."""
    from skewed_sequences.metrics import hill_estimator, sgt_increment_fit

    inc = np.diff(arr[:, :, 0], axis=1).ravel()
    s_hat = 1.4826 * float(np.median(np.abs(inc - np.median(inc))))
    fit = sgt_increment_fit(arr)
    return dict(
        s_hat=s_hat,
        kurtosis=float(_kurtosis(inc, fisher=True)),
        hill=float(hill_estimator(inc)),
        lam=float(fit["lam"]),
        q=float(fit["q"]),
        mu=float(fit.get("mu", np.median(inc))),
        increments=inc,
    )


def _typical_sequence(arr: np.ndarray) -> np.ndarray:
    """The sequence whose increment kurtosis is closest to the dataset median."""
    inc = np.diff(arr[:, :, 0], axis=1)
    k = _kurtosis(inc, axis=1, fisher=True)
    k = np.where(np.isfinite(k), k, np.nan)
    idx = int(np.nanargmin(np.abs(k - np.nanmedian(k))))
    return arr[idx, :, 0]


def figure_datasets(data_dir: Path, out: Path, datasets=None, ncols: int = 4) -> Path:
    """One card per dataset: a typical standardised sequence (top) and the distribution of
    the one-step increments in units of the residual scale (bottom, log density) with the
    fitted SGT density and a Gaussian of the same scale."""
    datasets = datasets or SYNTHETIC + REAL
    present = [d for d in datasets if (Path(data_dir) / f"{d}.npy").exists()]
    if not present:
        raise typer.BadParameter(f"no <dataset>.npy arrays in {data_dir}")
    nrows = int(np.ceil(len(present) / ncols))
    fig = plt.figure(figsize=(PAGE_IN, 2.5 * nrows + 0.3))
    outer = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.7, wspace=0.32)
    zmax = 12.0
    for k, ds in enumerate(present):
        arr = np.load(Path(data_dir) / f"{ds}.npy")
        st = dataset_card_stats(arr)
        inner = outer[k].subgridspec(2, 1, height_ratios=[0.9, 1.15], hspace=0.38)
        ax_s = fig.add_subplot(inner[0])
        ax_h = fig.add_subplot(inner[1])
        seq = _typical_sequence(arr)
        ax_s.plot(seq, color=COLOR_INK, lw=0.7)
        ax_s.set_xlim(0, len(seq) - 1)
        ax_s.set_xticks([0, 150, 300])
        ax_s.set_yticks([-2, 0, 2])
        ax_s.tick_params(labelsize=6.2)
        ax_s.grid(False)
        ax_s.set_ylabel("value", fontsize=6.4, color=COLOR_MUTED, labelpad=2)
        z = st["increments"] / st["s_hat"]
        z = z[np.abs(z) <= zmax]
        bins = np.linspace(-zmax, zmax, 61)
        ax_h.hist(z, bins=bins, density=True, color=COLOR_FAINT, alpha=0.55, lw=0)
        xx = np.linspace(-zmax, zmax, 600)
        s = st["s_hat"]
        sgt = s * _sgt_pdf(xx * s, sigma=s, lam=st["lam"], p=2.0, q=st["q"], mu=st["mu"])
        gau = s * _norm.pdf(xx * s, loc=st["mu"], scale=s)
        ax_h.plot(xx, gau, color=COLOR_CLASSICAL, lw=0.9, ls=(0, (3, 1.5)))
        ax_h.plot(xx, sgt, color=COLOR_SGT, lw=1.1)
        ax_h.set_yscale("log")
        ax_h.set_ylim(1e-5, 1.5)
        ax_h.set_yticks([1e-4, 1e-2, 1])
        ax_h.set_xlim(-zmax, zmax)
        ax_h.set_xticks([-10, -5, 0, 5, 10])
        ax_h.tick_params(labelsize=6.2)
        ax_h.grid(False)
        ax_h.set_xlabel("one-step increment / ŝ", fontsize=6.4, color=COLOR_MUTED, labelpad=1)
        ax_h.set_ylabel("density", fontsize=6.4, color=COLOR_MUTED, labelpad=2)
        info = (
            f"kurtosis {st['kurtosis']:.0f}, Hill α {st['hill']:.1f}\n"
            f"fit: λ̂ = {st['lam']:.2f}, q̂ = {st['q']:.2f}"
        )
        ax_s.set_title(DATASET_LABEL.get(ds, ds), loc="left", fontsize=8, pad=22)
        ax_s.annotate(
            info,
            (0.0, 1.04),
            xycoords="axes fraction",
            va="bottom",
            fontsize=6.3,
            color=COLOR_MUTED,
            linespacing=1.3,
        )
    # Legend in the spare cell (or below the grid when the grid is full).
    handles = [
        plt.Line2D([], [], color=COLOR_INK, lw=0.8, label="typical sequence (standardised)"),
        matplotlib.patches.Patch(color=COLOR_FAINT, alpha=0.55, label="one-step increments"),
        plt.Line2D([], [], color=COLOR_SGT, lw=1.1, label="SGT fit (p = 2, σ = ŝ)"),
        plt.Line2D(
            [], [], color=COLOR_CLASSICAL, lw=0.9, ls=(0, (3, 1.5)), label="Gaussian, same ŝ"
        ),
    ]
    if len(present) < nrows * ncols:
        ax_l = fig.add_subplot(outer[len(present)])
        ax_l.axis("off")
        ax_l.legend(handles=handles, loc="center left", fontsize=6.6, handlelength=1.8)
    else:
        fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    return _save(fig, out)


def _baseline_losses(e: np.ndarray, s: float = 1.0) -> dict:
    """The robust baselines at the paper's scale constants (``s`` = residual scale)."""
    d, g, c, eps = 1.345 * s, (2.3849 * s) ** 2, 4.685 * s, 1.345 * s
    huber = np.where(np.abs(e) <= d, 0.5 * e**2, d * (np.abs(e) - 0.5 * d))
    cauchy = g * np.log1p(e**2 / g)
    tukey = np.where(np.abs(e) <= c, (c**2 / 6) * (1 - (1 - (e / c) ** 2) ** 3), c**2 / 6)
    charb = np.sqrt(e**2 + eps**2) - eps
    return {
        "MSE": e**2,
        "MAE": np.abs(e),
        "Huber": huber,
        "Cauchy": cauchy,
        "Tukey": tukey,
        "Charbonnier": charb,
    }


def figure_loss_family(out: Path) -> Path:
    """The SGT loss as a family (q at fixed p; p at fixed q) and the gradient magnitude
    of every loss used in the paper, at the same residual scale ŝ = 1."""
    from skewed_sequences.visualization.visualize_losses import sgt_loss

    e = np.linspace(-8, 8, 1601)
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_IN, 2.35))
    qs = [1.3, 2.5, 5.0, 20.0]
    shades = [CMAP_BLUE(x) for x in (0.0, 0.25, 0.5, 0.7)]
    ax = axes[0]
    ax.plot(e, 0.5 * e**2, color=COLOR_FAINT, lw=0.9, ls=(0, (3, 1.5)), label="MSE / 2")
    for q, col in zip(qs, shades):
        ax.plot(e, sgt_loss(e, 2.0, q, 1.0), color=col, lw=1.2, label=f"q = {q:g}")
    ax.set_title("Tail parameter q (p = 2)", loc="left")
    ax.set_ylim(0, 12)
    ax.legend(loc="upper center", ncol=1, handlelength=1.6)
    ax = axes[1]
    ax.plot(e, 0.5 * e**2, color=COLOR_FAINT, lw=0.9, ls=(0, (3, 1.5)), label="MSE / 2")
    ax.plot(
        e, np.abs(e) / np.sqrt(2), color=COLOR_FAINT, lw=0.9, ls=(0, (1, 1.5)), label="MAE / √2"
    )
    for p, col, ls in (
        (1.0, shades[2], (0, (4, 1.5))),
        (1.5, shades[1], (0, (6, 1.5))),
        (2.0, shades[0], "-"),
    ):
        ax.plot(e, sgt_loss(e, p, 2.5, 1.0), color=col, lw=1.2, ls=ls, label=f"p = {p:g}")
    ax.set_title("Norm exponent p (q = 2.5)", loc="left")
    ax.set_ylim(0, 12)
    ax.legend(loc="upper center", ncol=1, handlelength=1.8)
    for ax in axes[:2]:
        ax.set_xlabel("residual e / ŝ")
        ax.set_ylabel("loss")
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
        ax.set_xlim(-8, 8)
    # Gradient magnitudes, each normalised to 1 at e = ŝ.
    ax = axes[2]
    ep = np.linspace(0.02, 8, 800)
    curves = {}
    for name, L in _baseline_losses(ep).items():
        curves[name] = (np.gradient(L, ep), COLOR_CLASSICAL)
    for (p, q), col in (
        ((2.0, 2.5), shades[0]),
        ((2.0, 20.0), shades[2]),
        ((1.5, 2.5), shades[1]),
    ):
        curves[f"SGT ({p:g}, {q:g})"] = (np.gradient(sgt_loss(ep, p, q, 1.0), ep), col)
    styles = {
        "MSE": (0, (3, 1.5)),
        "MAE": (0, (1, 1.5)),
        "Huber": "-",
        "Cauchy": (0, (5, 1.5)),
        "Tukey": (0, (2, 1)),
        "Charbonnier": (0, (4, 1.5, 1, 1.5)),
    }
    ymax = 2.6
    labels = []
    for name, (gr, col) in curves.items():
        y = gr / np.interp(1.0, ep, gr)
        ax.plot(ep, y, color=col, lw=1.15, ls=styles.get(name, "-"), label=name)
        inside = np.where(y <= ymax)[0]
        k = inside[-1] if len(inside) else len(ep) - 1
        labels.append((ep[k], min(y[k], ymax), name, col))
    # Direct labels: at the right edge for curves that stay in view, at the exit point
    # for the two that leave through the top.
    n_exit = 0
    for x, y, name, col in labels:
        exits_top = x < ep[-1] - 0.05
        if exits_top:
            n_exit += 1
        ax.annotate(
            name,
            (x, y),
            xytext=(3, -8 - 9 * (n_exit - 1) if exits_top else 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=6.2,
            color=col,
        )
    ax.set_xlabel("residual |e| / ŝ")
    ax.set_ylabel("gradient magnitude, = 1 at |e| = ŝ")
    ax.set_title("Gradient of every loss used here", loc="left")
    ax.set_xlim(0, 8)
    ax.set_ylim(0, ymax)
    ax.set_xlim(0, 9.6)
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.4)
    return _save(fig, out)


def figure_tukey_curves(curves_csv: Path, out: Path, metric_key: str = "train_mae") -> Path:
    """Per-epoch training error of every run on the sparse series: the Tukey runs that
    never leave their initialisation against the SGT anchor and the Huber loss."""
    d = pd.read_csv(curves_csv)
    d = d[d.key == metric_key]
    if d.empty:
        raise typer.BadParameter(f"no '{metric_key}' rows in {curves_csv}")
    datasets = [x for x in REAL if (d.dataset == x).any()]
    fig, axes = plt.subplots(1, len(datasets), figsize=(PAGE_IN, 2.5), sharey=True, squeeze=False)
    series = [
        ("tukey", COLOR_CLASSICAL, 0.85, "Tukey"),
        ("huber", COLOR_FAINT, 0.6, "Huber"),
        ("sgt", COLOR_SGT, 0.7, "SGT (p = 2, q = 2.5)"),
    ]
    for ax, ds in zip(axes.flat, datasets):
        g = d[d.dataset == ds]
        for loss, color, alpha, _ in series:
            for _, run in g[g.loss_type == loss].groupby("run_uuid"):
                run = run.sort_values("step")
                ax.plot(run.step, run.value, color=color, lw=0.8, alpha=alpha)
        ax.set_yscale("log")
        ax.set_title(DATASET_LABEL.get(ds, ds), loc="left")
        ax.set_xlabel("epoch")
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)
    axes[0, 0].set_ylabel("training MAE (standardised units)")
    handles = [plt.Line2D([], [], color=c, lw=1.2, label=lab) for _, c, _, lab in series]
    axes[0, -1].legend(handles=handles, loc="center right")
    fig.tight_layout(w_pad=1.4)
    return _save(fig, out)


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
            f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
            if r["mean"] < 10
            else f"{r['mean']:.1f} [{r['ci_low']:.1f}, {r['ci_high']:.1f}]"
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
    """Per dataset: the best loss overall, the best classical, the tail-tolerant anchor vs
    MSE and vs the best classical (paired p), and the best SGT vs the best classical."""
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
        r_mse = paired_diff(anchor, d[d.loss_type == "mse"], metric)
        r_cls = paired_diff(anchor, d[d.loss_type == best_cls], metric)
        best_sgt = best_sgt_config(d, metric)
        r_b = paired_diff(_sel_sgt(d, *best_sgt), d[d.loss_type == best_cls], metric)
        rows.append(
            {
                "Dataset": DATASET_LABEL.get(ds, ds),
                "Best overall": best_any_label,
                "Best classical": CLASSICAL_LABEL[best_cls],
                "SGT (2, 2.5) vs MSE": f"{r_mse.mean:+.3f} (p = {r_mse.p:.3f})",
                "SGT (2, 2.5) vs best classical": f"{r_cls.mean:+.3f} (p = {r_cls.p:.3f})",
                "Best SGT vs best classical": f"{r_b.mean:+.3f} (p = {r_b.p:.3f})",
            }
        )
    return pd.DataFrame(rows)


def holm_table(df: pd.DataFrame, metric: str, anchors=None) -> pd.DataFrame:
    """Every tail-tolerant anchor against every baseline on every dataset: the seed-paired
    difference, the exact permutation p and its Holm adjustment within the family of tests
    on that dataset (two anchors x six baselines = 12). The paper's appendix table."""
    anchors = anchors or ANCHORS[:2]
    frames = []
    for ds in SYNTHETIC + REAL:
        d = df[df.dataset == ds]
        if d.empty:
            continue
        rows = []
        for p, q, lam in anchors:
            a = _sel_sgt(d, p, q, lam)
            if a.empty:
                continue
            for loss in CLASSICAL:
                b = d[d.loss_type == loss]
                if b.empty:
                    continue
                r = paired_diff(a, b, metric)
                rows.append(
                    dict(
                        dataset=ds,
                        anchor=sgt_label(p, q, lam),
                        baseline=CLASSICAL_LABEL[loss],
                        baseline_mean=b[metric].mean(),
                        diff=r.mean,
                        ci_low=r.ci_low,
                        ci_high=r.ci_high,
                        perm_p=r.p,
                        wilcoxon_p=r.wilcoxon_p,
                        n=r.n,
                    )
                )
        if not rows:
            continue
        t = pd.DataFrame(rows)
        t["family_size"] = int(np.isfinite(t["perm_p"]).sum())  # what holm() adjusts over
        t["holm_p"] = holm(t["perm_p"].to_numpy())
        frames.append(t)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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
def summary(
    input_path: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT, metric: str = METRIC
):
    """Headline figure: anchors vs MSE and vs the best classical loss, per dataset."""
    _style()
    df = load_results(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_summary(df, metric, output_dir / f"summary_{metric}.png")
    anchor_effects(df, metric).round(4).to_csv(output_dir / f"summary_{metric}.csv", index=False)


@app.command()
def losses(
    input_path: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT, metric: str = METRIC
):
    """Per-loss mean, 95 % bootstrap CI, one panel per dataset (synthetic 2x2, real 1x3)."""
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
    output_dir.mkdir(parents=True, exist_ok=True)
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


@app.command("loss-family")
def loss_family(output_dir: Path = DEFAULT_OUTPUT):
    """The SGT loss and gradient shapes next to the robust baselines (no results needed)."""
    _style()
    figure_loss_family(output_dir / "loss_family.png")


@app.command()
def datasets(data_dir: Path = REPORTS_DIR / "datasets", output_dir: Path = DEFAULT_OUTPUT):
    """Dataset cards from ``<data_dir>/<dataset>.npy`` arrays of shape (N, T, 1)."""
    _style()
    figure_datasets(data_dir, output_dir / "datasets.png")


@app.command("tukey-curves")
def tukey_curves(
    curves_csv: Path = REPORTS_DIR / "training_curves_sparse.csv",
    output_dir: Path = DEFAULT_OUTPUT,
):
    """Per-epoch training error on the sparse series (needs the exported MLflow metrics)."""
    _style()
    figure_tukey_curves(curves_csv, output_dir / "tukey_curves.png")


@app.command()
def tables(
    input_path: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT, metric: str = METRIC
):
    """Main results table, summary table and lambda table (Markdown + CSV)."""
    df = load_results(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    main = results_table(df, metric)
    main.to_csv(output_dir / f"table_results_{metric}.csv")
    summ = summary_table(df, metric)
    summ.to_csv(output_dir / f"table_summary_{metric}.csv", index=False)
    lam = lambda_effect_table(df, metric)
    lam.round(4).to_csv(output_dir / f"table_lambda_{metric}.csv", index=False)
    hol = holm_table(df, metric)
    hol.round(4).to_csv(output_dir / f"table_holm_{metric}.csv", index=False)
    md = (
        f"## Results ({METRIC_LABEL.get(metric, metric)}, mean [95 % bootstrap CI] over 10"
        " seed-paired runs)\n\n"
        + _to_markdown(main.rename_axis("Loss"))
        + "\n## Summary (exact permutation p)\n\n"
        + _to_markdown(summ, index=False)
        + "\n## Skew parameter (seed-paired vs λ = 0; exact permutation p and Wilcoxon p)\n\n"
        + _to_markdown(lam.round(4), index=False)
        + "\n## Anchors vs baselines with raw permutation p < 0.05, Holm within dataset\n\n"
        + _to_markdown((hol[hol.perm_p < 0.05] if not hol.empty else hol).round(4), index=False)
    )
    (output_dir / f"tables_{metric}.md").write_text(md)
    typer.echo(md)
    logger.success(f"tables written to {output_dir}")


@app.command("all")
def all_figures(
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT,
    metric: str = METRIC,
    data_dir: Path | None = None,
    curves_csv: Path | None = None,
):
    """Every figure and table; the dataset cards and the training curves only when their
    inputs are given."""
    summary(input_path, output_dir, metric)
    losses(input_path, output_dir, metric)
    grid(input_path, output_dir, metric)
    lambda_effect(input_path, output_dir, metric)
    heads(input_path, output_dir, metric)
    loss_family(output_dir)
    tables(input_path, output_dir, metric)
    if data_dir is not None:
        datasets(data_dir, output_dir)
    if curves_csv is not None:
        tukey_curves(curves_csv, output_dir)


if __name__ == "__main__":
    app()
