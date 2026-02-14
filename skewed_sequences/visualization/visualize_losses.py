"""Visualize SGT loss interpolation between classical loss functions.

Creates figures showing how the SGT loss with different parameter
combinations (p, q, s) smoothly interpolates between and beyond
classical losses (MSE, MAE, Huber, Cauchy, Tukey).

Uses the q^p reparameterization throughout.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import beta as beta_func
import typer

from skewed_sequences.config import FIGURES_DIR
from skewed_sequences.visualization.style import PALETTE_SEQ, apply_style

app = typer.Typer(pretty_exceptions_show_locals=False)


# ---------------------------------------------------------------------------
# Loss function implementations (NumPy, for plotting)
# ---------------------------------------------------------------------------


def sgt_loss(x, p, q, s, lam=0.0):
    """SGT loss with q^p reparameterization (symmetric when lam=0)."""
    qp = q**p
    B1 = beta_func(1.0 / p, qp)
    B3 = beta_func(3.0 / p, qp - 2.0 / p)
    v = (1.0 / q) / np.sqrt(B3 / B1)  # simplified for lam=0
    scaled = np.abs(x / (s * v)) ** p
    return (1.0 / p + qp) * np.log(1 + scaled / qp)


def mse_loss(x):
    return x**2


def mae_loss(x):
    return np.abs(x)


def huber_loss(x, delta=1.0):
    ax = np.abs(x)
    return np.where(ax <= delta, 0.5 * x**2, delta * (ax - 0.5 * delta))


def cauchy_loss(x, gamma=2.0):
    return gamma * np.log(1 + x**2 / gamma)


def tukey_loss(x, c=4.685):
    ax = np.abs(x)
    r = x / c
    return np.where(ax <= c, (c**2 / 6) * (1 - (1 - r**2) ** 3), c**2 / 6)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


CLASSICAL = {
    "MSE": {"fn": mse_loss, "color": PALETTE_SEQ[0], "ls": "-"},
    "MAE": {"fn": mae_loss, "color": PALETTE_SEQ[1], "ls": "-"},
    "Huber": {"fn": lambda x: huber_loss(x, delta=1.0), "color": PALETTE_SEQ[2], "ls": "-"},
    "Cauchy": {"fn": lambda x: cauchy_loss(x, gamma=2.0), "color": PALETTE_SEQ[3], "ls": "-"},
    "Tukey": {"fn": lambda x: tukey_loss(x, c=4.685), "color": PALETTE_SEQ[4], "ls": "-"},
}


def _norm_at_1(fn, x):
    """Evaluate fn and divide by fn(1) for shape comparison."""
    y = fn(x)
    y1 = fn(np.array([1.0]))
    if hasattr(y1, "__len__"):
        y1 = y1[0]
    return y / y1 if y1 > 0 else y


def _plot_classical_normed(ax, x_pos, names=None, lw=2.5, alpha=0.7):
    """Plot selected classical losses normalized by f(1)."""
    names = names or list(CLASSICAL.keys())
    for name in names:
        spec = CLASSICAL[name]
        y = _norm_at_1(spec["fn"], x_pos)
        ax.plot(x_pos, y, color=spec["color"], ls=spec["ls"], lw=lw, alpha=alpha, label=name)


# ---------------------------------------------------------------------------
# Main plotting commands
# ---------------------------------------------------------------------------


@app.command()
def main(
    output_dir: Path = FIGURES_DIR / "loss_comparison",
    x_max: float = 5.0,
    n_points: int = 1000,
):
    """Generate all loss comparison figures."""
    apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-x_max, x_max, n_points)
    x_pos = np.linspace(0.01, x_max, n_points // 2)  # positive half for normalized

    _fig_overview(x, x_pos, output_dir)
    _fig_p_sweep(x_pos, output_dir)
    _fig_q_sweep(x_pos, output_dir)
    _fig_s_sweep(x_pos, output_dir)
    _fig_interpolation_grid(x_pos, output_dir)
    _fig_sq_interaction(x_pos, output_dir)
    _fig_tukey_comparison(output_dir)

    typer.echo(f"Figures saved to {output_dir}")


def _fig_overview(x, x_pos, output_dir):
    """Figure 1: All classical losses + key SGT curves (normalized)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left panel: raw scale ---
    ax = axes[0]
    for name, spec in CLASSICAL.items():
        ax.plot(
            x, spec["fn"](x), color=spec["color"], ls=spec["ls"], lw=2.0, alpha=0.85, label=name
        )
    sgt_configs = [
        (2.0, 20.0, 1.0, "SGT(p=2, q=20, s=1)", "#1f77b4", "--"),
        (1.0, 20.0, 1.0, "SGT(p=1, q=20, s=1)", "#ff7f0e", "--"),
        (2.0, 2.5, 1.0, "SGT(p=2, q=2.5, s=1)", "#d62728", "--"),
        (1.5, 20.0, 1.0, "SGT(p=1.5, q=20, s=1)", "#2ca02c", "--"),
        (2.0, 2.5, 100.0, "SGT(p=2, q=2.5, s=100)", "#8c564b", "-."),
    ]
    for p, q, s, label, color, ls in sgt_configs:
        ax.plot(x, sgt_loss(x, p, q, s), color=color, ls=ls, lw=1.8, label=label)
    ax.set_xlabel("Residual $x$")
    ax.set_ylabel("Loss $f(x)$")
    ax.set_title("(a) Raw scale")
    ax.set_ylim(0, 30)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True)

    # --- Right panel: normalized (f(x)/f(1)) ---
    ax = axes[1]
    _plot_classical_normed(ax, x_pos, lw=2.0, alpha=0.85)
    for p, q, s, label, color, ls in sgt_configs:
        y = _norm_at_1(lambda t: sgt_loss(t, p, q, s), x_pos)
        ax.plot(x_pos, y, color=color, ls=ls, lw=1.8, label=label)
    ax.set_xlabel("Residual $|x|$")
    ax.set_ylabel("Normalized loss $f(x)/f(1)$")
    ax.set_title("(b) Normalized (shape comparison)")
    ax.set_ylim(0, 30)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(output_dir / "overview.png")
    plt.close(fig)


def _fig_p_sweep(x_pos, output_dir):
    """Figure 2: Effect of p at fixed q=20, s=1 (L1 ↔ L2 interpolation)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_classical_normed(ax, x_pos, names=["MSE", "MAE", "Huber"])

    q, s = 20.0, 1.0
    p_values = [1.0, 1.25, 1.5, 1.75, 2.0]
    cmap = plt.cm.viridis
    for i, p in enumerate(p_values):
        y = _norm_at_1(lambda t, _p=p: sgt_loss(t, _p, q, s), x_pos)
        ax.plot(
            x_pos,
            y,
            color=cmap(i / (len(p_values) - 1)),
            ls="--",
            lw=1.8,
            label=f"SGT(p={p}, q={q}, s={s})",
        )

    ax.set_xlabel("Residual $|x|$")
    ax.set_ylabel("Normalized loss")
    ax.set_title("Effect of $p$ at $q=20,\\; s=1$: L1 $\\leftrightarrow$ L2 interpolation")
    ax.set_ylim(0, 30)
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "p_sweep.png")
    plt.close(fig)


def _fig_q_sweep(x_pos, output_dir):
    """Figure 3: Effect of q at fixed p=2, s=1 (power-law ↔ logarithmic)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_classical_normed(ax, x_pos, names=["MSE", "Cauchy"])

    p, s = 2.0, 1.0
    q_values = [1.3, 2.5, 4.0, 7.0, 12.0, 20.0]
    cmap = plt.cm.plasma
    for i, q in enumerate(q_values):
        y = _norm_at_1(lambda t, _q=q: sgt_loss(t, p, _q, s), x_pos)
        ax.plot(
            x_pos,
            y,
            color=cmap(i / (len(q_values) - 1)),
            ls="--",
            lw=1.8,
            label=f"SGT(p={p}, q={q}, s={s})",
        )

    ax.set_xlabel("Residual $|x|$")
    ax.set_ylabel("Normalized loss")
    ax.set_title("Effect of $q$ at $p=2,\\; s=1$: quadratic $\\leftrightarrow$ logarithmic")
    ax.set_ylim(0, 30)
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_dir / "q_sweep.png")
    plt.close(fig)


def _fig_s_sweep(x_pos, output_dir):
    """Figure 4: Effect of s at fixed (p, q) — shifts the transition point.

    For small q (logarithmic regime) the power-law → log transition happens
    at |x| ≈ s·v·q, so increasing s dramatically delays the dampening.
    For large q (power-law regime) the transition is far out and s has less
    visual impact on the normalized shape.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    s_values = [1.0, 5.0, 10.0, 50.0, 100.0]
    cmap = plt.cm.coolwarm

    configs = [
        (2.0, 2.5, "p=2, q=2.5 (logarithmic regime)"),
        (2.0, 5.0, "p=2, q=5 (intermediate)"),
        (2.0, 20.0, "p=2, q=20 (power-law regime)"),
    ]

    for ax, (p, q, title) in zip(axes, configs):
        _plot_classical_normed(ax, x_pos, names=["MSE", "Cauchy"])
        for i, s in enumerate(s_values):
            y = _norm_at_1(lambda t, _s=s: sgt_loss(t, p, q, _s), x_pos)
            ax.plot(x_pos, y, color=cmap(i / (len(s_values) - 1)), ls="--", lw=1.8, label=f"s={s}")
        ax.set_xlabel("Residual $|x|$")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 30)
        ax.legend(fontsize=8)
        ax.grid(True)

    axes[0].set_ylabel("Normalized loss")
    fig.suptitle(
        "Effect of $\\sigma$ (scale): shifts power-law $\\rightarrow$ log transition",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "s_sweep.png")
    plt.close(fig)


def _fig_sq_interaction(x_pos, output_dir):
    """Figure 5: (s, q) interaction — both push towards pure Lp norm.

    Shows that for large q, increasing s adds little; but for small q,
    increasing s from 1 to 100 transforms the loss from Cauchy-like to
    near-power-law, demonstrating the interplay the team lead noted.
    """
    q_values = [1.3, 2.5, 5.0, 20.0]
    s_values = [1.0, 10.0, 100.0]
    p = 2.0

    fig, axes = plt.subplots(
        len(s_values), len(q_values), figsize=(18, 12), sharex=True, sharey=True
    )

    for i, s in enumerate(s_values):
        for j, q in enumerate(q_values):
            ax = axes[i, j]

            # All classical as thin grey
            for name, spec in CLASSICAL.items():
                y = _norm_at_1(spec["fn"], x_pos)
                ax.plot(x_pos, y, color="grey", ls=":", lw=1.0, alpha=0.5)

            y_sgt = _norm_at_1(lambda t: sgt_loss(t, p, q, s), x_pos)
            ax.plot(x_pos, y_sgt, color="black", lw=2.5, label=f"SGT(p={p}, q={q}, s={s})")

            # Find and highlight closest classical loss
            best_name, best_dist = None, np.inf
            for name, spec in CLASSICAL.items():
                y_cl = _norm_at_1(spec["fn"], x_pos)
                dist = np.mean((y_sgt - y_cl) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_name = name
            if best_name:
                spec = CLASSICAL[best_name]
                y_cl = _norm_at_1(spec["fn"], x_pos)
                ax.plot(
                    x_pos,
                    y_cl,
                    color=spec["color"],
                    ls="-",
                    lw=2.0,
                    alpha=0.8,
                    label=f"{best_name} (closest)",
                )

            ax.set_title(f"q={q}, s={s}", fontsize=10)
            ax.set_ylim(0, 15)
            ax.legend(fontsize=7)
            ax.grid(True)
            if i == len(s_values) - 1:
                ax.set_xlabel("$|x|$")
            if j == 0:
                ax.set_ylabel("Normalized loss")

    fig.suptitle(
        f"$(s, q)$ interaction at $p={p}$: both push towards pure L$_p$ norm",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "sq_interaction.png")
    plt.close(fig)


def _fig_interpolation_grid(x_pos, output_dir):
    """Figure 6: Full (p, q) grid at s=1 showing interpolation landscape.

    q=1.3 with p=1.0 is invalid (q^p < 2/p), so that cell is left blank.
    """
    p_values = [1.0, 1.5, 2.0]
    q_values = [1.3, 2.5, 5.0, 20.0]
    s = 1.0

    fig, axes = plt.subplots(
        len(p_values), len(q_values), figsize=(18, 12), sharex=True, sharey=True
    )

    for i, p in enumerate(p_values):
        for j, q in enumerate(q_values):
            ax = axes[i, j]

            # q=1.3, p=1.0 violates constraint q^p > 2/p
            if q**p <= 2.0 / p:
                ax.text(
                    0.5,
                    0.5,
                    f"N/A\n$q^p={q**p:.2f} < 2/p={2/p:.1f}$",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="grey",
                )
                ax.set_title(f"p={p}, q={q}", fontsize=10, color="grey")
                ax.grid(True)
                if i == len(p_values) - 1:
                    ax.set_xlabel("$|x|$")
                if j == 0:
                    ax.set_ylabel("Normalized loss")
                continue

            for name, spec in CLASSICAL.items():
                y = _norm_at_1(spec["fn"], x_pos)
                ax.plot(x_pos, y, color="grey", ls=":", lw=1.0, alpha=0.5)

            y_sgt = _norm_at_1(lambda t: sgt_loss(t, p, q, s), x_pos)
            ax.plot(x_pos, y_sgt, color="black", lw=2.5, label=f"SGT(p={p}, q={q}, s={s})")

            best_name, best_dist = None, np.inf
            for name, spec in CLASSICAL.items():
                y_cl = _norm_at_1(spec["fn"], x_pos)
                dist = np.mean((y_sgt - y_cl) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_name = name
            if best_name:
                spec = CLASSICAL[best_name]
                y_cl = _norm_at_1(spec["fn"], x_pos)
                ax.plot(
                    x_pos,
                    y_cl,
                    color=spec["color"],
                    ls="-",
                    lw=2.0,
                    alpha=0.8,
                    label=f"{best_name} (closest)",
                )

            ax.set_title(f"p={p}, q={q}", fontsize=10)
            ax.set_ylim(0, 15)
            ax.legend(fontsize=7)
            ax.grid(True)
            if i == len(p_values) - 1:
                ax.set_xlabel("$|x|$")
            if j == 0:
                ax.set_ylabel("Normalized loss")

    fig.suptitle("SGT interpolation grid ($\\sigma=1$, $\\lambda=0$)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "interpolation_grid.png")
    plt.close(fig)


def _fig_tukey_comparison(output_dir):
    """Figure 7: SGT at small q vs Tukey — tail behavior comparison.

    Tukey loss is bounded (saturates at c²/6 for |x| > c), while SGT always
    grows logarithmically.  At very small q (≈1.2–1.5) the SGT growth is so
    slow that it approximates Tukey's practical behavior over a wide range.

    Uses a wider x-range (0–15) to clearly show tail saturation / growth.
    """
    x_wide = np.linspace(0.01, 15.0, 2000)

    # --- Left panel: raw loss values ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    tukey_c = 4.685
    y_tukey = tukey_loss(x_wide, c=tukey_c)
    ax.plot(x_wide, y_tukey, color="#9467bd", lw=2.5, label="Tukey (c=4.685)")
    ax.axhline(tukey_c**2 / 6, color="#9467bd", ls=":", lw=1.0, alpha=0.5)

    q_values = [1.2, 1.3, 1.5, 2.0, 2.5]
    p, s = 2.0, 1.0
    cmap = plt.cm.plasma
    for i, q in enumerate(q_values):
        y = sgt_loss(x_wide, p, q, s)
        ax.plot(
            x_wide,
            y,
            color=cmap(i / (len(q_values) - 1)),
            ls="--",
            lw=1.8,
            label=f"SGT(p=2, q={q}, s=1)",
        )

    ax.set_xlabel("Residual $|x|$")
    ax.set_ylabel("Loss $f(x)$")
    ax.set_title("(a) Raw loss: Tukey vs SGT at small $q$")
    ax.set_ylim(0, 12)
    ax.legend(fontsize=8)
    ax.grid(True)

    # --- Right panel: normalized by f(1) ---
    ax = axes[1]
    y_tukey_n = _norm_at_1(lambda t: tukey_loss(t, c=tukey_c), x_wide)
    ax.plot(x_wide, y_tukey_n, color="#9467bd", lw=2.5, label="Tukey (c=4.685)")

    for i, q in enumerate(q_values):
        y = _norm_at_1(lambda t, _q=q: sgt_loss(t, p, _q, s), x_wide)
        ax.plot(
            x_wide,
            y,
            color=cmap(i / (len(q_values) - 1)),
            ls="--",
            lw=1.8,
            label=f"SGT(p=2, q={q}, s=1)",
        )

    ax.set_xlabel("Residual $|x|$")
    ax.set_ylabel("Normalized loss $f(x)/f(1)$")
    ax.set_title("(b) Normalized shape: tail growth comparison")
    ax.set_ylim(0, 15)
    ax.legend(fontsize=8)
    ax.grid(True)

    fig.suptitle(
        "Tukey (bounded) vs SGT (log-growth): smaller $q$ $\\rightarrow$ slower tails",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "tukey_comparison.png")
    plt.close(fig)


if __name__ == "__main__":
    app()
