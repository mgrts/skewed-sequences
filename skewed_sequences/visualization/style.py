"""Shared plotting style for the skewed-sequences project."""

import matplotlib.pyplot as plt
import seaborn as sns

DPI = 150

COLORS = {
    "ground_truth": "#2d3436",
    "prediction": "#0984e3",
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "context_boundary": "#636e72",
}

PALETTE_SEQ = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
]


def apply_style() -> None:
    """Apply the project-wide plotting style."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.3,
        }
    )
