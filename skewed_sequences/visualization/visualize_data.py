"""Visualize dataset samples for synthetic and real-world datasets.

Generates individual figures per observation, organized into a directory
hierarchy::

    reports/figures/dataset_samples/
    └── synthetic/
        └── <variant>/            # default, exp, gaussian, etc.
            └── <data_type>/      # normal, heavy-tailed, etc.
                ├── sample_001.png
                ├── sample_002.png
                └── ...

    reports/figures/dataset_samples/
    └── real/
        └── <dataset_name>/
            ├── sample_001.png
            └── ...

Usage::

    poetry run skseq visualize synthetic
    poetry run skseq visualize real
    poetry run skseq visualize variants
"""

from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import typer

from skewed_sequences.config import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    SEED,
    SEQUENCE_LENGTH,
    SYNTHETIC_DATA_CONFIGS,
)
from skewed_sequences.data.synthetic.generate_data import (
    SkewedGeneralizedT,
    smooth_sequence,
)
from skewed_sequences.visualization.style import COLORS, apply_style

app = typer.Typer(pretty_exceptions_show_locals=False)

# ---------------------------------------------------------------------------
# Generation variant definitions
# ---------------------------------------------------------------------------

SMOOTHING_VARIANTS = [
    {
        "name": "default",
        "label": "Combined cosine-gauss (k=99, σ=10, T=30)",
        "apply_smoothing": True,
        "smoothing_type": "combined_cosine_gaussian",
        "kernel_size": 99,
        "kernel_sigma": 10.0,
        "period": 30.0,
        "exp_transform": False,
    },
    {
        "name": "low_smoothing",
        "label": "Low smoothing (k=31, σ=3, T=30)",
        "apply_smoothing": True,
        "smoothing_type": "combined_cosine_gaussian",
        "kernel_size": 31,
        "kernel_sigma": 3.0,
        "period": 30.0,
        "exp_transform": False,
    },
    {
        "name": "gaussian_narrow",
        "label": "Narrow Gaussian (k=15, σ=2)",
        "apply_smoothing": True,
        "smoothing_type": "gaussian",
        "kernel_size": 15,
        "kernel_sigma": 2.0,
        "period": 30.0,
        "exp_transform": False,
    },
    {
        "name": "no_smoothing",
        "label": "No smoothing (raw SGT)",
        "apply_smoothing": False,
        "smoothing_type": "combined_cosine_gaussian",
        "kernel_size": 99,
        "kernel_sigma": 10.0,
        "period": 30.0,
        "exp_transform": False,
    },
    {
        "name": "exp_default",
        "label": "exp(X) + default smoothing",
        "apply_smoothing": True,
        "smoothing_type": "combined_cosine_gaussian",
        "kernel_size": 99,
        "kernel_sigma": 10.0,
        "period": 30.0,
        "exp_transform": True,
        "exp_scale": 0.1,
    },
    {
        "name": "exp_low_smoothing",
        "label": "exp(X) + low smoothing (k=31, σ=3)",
        "apply_smoothing": True,
        "smoothing_type": "combined_cosine_gaussian",
        "kernel_size": 31,
        "kernel_sigma": 3.0,
        "period": 30.0,
        "exp_transform": True,
        "exp_scale": 0.1,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_sequences(
    n_sequences: int,
    sequence_length: int,
    mu: float,
    sigma: float,
    lam: float,
    p: float,
    q: float,
    apply_smoothing: bool,
    smoothing_type: str,
    kernel_size: int,
    kernel_sigma: float,
    period: float,
    exp_transform: bool = False,
    exp_scale: float = 0.1,
    seed: int = SEED,
) -> np.ndarray:
    """Generate synthetic sequences in-memory (no file I/O)."""
    np.random.seed(seed)
    sgt = SkewedGeneralizedT(mu=mu, sigma=sigma, lam=lam, p=p, q=q)
    raw = sgt.generate_sequences(n_sequences, sequence_length, n_features=1)

    dataset = np.zeros_like(raw)
    for i in range(n_sequences):
        if apply_smoothing:
            dataset[i, :, 0] = smooth_sequence(
                raw[i, :, 0],
                smoothing_type,
                kernel_size,
                kernel_sigma,
                period,
            )
        else:
            dataset[i, :, 0] = raw[i, :, 0]

    if exp_transform:
        for i in range(n_sequences):
            seq = dataset[i, :, 0]
            std = np.std(seq)
            if std > 0:
                seq = (seq - np.mean(seq)) / std
            else:
                seq = seq - np.mean(seq)
            dataset[i, :, 0] = np.exp(exp_scale * seq)

    return dataset


def _save_individual_samples(
    data: np.ndarray,
    n_samples: int,
    output_dir: Path,
    title_prefix: str,
    seed: int = SEED,
) -> int:
    """Save one PNG per sample sequence into *output_dir*.

    Returns the number of images saved.
    """
    if data.ndim == 3:
        data = data[:, :, 0]

    rng = np.random.RandomState(seed)
    n_total = data.shape[0]
    n_to_plot = min(n_samples, n_total)
    indices = rng.choice(n_total, size=n_to_plot, replace=False)

    output_dir.mkdir(parents=True, exist_ok=True)

    for rank, idx in enumerate(indices, start=1):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(data[idx], linewidth=0.9, color=COLORS["primary"])
        ax.set_title(f"{title_prefix} — sample {rank:03d}", fontsize=11)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Value")
        fig.tight_layout()

        path = output_dir / f"sample_{rank:03d}.png"
        fig.savefig(path)
        plt.close(fig)

    logger.info(f"  {n_to_plot} images → {output_dir}")
    return n_to_plot


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


@app.command()
def synthetic(
    n_samples: int = 10,
    output_dir: Path = FIGURES_DIR / "dataset_samples" / "synthetic" / "default",
    seed: int = SEED,
):
    """Visualize samples for each of the 4 standard SYNTHETIC_DATA_CONFIGS.

    Uses the default generation parameters (combined cosine-gaussian kernel).
    Saves individual images into::

        <output_dir>/<data_type>/sample_001.png
    """
    apply_style()
    logger.info(
        f"Generating sample visualizations for {len(SYNTHETIC_DATA_CONFIGS)} "
        f"synthetic configs (default variant)"
    )

    total = 0
    for cfg in SYNTHETIC_DATA_CONFIGS:
        ds_name = cfg["experiment_name"]
        logger.info(f"Generating: {ds_name}")
        data = _generate_sequences(
            n_sequences=max(n_samples * 5, 50),
            sequence_length=SEQUENCE_LENGTH,
            mu=0.0,
            sigma=cfg["sigma"],
            lam=cfg["lam"],
            p=2.0,
            q=cfg["q"],
            apply_smoothing=True,
            smoothing_type="combined_cosine_gaussian",
            kernel_size=99,
            kernel_sigma=10.0,
            period=30.0,
            seed=seed,
        )
        total += _save_individual_samples(
            data,
            n_samples,
            output_dir=output_dir / ds_name,
            title_prefix=f"synthetic/{ds_name} (λ={cfg['lam']}, q={cfg['q']}, σ={cfg['sigma']})",
            seed=seed,
        )

    logger.success(f"Saved {total} images across {len(SYNTHETIC_DATA_CONFIGS)} data types.")


@app.command()
def real(
    n_samples: int = 10,
    output_dir: Path = FIGURES_DIR / "dataset_samples" / "real",
    seed: int = SEED,
):
    """Visualize samples for each available real-world dataset (.npy files).

    Saves individual images into::

        <output_dir>/<dataset_name>/sample_001.png
    """
    apply_style()
    dataset_specs = [
        {
            "name": "lanl_earthquake",
            "path": PROCESSED_DATA_DIR / "lanl_sequences.npy",
            "label": "LANL Earthquake Acoustic",
        },
        {
            "name": "owid_covid",
            "path": PROCESSED_DATA_DIR / "dataset.npy",
            "label": "OWID COVID New Cases",
        },
        {
            "name": "rvr_us",
            "path": PROCESSED_DATA_DIR / "rvr_us_data.npy",
            "label": "RVR US Hospitalization",
        },
        {
            "name": "health_fitness",
            "path": PROCESSED_DATA_DIR / "health_fitness_data.npy",
            "label": "Health Fitness Tracker",
        },
    ]

    total = 0
    for spec in dataset_specs:
        path = spec["path"]
        if not path.exists():
            logger.warning(f"Dataset not found, skipping: {path}")
            continue

        logger.info(f"Loading: {spec['name']}")
        data = np.load(path)
        total += _save_individual_samples(
            data,
            n_samples,
            output_dir=output_dir / spec["name"],
            title_prefix=f"real/{spec['name']}",
            seed=seed,
        )

    logger.success(f"Saved {total} real-world sample images.")


@app.command()
def variants(
    n_samples: int = 10,
    output_dir: Path = FIGURES_DIR / "dataset_samples" / "synthetic",
    seed: int = SEED,
):
    """Generate and visualize multiple smoothing/transform variants.

    Creates individual images organized as::

        <output_dir>/<variant>/<data_type>/sample_001.png

    For example::

        reports/figures/dataset_samples/synthetic/exp_default/normal/sample_001.png
    """
    apply_style()
    logger.info(
        f"Generating {len(SMOOTHING_VARIANTS)} variants × "
        f"{len(SYNTHETIC_DATA_CONFIGS)} data configs × "
        f"{n_samples} samples each"
    )

    grand_total = 0
    for variant in SMOOTHING_VARIANTS:
        var_name = variant["name"]
        for cfg in SYNTHETIC_DATA_CONFIGS:
            ds_name = cfg["experiment_name"]
            logger.info(f"  {var_name} / {ds_name}")

            data = _generate_sequences(
                n_sequences=max(n_samples * 5, 50),
                sequence_length=SEQUENCE_LENGTH,
                mu=0.0,
                sigma=cfg["sigma"],
                lam=cfg["lam"],
                p=2.0,
                q=cfg["q"],
                apply_smoothing=variant["apply_smoothing"],
                smoothing_type=variant["smoothing_type"],
                kernel_size=variant["kernel_size"],
                kernel_sigma=variant["kernel_sigma"],
                period=variant["period"],
                exp_transform=variant.get("exp_transform", False),
                exp_scale=variant.get("exp_scale", 0.1),
                seed=seed,
            )
            grand_total += _save_individual_samples(
                data,
                n_samples,
                output_dir=output_dir / var_name / ds_name,
                title_prefix=f"{var_name}/{ds_name} — {variant['label']}",
                seed=seed,
            )

    logger.success(
        f"All variant visualizations complete: {grand_total} images across "
        f"{len(SMOOTHING_VARIANTS)} variants × {len(SYNTHETIC_DATA_CONFIGS)} data types."
    )

    # ------------------------------------------------------------------
    # Summary figures: one per data type, all variants as subplots,
    # max 3 observations per subplot for readability.
    # ------------------------------------------------------------------
    summary_n = min(3, n_samples)
    logger.info("Generating summary figures (max 3 observations per subplot)…")

    for cfg in SYNTHETIC_DATA_CONFIGS:
        ds_name = cfg["experiment_name"]
        n_variants = len(SMOOTHING_VARIANTS)
        fig, axes = plt.subplots(n_variants, 1, figsize=(16, 3.5 * n_variants), sharex=True)
        if n_variants == 1:
            axes = [axes]

        rng = np.random.RandomState(seed)

        for ax, variant in zip(axes, SMOOTHING_VARIANTS):
            data = _generate_sequences(
                n_sequences=max(n_samples * 5, 50),
                sequence_length=SEQUENCE_LENGTH,
                mu=0.0,
                sigma=cfg["sigma"],
                lam=cfg["lam"],
                p=2.0,
                q=cfg["q"],
                apply_smoothing=variant["apply_smoothing"],
                smoothing_type=variant["smoothing_type"],
                kernel_size=variant["kernel_size"],
                kernel_sigma=variant["kernel_sigma"],
                period=variant["period"],
                exp_transform=variant.get("exp_transform", False),
                exp_scale=variant.get("exp_scale", 0.1),
                seed=seed,
            )
            if data.ndim == 3:
                data = data[:, :, 0]

            indices = rng.choice(data.shape[0], size=min(summary_n, data.shape[0]), replace=False)
            for idx in indices:
                ax.plot(data[idx], alpha=0.8, linewidth=0.8)
            ax.set_title(variant["label"], fontsize=10)
            ax.set_ylabel("Value")
            ax.grid(True)

        axes[-1].set_xlabel("Timestep")
        fig.suptitle(
            f"Generation variants — {ds_name} (λ={cfg['lam']}, q={cfg['q']})",
            fontsize=13,
        )
        fig.tight_layout()

        summary_path = output_dir / f"summary_{ds_name}.png"
        fig.savefig(summary_path)
        plt.close(fig)
        logger.success(f"Summary grid saved: {summary_path}")

    logger.success("Summary figures complete.")


if __name__ == "__main__":
    app()
