from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import typer

from skewed_sequences.config import FIGURES_DIR, PROCESSED_DATA_DIR, SYNTHETIC_DATA_CONFIGS
from skewed_sequences.data.rvr_us.dataset import main as create_rvr_dataset_main
from skewed_sequences.data.synthetic.generate_data import main as generate_data_main
from skewed_sequences.visualization.style import PALETTE_SEQ, apply_style

app = typer.Typer(pretty_exceptions_show_locals=False)


def create_boxplot(data_dict: dict, output_path: Path, title: str, xlim: int):
    # Sized for a single journal column (3.35 in) at 300 dpi with >= 8 pt type, so the
    # figure is legible at print size without scaling (reviewers A9 / B7).
    from skewed_sequences.visualization.results_figures import _style

    _style()
    plt.figure(figsize=(3.35, 0.45 * len(data_dict) + 0.9))
    sns.boxplot(
        data=list(data_dict.values()),
        orient="h",
        palette=PALETTE_SEQ[: len(data_dict)],
        width=0.6,
        linewidth=0.8,
        fliersize=1.2,
    )

    plt.yticks(ticks=range(len(data_dict)), labels=list(data_dict.keys()))
    if title:
        plt.title(title, loc="left")
    plt.xlabel("Standardized value")
    plt.ylabel("")
    sns.despine(trim=True)

    plt.xticks(ticks=np.arange(-xlim, xlim + 1, 1))
    plt.xlim(-xlim, xlim)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    logger.success(f"Boxplots saved to {output_path}")


@app.command()
def synthetic(
    output_path: Path = FIGURES_DIR / "dataset_boxplots.png",
    sample_size: int = 1000,
    xlim: int = 5,
):
    apply_style()
    logger.info("Generating synthetic datasets and creating boxplots...")

    import copy

    dataset_params = copy.deepcopy(SYNTHETIC_DATA_CONFIGS)
    for i, params in enumerate(dataset_params):
        params["label"] = {
            "normal": "Gaussian",
            "heavy-tailed": "Heavy-tailed",
            "normal-skewed": "Gaussian, skewed",
            "heavy-tailed-skewed": "Heavy-tailed, skewed",
        }.get(params["experiment_name"], params["experiment_name"])

    datasets = {}

    diagnostic_path = PROCESSED_DATA_DIR / "diagnostic_synthetic_dataset.npy"
    for params in dataset_params:
        generate_data_main(
            output_path=diagnostic_path,
            lam=params["lam"],
            q=params["q"],
            sigma=params["sigma"],
            n_sequences=sample_size,
            apply_smoothing=False,
            standardize=False,  # raw generative distribution; don't clobber training data
        )
        data = np.load(diagnostic_path).flatten()
        datasets[params["label"]] = data
        logger.info(f'Generated dataset: {params["label"]}')

    create_boxplot(datasets, output_path, "", xlim)


@app.command()
def real(
    output_path: Path = FIGURES_DIR / "real_dataset_boxplots.png",
    xlim: int = 5,
):
    apply_style()
    logger.info("Creating real-world datasets and generating boxplots...")

    dataset_specs = [
        {"label": "RVR bed occupancy", "time_series": "average_inpatient_beds_occupied"},
        {
            "label": "RVR influenza admissions",
            "time_series": "total_admissions_all_influenza_confirmed_past_7days",
        },
        {"label": "OWID COVID-19 cases", "path": PROCESSED_DATA_DIR / "dataset.npy"},
    ]

    datasets = {}

    for spec in dataset_specs:
        label = spec["label"]
        if "time_series" in spec:
            try:
                create_rvr_dataset_main(time_series=spec["time_series"])
                data = np.load(PROCESSED_DATA_DIR / "rvr_us_data.npy").flatten()
                datasets[label] = data
                logger.info(f"Created and loaded dataset: {label}")
            except Exception as e:
                logger.warning(f"Failed to process {label}: {e}")
        else:
            try:
                data = np.load(spec["path"]).flatten()
                datasets[label] = data
                logger.info(f"Loaded dataset: {label}")
            except Exception as e:
                logger.warning(f"Failed to load {label}: {e}")

    create_boxplot(datasets, output_path, "", xlim)


if __name__ == "__main__":
    app()
