"""Data-driven SGT parameter guidance (reviewer A3 / A11).

Fits the SGT skew and tail parameters ``(lambda, q)`` by maximum likelihood to the
one-step increments of each training dataset, with ``p`` and the residual scale
fixed exactly as the training loss fixes them (``metrics.sgt_increment_fit``). The
fitted ``lambda`` is the asymmetry the data actually support *at the forecast
horizon*: about 0.5 on ``heavy-tailed-skewed`` but ~0 on the smoothed
``normal-skewed`` data, whose marginal skew is averaged away by the kernel.
``delta_nll < 0`` means the skewed fit beats the symmetric one.

Synthetic datasets are regenerated with the training settings (smoothing +
per-sequence standardization) into ``diagnostic_synthetic_dataset.npy`` so the real
``synthetic_dataset.npy`` is never clobbered; real datasets are read from their
processed ``.npy`` files when present. Writes ``reports/increment_sgt_fit.csv``.
"""

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from skewed_sequences.config import PROCESSED_DATA_DIR, REPORTS_DIR, SYNTHETIC_DATA_CONFIGS
from skewed_sequences.data.synthetic.generate_data import main as generate_data_main
from skewed_sequences.metrics import sgt_increment_fit

app = typer.Typer(pretty_exceptions_show_locals=False)

REAL_DATASETS = {
    "covid-owid": PROCESSED_DATA_DIR / "dataset.npy",
    "rvr-us (last processed series)": PROCESSED_DATA_DIR / "rvr_us_data.npy",
}


def fit_row(label: str, data: np.ndarray, p: float = 2.0) -> dict:
    """One CSV row: dataset label + the ``sgt_increment_fit`` result."""
    return {"dataset": label, **sgt_increment_fit(data, p=p)}


@app.command()
def main(
    output_path: Path = REPORTS_DIR / "increment_sgt_fit.csv",
    p: float = 2.0,
    n_sequences: int = 1000,
    skip_synthetic: bool = False,
    skip_real: bool = False,
):
    rows = []

    if not skip_synthetic:
        diagnostic_path = PROCESSED_DATA_DIR / "diagnostic_synthetic_dataset.npy"
        for cfg in SYNTHETIC_DATA_CONFIGS:
            typer.echo(f"Generating {cfg['experiment_name']} (training settings)...")
            generate_data_main(
                output_path=diagnostic_path,
                lam=cfg["lam"],
                q=cfg["q"],
                sigma=cfg["sigma"],
                kernel_size=cfg.get("kernel_size", 99),
                n_sequences=n_sequences,
            )
            rows.append(fit_row(cfg["experiment_name"], np.load(diagnostic_path), p))

    if not skip_real:
        for label, path in REAL_DATASETS.items():
            if not path.exists():
                logger.warning(f"{label}: {path.name} not found, skipping")
                continue
            try:
                rows.append(fit_row(label, np.load(path), p))
            except ValueError as e:
                logger.warning(f"{label}: {e}")

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    if not df.empty:
        typer.echo(df.round(4).to_string(index=False))
    typer.echo(f"Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    app()
