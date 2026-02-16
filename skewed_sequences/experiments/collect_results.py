"""Collect experiment results from MLflow into a CSV file."""

from datetime import datetime, timezone
from pathlib import Path
import re

from mlflow import MlflowClient
from mlflow.entities import ViewType
import pandas as pd
import typer

from skewed_sequences.config import REPORTS_DIR, TRACKING_URI

app = typer.Typer(pretty_exceptions_show_locals=False)


def _derive_dataset(experiment_name: str) -> str:
    """Derive dataset name from MLflow experiment name.

    Experiment names follow patterns like:
    - "normal_run_1" -> "normal"
    - "heavy-tailed-skewed_run_3" -> "heavy-tailed-skewed"
    - "covid-owid_run_1" -> "covid-owid"
    - "lanl_sgt_run_1" -> "lanl"
    - "rvr-us-bed-occupancy_run_1" -> "rvr-us-bed-occupancy"
    """
    name = re.sub(r"_run_\d+$", "", experiment_name)
    if name.startswith("lanl_"):
        return "lanl"
    return name


def collect_experiment_results(tracking_uri: str = TRACKING_URI) -> pd.DataFrame:
    """Collect all experiment results from MLflow into a DataFrame.

    Returns a DataFrame with columns:
        run_name, experiment_name, created_at, status, random_state,
        model_type, context_length, output_length, stride, dataset,
        loss_type, sgt_loss_lambda, sgt_loss_q, sgt_loss_sigma, sgt_loss_p,
        best_train_smape, best_val_smape, best_test_smape
    """
    client = MlflowClient(tracking_uri=tracking_uri)

    experiments = client.search_experiments(view_type=ViewType.ALL)
    exp_id_to_name = {exp.experiment_id: exp.name for exp in experiments}

    rows = []
    for experiment_id, exp_name in exp_id_to_name.items():
        page_token = None
        while True:
            page = client.search_runs(
                experiment_ids=[experiment_id],
                run_view_type=ViewType.ALL,
                page_token=page_token,
            )
            for run in page:
                params = run.data.params
                metrics = run.data.metrics
                rows.append(
                    {
                        "run_name": run.info.run_name,
                        "experiment_name": exp_name,
                        "created_at": datetime.fromtimestamp(
                            run.info.start_time / 1000, tz=timezone.utc
                        ),
                        "status": run.info.status,
                        "random_state": params.get("random_state", params.get("seed")),
                        "model_type": params.get("model_type"),
                        "context_length": params.get("context_length"),
                        "output_length": params.get("output_length"),
                        "stride": params.get("stride"),
                        "dataset": _derive_dataset(exp_name),
                        "loss_type": params.get("loss_type"),
                        "sgt_loss_lambda": params.get("sgt_loss_lambda"),
                        "sgt_loss_q": params.get("sgt_loss_q"),
                        "sgt_loss_sigma": params.get("sgt_loss_sigma"),
                        "sgt_loss_p": params.get("sgt_loss_p"),
                        "best_train_smape": metrics.get("best_train_smape"),
                        "best_val_smape": metrics.get("best_val_smape"),
                        "best_test_smape": metrics.get("best_test_smape"),
                    }
                )
            page_token = page.token if hasattr(page, "token") else None
            if not page_token:
                break

    df = pd.DataFrame(rows)

    if not df.empty:
        numeric_cols = [
            "random_state",
            "context_length",
            "output_length",
            "stride",
            "sgt_loss_lambda",
            "sgt_loss_q",
            "sgt_loss_sigma",
            "sgt_loss_p",
            "best_train_smape",
            "best_val_smape",
            "best_test_smape",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@app.command()
def main(
    output_path: Path = REPORTS_DIR / "experiment_results.csv",
):
    """Collect all MLflow experiment results and save to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    typer.echo("Collecting experiment results from MLflow...")
    df = collect_experiment_results()
    df.to_csv(output_path, index=False)
    typer.echo(f"Saved {len(df)} runs to {output_path}")


if __name__ == "__main__":
    app()
