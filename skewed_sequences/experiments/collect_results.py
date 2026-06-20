"""Collect experiment results from MLflow into a CSV file."""

from datetime import datetime, timezone
from pathlib import Path
import re

from mlflow import MlflowClient
from mlflow.entities import ViewType
import pandas as pd
import typer

from skewed_sequences.config import REPORTS_DIR, TRACKING_URI
from skewed_sequences.mlflow_contract import ALL_SUMMARY_METRIC_KEYS

app = typer.Typer(pretty_exceptions_show_locals=False)


def _derive_dataset(experiment_name: str) -> str:
    """Derive dataset name from MLflow experiment name.

    Experiment names follow patterns like:
    - "normal_run_1" -> "normal"
    - "heavy-tailed-skewed_run_3" -> "heavy-tailed-skewed"
    - "exp-normal_run_1" -> "exp-normal"
    - "covid-owid_run_1" -> "covid-owid"
    - "lanl_sgt_run_1" -> "lanl"
    - "rvr-us-bed-occupancy_run_1" -> "rvr-us-bed-occupancy"
    """
    name = re.sub(r"_run_\d+$", "", experiment_name)
    if name.startswith("lanl_"):
        return "lanl"
    return name


def collect_experiment_results(
    tracking_uri: str = TRACKING_URI,
    since: datetime | None = None,
) -> pd.DataFrame:
    """Collect all experiment results from MLflow into a DataFrame.

    Returns a DataFrame with columns:
        run_name, experiment_name, created_at, status, random_state,
        model_type, context_length, output_length, stride, embed_dim, num_heads,
        num_layers, exp_transform, dataset, loss_type, sgt_loss_lambda, sgt_loss_q,
        sgt_loss_sigma, sgt_loss_p, best_train_smape, best_val_smape, best_test_smape
    """
    client = MlflowClient(tracking_uri=tracking_uri)

    experiments = client.search_experiments(view_type=ViewType.ALL)
    exp_id_to_name = {exp.experiment_id: exp.name for exp in experiments}

    filter_parts = []
    if since is not None:
        since_ms = int(since.timestamp() * 1000)
        filter_parts.append(f"attributes.start_time >= {since_ms}")
    filter_string = " AND ".join(filter_parts) if filter_parts else ""

    rows = []
    for experiment_id, exp_name in exp_id_to_name.items():
        page_token = None
        while True:
            page = client.search_runs(
                experiment_ids=[experiment_id],
                run_view_type=ViewType.ALL,
                filter_string=filter_string,
                page_token=page_token,
            )
            for run in page:
                params = run.data.params
                metrics = run.data.metrics
                row = {
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
                    "embed_dim": params.get("embed_dim"),
                    "num_heads": params.get("num_heads"),
                    "num_layers": params.get("num_layers"),
                    "exp_transform": params.get("exp_transform"),
                    "dataset": _derive_dataset(exp_name),
                    "loss_type": params.get("loss_type"),
                    "sgt_loss_lambda": params.get("sgt_loss_lambda"),
                    "sgt_loss_q": params.get("sgt_loss_q"),
                    "sgt_loss_sigma": params.get("sgt_loss_sigma"),
                    "sgt_loss_p": params.get("sgt_loss_p"),
                    "residual_scale": params.get("residual_scale"),
                }
                # All summary keys: best_{split}_{smape,mape,rmse,mae} + persistence
                # baseline (best_test_naive_*) + best_test_mase.
                for key in ALL_SUMMARY_METRIC_KEYS:
                    row[key] = metrics.get(key)
                rows.append(row)
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
            "embed_dim",
            "num_heads",
            "num_layers",
            "sgt_loss_lambda",
            "sgt_loss_q",
            "sgt_loss_sigma",
            "sgt_loss_p",
            "residual_scale",
            *ALL_SUMMARY_METRIC_KEYS,
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@app.command()
def main(
    output_path: Path = REPORTS_DIR / "experiment_results.csv",
    since: str | None = typer.Option(
        None, help="Only include runs started on or after this date (YYYY-MM-DD)."
    ),
):
    """Collect all MLflow experiment results and save to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    since_dt = None
    if since is not None:
        since_dt = datetime.strptime(since, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    typer.echo("Collecting experiment results from MLflow...")
    df = collect_experiment_results(since=since_dt)
    df.to_csv(output_path, index=False)
    typer.echo(f"Saved {len(df)} runs to {output_path}")


if __name__ == "__main__":
    app()
