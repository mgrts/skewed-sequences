"""Shared scaffolding for the grid runners.

Every grid runner (synthetic / lanl / owid / rvr / head / lambda) loops over a
config list and calls ``train.main`` once per (config, run), expanding the SGT loss
params only for ``loss_type == "sgt"``. That call lived as a duplicated if/else
block in each runner; it lives here once.

Resume support (``resume=True``): a sweep that was killed mid-way (the June-2026
JupyterHub run died in its OWID stage) can be re-launched with the same command.
``draw_experiment_seed`` reuses the ``random_state`` already logged under an
experiment name, so configs appended later stay seed-paired with the finished
ones, and ``run_training_config`` skips a config that already has a FINISHED run
with the same loss params / seed / architecture in that experiment. Both only
*read* the MLflow store at ``config.TRACKING_URI``; the writes are ``train.main``'s.
"""

from pathlib import Path
import random

from mlflow import MlflowClient
from mlflow.entities import ViewType
import typer

from skewed_sequences.config import CONTEXT_LENGTH, OUTPUT_LENGTH, STRIDE, TRACKING_URI
from skewed_sequences.modeling.train import main as train_main

# Params logged by train.main that, when a runner passes them, must also match for a
# run to count as "already done": the architecture (the head study varies it) and the
# training budget (a 1-epoch smoke run must never be mistaken for a finished sweep run).
_ARCH_MATCH_KEYS = ("model_type", "embed_dim", "num_heads", "num_layers")
_TRAIN_MATCH_KEYS = ("batch_size", "num_epochs", "early_stopping_patience")
_EXTRA_MATCH_KEYS = _ARCH_MATCH_KEYS + _TRAIN_MATCH_KEYS
# Upper bound on runs read back per experiment (36-40 per experiment in practice).
_MAX_RUNS_PER_EXPERIMENT = 5000


def _client() -> MlflowClient:
    return MlflowClient(tracking_uri=TRACKING_URI)


def _param_filter(params: dict, status: str | None = "FINISHED") -> str:
    """MLflow ``filter_string`` matching every param.

    MLflow stores param values as ``str(value)`` (``2.5`` -> ``'2.5'``, ``0.0`` ->
    ``'0.0'``), so the same stringification is applied here.
    """
    parts = [f"params.{key} = '{value}'" for key, value in params.items()]
    if status is not None:
        parts.append(f"attributes.status = '{status}'")
    return " and ".join(parts)


def _experiment_id(client: MlflowClient, experiment_name: str) -> str | None:
    experiment = client.get_experiment_by_name(experiment_name)
    return None if experiment is None else experiment.experiment_id


def logged_experiment_param(experiment_names, model_type: str, key: str) -> str | None:
    """Value of param ``key`` logged under any of ``experiment_names`` for ``model_type``.

    Scans every run of the experiment (not just the newest) and raises if the runs
    disagree, so a resumed sweep can never silently attach to an experiment that
    already mixes two seeds / strides. Returns ``None`` when nothing is logged.
    """
    names = [experiment_names] if isinstance(experiment_names, str) else list(experiment_names)
    client = _client()
    for name in names:
        experiment_id = _experiment_id(client, name)
        if experiment_id is None:
            continue
        runs = client.search_runs(
            [experiment_id],
            filter_string=_param_filter({"model_type": model_type}, status=None),
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=_MAX_RUNS_PER_EXPERIMENT,
        )
        values = {run.data.params[key] for run in runs if key in run.data.params}
        if len(values) > 1:
            raise RuntimeError(
                f"experiment {name!r} holds {len(values)} distinct values of {key!r} "
                f"({sorted(values)}); it is not a clean seed-paired block. Use a fresh "
                "experiment name (or tracking DB) instead of resuming into it."
            )
        if values:
            return values.pop()
    return None


def logged_experiment_seed(experiment_names, model_type: str) -> int | None:
    """``random_state`` already logged under any of ``experiment_names`` for ``model_type``."""
    seed = logged_experiment_param(experiment_names, model_type, "random_state")
    return None if seed is None else int(seed)


def draw_experiment_seed(
    experiment_names, model_type: str, resume: bool = True, rng=random
) -> int:
    """One seed per (run_idx, model_type), shared across every loss config.

    Runners draw it ONCE per experiment name and reuse it for every loss, so the
    replicates are seed-paired across loss types (the paired Wilcoxon in
    ``aggregate_results`` relies on this). With ``resume`` the seed already logged
    under the experiment is reused, so configs appended later (a resumed sweep,
    the lambda sub-sweep) pair with the existing runs instead of opening a second,
    unpaired seed. Python ``random`` is never seeded (CLAUDE.md #7): a fresh draw
    is reproducible only through the ``random_state`` MLflow logs.
    """
    if resume:
        seed = logged_experiment_seed(experiment_names, model_type)
        if seed is not None:
            typer.echo(f"==== resume: reusing logged seed {seed} for {experiment_names} ====")
            return seed
    return rng.randint(0, 2**32 - 1)


def has_finished_run(experiment_name: str, params: dict) -> bool:
    """True if ``experiment_name`` holds a FINISHED run whose params all equal ``params``."""
    client = _client()
    experiment_id = _experiment_id(client, experiment_name)
    if experiment_id is None:
        return False
    runs = client.search_runs(
        [experiment_id],
        filter_string=_param_filter(params),
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
    )
    return len(runs) > 0


def run_training_config(
    train_config: dict,
    *,
    dataset_path: Path,
    experiment_name: str,
    seed: int,
    stride: int = STRIDE,
    context_length: int = CONTEXT_LENGTH,
    resume: bool = False,
    **extra_train_kwargs,
) -> bool:
    """Run ``train.main`` for a single grid config.

    Loop literals (batch_size/num_epochs/early_stopping_patience/num_workers) are
    NOT re-listed here — they default in ``train.main`` from ``config.py``. Pass
    overrides (and runner-specific kwargs like ``exp_transform``) via
    ``extra_train_kwargs``.

    With ``resume`` a config that already has a FINISHED run in ``experiment_name``
    (same loss params, seed, horizon, stride and — when the runner sets them —
    architecture and training budget) is skipped. Returns ``True`` if a run was
    trained, ``False`` if it was skipped.
    """
    loss_type = train_config["loss_type"]
    # Explicit output_length key is required on every config; OUTPUT_LENGTH is the
    # (correct) fallback so a dropped key can never silently change the horizon
    # (CLAUDE.md invariant #3).
    output_length = train_config.get("output_length", OUTPUT_LENGTH)

    kwargs = dict(
        dataset_path=dataset_path,
        loss_type=loss_type,
        output_length=output_length,
        context_length=context_length,
        stride=stride,
        experiment_name=experiment_name,
        seed=seed,
        **extra_train_kwargs,
    )
    sgt_kwargs = {}
    if loss_type.lower() == "sgt":
        sgt_kwargs = dict(
            sgt_loss_lambda=train_config["sgt_loss_lambda"],
            sgt_loss_q=train_config["sgt_loss_q"],
            sgt_loss_sigma=train_config["sgt_loss_sigma"],
            sgt_loss_p=train_config["sgt_loss_p"],
        )
        kwargs.update(sgt_kwargs)

    if resume:
        match = {
            "loss_type": loss_type,
            "random_state": seed,
            "output_length": output_length,
            "context_length": context_length,
            "stride": stride,
            **sgt_kwargs,
        }
        match.update(
            {k: extra_train_kwargs[k] for k in _EXTRA_MATCH_KEYS if k in extra_train_kwargs}
        )
        if has_finished_run(experiment_name, match):
            typer.echo(
                f"==== resume: skipping {experiment_name} loss={loss_type} "
                f"{sgt_kwargs or ''} seed={seed} (already FINISHED) ===="
            )
            return False

    train_main(**kwargs)
    return True
