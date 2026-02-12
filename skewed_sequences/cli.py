"""Unified Typer CLI entry point for the skewed-sequences project.

Heavy scientific imports (scipy, sklearn, torch, …) are deferred until the
user actually invokes a sub-command so that ``skseq --help`` stays fast.

After installing the package (``poetry install``), every command is available
under the ``skseq`` binary::

    skseq --help
    skseq train --help
    skseq data generate-synthetic --help
    skseq visualize synthetic --help
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lazy_typer(module_path: str, attr: str = "app") -> typer.Typer:
    """Return a Typer instance that is only imported when the sub-command is
    invoked.  We create a thin wrapper ``Typer`` whose single callback imports
    the real app and re-dispatches.

    For ``add_typer`` to work we just need to return the *real* app object, so
    we do import it — but we do so inside a function called lazily by
    ``add_typer`` at registration time.  To keep top-level import fast, we
    instead use a placeholder and swap it in via ``_register_lazy``.
    """
    mod = importlib.import_module(module_path)
    return getattr(mod, attr)


def _register_lazy(
    parent: typer.Typer,
    module_path: str,
    *,
    name: str,
    help: str,  # noqa: A002
    attr: str = "app",
) -> None:
    """Register a sub-command Typer that is imported only when Typer resolves
    its Click group (i.e. when the user actually types the sub-command)."""

    class _LazyTyper:
        """Descriptor-like wrapper.  ``typer.main.get_group`` calls
        ``typer_instance.registered_groups`` which eventually calls
        ``click.Group``.  We intercept by making ``registered_commands``
        etc. available only after lazy import.
        """

        def __init__(self) -> None:
            self._real: typer.Typer | None = None

        def _ensure(self) -> typer.Typer:
            if self._real is None:
                mod = importlib.import_module(module_path)
                self._real = getattr(mod, attr)
            return self._real

        # Typer inspects these when building the Click group
        def __getattr__(self, item: str):  # type: ignore[override]
            return getattr(self._ensure(), item)

    parent.add_typer(_LazyTyper(), name=name, help=help)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Root app
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="skseq",
    help="Skewed sequences — loss function analysis for transformer neural networks.",
    pretty_exceptions_show_locals=False,
)

# ---------------------------------------------------------------------------
# Sub-apps — all imports are deferred via _register_lazy
# ---------------------------------------------------------------------------

# -- Data -------------------------------------------------------------------
data_app = typer.Typer(name="data", help="Dataset generation and preprocessing.")
app.add_typer(data_app)

_register_lazy(
    data_app,
    "skewed_sequences.data.synthetic.generate_data",
    name="generate-synthetic",
    help="Generate synthetic SGT data.",
)
_register_lazy(
    data_app,
    "skewed_sequences.data.owid_covid.load_data",
    name="download-owid",
    help="Download OWID COVID CSV.",
)
_register_lazy(
    data_app,
    "skewed_sequences.data.owid_covid.dataset",
    name="process-owid",
    help="Process OWID COVID dataset.",
)
_register_lazy(
    data_app,
    "skewed_sequences.data.lanl.dataset",
    name="process-lanl",
    help="Process LANL earthquake dataset.",
)
_register_lazy(
    data_app,
    "skewed_sequences.data.rvr_us.dataset",
    name="process-rvr",
    help="Process RVR US hospitalization dataset.",
)
_register_lazy(
    data_app,
    "skewed_sequences.data.health_fitness.dataset",
    name="process-health-fitness",
    help="Process health-fitness data.",
)

# -- Modeling ---------------------------------------------------------------
_register_lazy(app, "skewed_sequences.modeling.train", name="train", help="Train a model.")

# -- Visualization ----------------------------------------------------------
_register_lazy(
    app, "skewed_sequences.visualize_data", name="visualize", help="Visualize dataset samples."
)
_register_lazy(app, "skewed_sequences.plots", name="plots", help="Generate boxplot comparisons.")
_register_lazy(
    app,
    "skewed_sequences.visualize_losses",
    name="visualize-losses",
    help="Visualize SGT loss interpolation with classical losses.",
)

# -- Experiments ------------------------------------------------------------
experiments_app = typer.Typer(name="experiments", help="Run experiment suites.")
app.add_typer(experiments_app)

_register_lazy(
    experiments_app,
    "skewed_sequences.experiments.run_experiments.synthetic_data",
    name="run-synthetic",
    help="Run synthetic data experiments.",
)
_register_lazy(
    experiments_app,
    "skewed_sequences.experiments.run_experiments.lanl_data",
    name="run-lanl",
    help="Run LANL data experiments.",
)
_register_lazy(
    experiments_app,
    "skewed_sequences.experiments.run_experiments.owid_covid_data",
    name="run-owid",
    help="Run OWID COVID experiments.",
)
_register_lazy(
    experiments_app,
    "skewed_sequences.experiments.run_experiments.rvr_us_data",
    name="run-rvr",
    help="Run RVR US experiments.",
)
_register_lazy(
    experiments_app,
    "skewed_sequences.experiments.calculate_dispersion_scaling",
    name="dispersion-scaling",
    help="Compute dispersion scaling.",
)
_register_lazy(
    experiments_app,
    "skewed_sequences.experiments.calculate_metrics",
    name="metrics",
    help="Compute dataset metrics.",
)


if __name__ == "__main__":
    app()
