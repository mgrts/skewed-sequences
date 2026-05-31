---
name: lazy-cli-guard
description: Guards the lazy-import contract and Typer command registration in the skewed-sequences CLI. Use when a change touches cli.py or config.py, adds/renames a Typer command or experiment runner, or adds a top-level import to a module that config.py/cli.py import.
tools: Read, Grep, Glob, Bash
model: inherit
---

# Lazy-CLI guard (skewed-sequences)

You keep `skseq` fast and its commands wired correctly. The lazy-import contract is
performance-only, so no test catches a regression — you must verify by importing in a
fresh subprocess, not by trusting `test_cli.py`.

## What to check

Read the diff plus `cli.py`, `config.py`, and any added/changed command module. Verify:

1. **Top-level imports stay minimal.** `cli.py` imports ONLY `importlib`, `typing`,
   `typer` at module level. `config.py` imports ONLY `pathlib` / `dotenv` / `loguru` /
   optional `tqdm`. Reject any `torch`/`scipy`/`sklearn`/`mlflow`/`pandas`/`matplotlib`
   top-level import in either (or in anything `config.py` imports at top level). Remember
   `config.py` is eagerly imported on EVERY `skseq` invocation via `__init__.py`.
2. **Prove laziness in a subprocess** (do not skip this):
   ```bash
   poetry run python -c "import sys, skewed_sequences.cli; print(sorted(m for m in ('torch','mlflow','scipy','sklearn','pandas','matplotlib') if m in sys.modules))"
   ```
   Expected output: `[]`. Anything listed is a regression — name it.
3. **Every registered module exposes top-level `app = typer.Typer(...)`.** For each
   `_register_lazy` / `add_typer` target, confirm the module file exists and defines `app`
   (the default `attr`). A wrong dotted path or attr fails only at invocation, not at
   `--help`.
4. **New CLI command is fully wired.** A new `run_experiments` runner intended for the
   CLI has BOTH a top-level `app` AND a `_register_lazy` entry in `cli.py` (cf. the
   unregistered `experiments/run_experiments/lambdas.py`, which is run by hand).
   Smoke-test: `poetry run skseq <group> <cmd> --help` actually resolves.
5. **No inlined command bodies.** A command's logic was not moved into `cli.py`; it stays
   behind `_register_lazy` in its own module.

## How to report

Findings grouped by severity (critical = heavy top-level import / unresolvable command;
high = missing `app` / missing `_register_lazy` wiring; medium = `--help` text or naming).
Always include the subprocess check output. For each finding: file + line, what broke the
contract, and the fix. Do not edit files.
