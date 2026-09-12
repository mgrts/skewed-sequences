#!/usr/bin/env bash
#
# Experiment sweep (transformer-only), resumable and stage-selectable.
#
#     nohup poetry run bash scripts/run_sweep.sh > sweep.log 2>&1 &
#     tail -f sweep.log
#
# Stages (STAGES env var, comma-separated; default = all, in this order):
#     synthetic  4 synthetic datasets x 36 losses x N_RUNS          (~1440 runs)
#     head       multi-head study on heavy-tailed synthetic         (~400 runs)
#     lambda     fine lambda sub-sweep on heavy-tailed-skewed       (~120 runs)
#     owid       OWID COVID (daily JHU file, auto-downloaded)       (~360 runs)
#     rvr        RVR US bed occupancy + influenza (auto-downloaded) (~720 runs)
#     collect    collect-results + aggregate-results + increment-fit
#
#     STAGES=owid,rvr,lambda,collect nohup poetry run bash scripts/run_sweep.sh > sweep.log 2>&1 &
#
# RESUME: every runner is called with --resume. A config that already has a
# FINISHED run in MLflow (same experiment name, loss params, seed, stride,
# architecture) is skipped, and the seed already logged for an experiment is
# reused so appended runs stay seed-paired. A killed sweep is therefore simply
# re-launched with the same command. NEVER delete mlruns.db between launches —
# it is the only record of the finished runs (and of the seeds).
#
# Real data: the OWID and RVR downloads are small (~1 MB / ~3 MB) and row-count /
# cadence validated, so they always run; an offline host falls back to a file that
# is already present and the loaders refuse piecewise-constant (weekly) data.
#
# A failing training stage is logged and the remaining stages still run (collect /
# aggregate always see whatever finished); grep the log for 'FAILED'.
#
# Tunable: N_RUNS [10], SYNTH_N [1000], SYNTH_STRIDE [5], NUM_WORKERS [0]. Keep
# NUM_WORKERS at 0: the datasets are in-memory tensors, so DataLoader workers only add
# IPC overhead (NUM_WORKERS=4 doubled the epoch time on the JupyterHub GPU, 2026-09-12).
# SYNTH_N and
# SYNTH_STRIDE mirror config.SYNTHETIC_N_SEQUENCES / SYNTHETIC_STRIDE — the synthetic,
# head and lambda stages must all use the same values (the lambda stage appends to the
# synthetic experiments and refuses a different logged stride).
#
set -euo pipefail

# Headless matplotlib backend (Jupyter kernels export an inline backend that
# crashes `import matplotlib.pyplot` in the venv).
export MPLBACKEND="${MPLBACKEND:-Agg}"
if [ "${MPLBACKEND}" = "module://matplotlib_inline.backend_inline" ]; then
  export MPLBACKEND=Agg
fi

STAGES="${STAGES:-synthetic,head,lambda,owid,rvr,collect}"
N_RUNS="${N_RUNS:-10}"
SYNTH_N="${SYNTH_N:-1000}"
SYNTH_STRIDE="${SYNTH_STRIDE:-5}"
NUM_WORKERS="${NUM_WORKERS:-0}"
OWID_RAW="data/raw/owid_jhu_new_cases.csv"
RVR_RAW="data/external/rvr_us_hospitalization_daily.csv"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
has_stage() { case ",${STAGES}," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

log "Device check"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
log "Stages: ${STAGES} | N_RUNS=${N_RUNS} SYNTH_N=${SYNTH_N} SYNTH_STRIDE=${SYNTH_STRIDE} NUM_WORKERS=${NUM_WORKERS}"

# ---- internet-free core --------------------------------------------------
if has_stage synthetic; then
  log "Synthetic sweep (n_sequences=${SYNTH_N}, stride=${SYNTH_STRIDE})"
  skseq experiments run-synthetic main --n-runs "${N_RUNS}" --n-sequences "${SYNTH_N}" \
    --stride "${SYNTH_STRIDE}" --num-workers "${NUM_WORKERS}" --resume \
    || log "synthetic stage FAILED (exit $?) — continuing"
fi

if has_stage head; then
  log "Multi-head attention study (heavy-tailed synthetic)"
  skseq experiments run-head-sweep main --n-runs "${N_RUNS}" --n-sequences "${SYNTH_N}" \
    --stride "${SYNTH_STRIDE}" --num-workers "${NUM_WORKERS}" --resume \
    || log "head stage FAILED (exit $?) — continuing"
fi

if has_stage lambda; then
  log "Fine lambda sub-sweep (heavy-tailed-skewed; appended to the synthetic experiments)"
  skseq experiments run-lambda-sweep main --n-runs "${N_RUNS}" --n-sequences "${SYNTH_N}" \
    --stride "${SYNTH_STRIDE}" --num-workers "${NUM_WORKERS}" --resume \
    || log "lambda stage FAILED (exit $?) — continuing"
fi

# ---- real datasets -------------------------------------------------------
if has_stage owid; then
  skseq data download-owid download || log "OWID download failed (offline?) — using ${OWID_RAW} if present"
  if [ -f "${OWID_RAW}" ]; then
    log "OWID COVID sweep"
    ( skseq data process-owid main \
      && skseq experiments run-owid main --n-runs "${N_RUNS}" --num-workers "${NUM_WORKERS}" --resume ) \
      || log "OWID stage FAILED (exit $?) — continuing"
  else
    log "[skip] OWID — ${OWID_RAW} not found"
  fi
fi

if has_stage rvr; then
  skseq data download-rvr download || log "RVR download failed (offline?) — using ${RVR_RAW} if present"
  if [ -f "${RVR_RAW}" ]; then
    log "RVR sweep (bed occupancy + influenza)"
    skseq experiments run-rvr main --n-runs "${N_RUNS}" --num-workers "${NUM_WORKERS}" --resume \
      || log "RVR stage FAILED (exit $?) — continuing"
  else
    log "[skip] RVR — ${RVR_RAW} not found"
  fi
fi

# ---- collect + aggregate whatever ran ------------------------------------
if has_stage collect; then
  log "Collecting MLflow runs -> reports/experiment_results.csv"
  skseq experiments collect-results main

  log "Aggregating per metric -> reports/experiment_summary_<metric>*.csv"
  for metric in best_test_mase best_test_mae best_test_rmse best_test_smape; do
    skseq experiments aggregate-results main \
      --metric "${metric}" \
      --output-path "reports/experiment_summary_${metric}.csv"
  done

  log "Increment SGT fit (lambda / q guidance) -> reports/increment_sgt_fit.csv"
  skseq experiments increment-fit main --n-sequences "${SYNTH_N}" || log "increment-fit failed — continuing"
fi

log "SWEEP COMPLETE — copy back mlruns.db and reports/"
