#!/usr/bin/env bash
#
# Full experiment sweep (transformer-only). Regenerates results from scratch.
#
#     poetry run bash scripts/run_sweep.sh 2>&1 | tee sweep.log
#
# The synthetic + multi-head studies need no internet and always run. OWID and
# RVR run only if their source CSV is present (so the sweep does not abort on an
# offline host):
#     OWID -> data/raw/dataset.csv            (download, or copy in manually)
#     RVR  -> data/external/rvr_us_data.csv    (Kaggle: respiratory-virus-response)
#
# Afterwards copy back mlruns.db + reports/.
#
# Tunable: N_RUNS [10], SYNTH_N [1000], SYNTH_STRIDE [5].
#
set -euo pipefail

# Headless matplotlib backend (Jupyter kernels export an inline backend that
# crashes `import matplotlib.pyplot` in the venv).
export MPLBACKEND="${MPLBACKEND:-Agg}"
if [ "${MPLBACKEND}" = "module://matplotlib_inline.backend_inline" ]; then
  export MPLBACKEND=Agg
fi

N_RUNS="${N_RUNS:-10}"
SYNTH_N="${SYNTH_N:-1000}"
SYNTH_STRIDE="${SYNTH_STRIDE:-5}"
OWID_RAW="data/raw/dataset.csv"
RVR_RAW="data/external/rvr_us_data.csv"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "Device check"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# ---- internet-free core (always runs) -------------------------------------
log "Synthetic sweep (n_sequences=${SYNTH_N}, stride=${SYNTH_STRIDE})"
skseq experiments run-synthetic main --n-runs "${N_RUNS}" --n-sequences "${SYNTH_N}" --stride "${SYNTH_STRIDE}"

log "Multi-head attention study (heavy-tailed synthetic)"
skseq experiments run-head-sweep main --n-runs "${N_RUNS}" --n-sequences "${SYNTH_N}" --stride "${SYNTH_STRIDE}"

# ---- real datasets (only if their source CSV is present) ------------------
if [ ! -f "${OWID_RAW}" ]; then
  skseq data download-owid download || log "OWID download failed (offline?) — will skip OWID"
fi
if [ -f "${OWID_RAW}" ]; then
  log "OWID COVID sweep"
  ( skseq data process-owid main && skseq experiments run-owid main --n-runs "${N_RUNS}" ) \
    || log "OWID stage failed — continuing"
else
  log "[skip] OWID — ${OWID_RAW} not found"
fi

if [ ! -f "${RVR_RAW}" ]; then
  skseq data download-rvr download || log "RVR download failed (offline?) — will skip RVR"
fi
if [ -f "${RVR_RAW}" ]; then
  log "RVR sweep (bed occupancy + influenza)"
  skseq experiments run-rvr main --n-runs "${N_RUNS}" || log "RVR stage failed — continuing"
else
  log "[skip] RVR — ${RVR_RAW} not found"
fi

# ---- collect + aggregate whatever ran -------------------------------------
log "Collecting MLflow runs -> reports/experiment_results.csv"
skseq experiments collect-results main

log "Aggregating per metric -> reports/experiment_summary_<metric>.csv"
for metric in best_test_smape best_test_rmse best_test_mae best_test_mase; do
  skseq experiments aggregate-results main \
    --metric "${metric}" \
    --output-path "reports/experiment_summary_${metric}.csv"
done

log "SWEEP COMPLETE — copy back mlruns.db and reports/"
