#!/usr/bin/env bash
#
# Full experiment sweep (transformer-only). Regenerates every result from scratch
# and writes the aggregated summaries to reports/. Run on a CUDA GPU host:
#
#     poetry run bash scripts/run_sweep.sh 2>&1 | tee sweep.log
#
# Afterwards, copy back  mlruns.db  and  reports/  (mlruns.db is the source of truth).
#
# Tunable via env vars (defaults finish in ~1-2 days on a single A100):
#     N_RUNS         replicates per (loss, dataset)   [10]
#     SYNTH_N        synthetic sequences per config   [1000]
#     SYNTH_STRIDE   window stride (synthetic + head) [5]
#
set -euo pipefail

# Headless matplotlib backend. Jupyter kernels export MPLBACKEND=module://
# matplotlib_inline.backend_inline, which is invalid outside the notebook and
# crashes `import matplotlib.pyplot` in the venv. Force a non-interactive backend.
export MPLBACKEND="${MPLBACKEND:-Agg}"
if [ "${MPLBACKEND}" = "module://matplotlib_inline.backend_inline" ]; then
  export MPLBACKEND=Agg
fi

N_RUNS="${N_RUNS:-10}"
SYNTH_N="${SYNTH_N:-1000}"
SYNTH_STRIDE="${SYNTH_STRIDE:-5}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "Device check"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

log "Real-data prep (OWID COVID)"
skseq data download-owid download
skseq data process-owid main

log "Synthetic sweep (n_sequences=${SYNTH_N}, stride=${SYNTH_STRIDE})"
skseq experiments run-synthetic main --n-runs "${N_RUNS}" --n-sequences "${SYNTH_N}" --stride "${SYNTH_STRIDE}"

log "OWID COVID sweep"
skseq experiments run-owid main --n-runs "${N_RUNS}"

log "RVR sweep (bed occupancy + influenza)"
skseq experiments run-rvr main --n-runs "${N_RUNS}"

log "Multi-head attention study (heavy-tailed synthetic)"
skseq experiments run-head-sweep main --n-runs "${N_RUNS}" --n-sequences "${SYNTH_N}" --stride "${SYNTH_STRIDE}"

log "Collecting MLflow runs -> reports/experiment_results.csv"
skseq experiments collect-results main

log "Aggregating per metric -> reports/experiment_summary_<metric>.csv"
for metric in best_test_smape best_test_rmse best_test_mae best_test_mase; do
  skseq experiments aggregate-results main \
    --metric "${metric}" \
    --output-path "reports/experiment_summary_${metric}.csv"
done

log "SWEEP COMPLETE — copy back mlruns.db and reports/"
