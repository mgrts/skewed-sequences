#!/usr/bin/env bash
#
# Run the real-data sweeps as several processes on ONE GPU host.
#
#     bash scripts/launch_parallel.sh            # launch (idempotent: skips slots already running)
#     bash scripts/status_parallel.sh            # progress of every slot
#
# The model is tiny, so one training process leaves the GPU and the 32 CPUs mostly
# idle. Each slot below is an independent process with its own SKSEQ_PROJ_ROOT
# (own mlruns.db, data/processed, sweep log) using the SAME code + virtualenv, so
# there is no clone to keep in sync and no SQLite write contention. The seeds of
# one dataset are split by run index (--first-run), which never breaks seed
# pairing (pairing is within a run index). Every slot runs with --resume, so a
# killed slot is just re-launched with this same script.
#
# Slot A stays in the main repo (its mlruns.db already holds the finished synthetic,
# head, lambda and OWID runs); the others live under ${WORK_BASE}. Merge at the end:
#
#     skseq experiments collect-results main \
#         --tracking-uri sqlite:///$HOME/skewed-sequences/mlruns.db \
#         --tracking-uri sqlite:///$HOME/sweep_slots/owid_b/mlruns.db  ...   (one per slot)
#
set -uo pipefail

REPO="${REPO:-$HOME/skewed-sequences}"
WORK_BASE="${WORK_BASE:-$HOME/sweep_slots}"
N_RUNS="${N_RUNS:-10}"
SPLIT="${SPLIT:-5}"                 # runs 1..SPLIT in slot "a", SPLIT+1..N_RUNS in slot "b"
export MPLBACKEND=Agg OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

BED="average_inpatient_beds_occupied"
FLU="total_admissions_all_influenza_confirmed_past_7days"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# slot name | project root | runner command (without the common flags)
SLOTS=(
  "owid_a|${REPO}|run-owid main --n-runs ${N_RUNS}"
  "owid_b|${WORK_BASE}/owid_b|run-owid main --n-runs ${N_RUNS} --first-run $((SPLIT + 1))"
  "rvr_bed_a|${WORK_BASE}/rvr_bed_a|run-rvr main --n-runs ${SPLIT} --time-series ${BED}"
  "rvr_bed_b|${WORK_BASE}/rvr_bed_b|run-rvr main --n-runs ${N_RUNS} --first-run $((SPLIT + 1)) --time-series ${BED}"
  "rvr_flu_a|${WORK_BASE}/rvr_flu_a|run-rvr main --n-runs ${SPLIT} --time-series ${FLU}"
  "rvr_flu_b|${WORK_BASE}/rvr_flu_b|run-rvr main --n-runs ${N_RUNS} --first-run $((SPLIT + 1)) --time-series ${FLU}"
)
# slot owid_a covers runs 1..N_RUNS but with --resume it skips what the main sweep already
# finished; owid_b starts at SPLIT+1. Give owid_a an explicit end so they don't overlap:
SLOTS[0]="owid_a|${REPO}|run-owid main --n-runs ${SPLIT}"

cd "${REPO}" || exit 1
mkdir -p "${WORK_BASE}"

for slot in "${SLOTS[@]}"; do
  IFS='|' read -r name root cmd <<< "${slot}"
  logfile="${root}/sweep_${name}.log"
  if pgrep -f "sweep-slot=${name}" >/dev/null 2>&1; then
    log "${name}: already running — skipped"
    continue
  fi
  mkdir -p "${root}/data/processed" "${root}/data/external" "${root}/data/raw" "${root}/reports"
  # Each slot needs the inputs its runner reads: OWID reads the processed .npy, RVR
  # regenerates its .npy from the raw CSV (per series, hence one root per series).
  [ -f "${root}/data/processed/dataset.npy" ] || cp "${REPO}/data/processed/dataset.npy" "${root}/data/processed/" 2>/dev/null
  [ -f "${root}/data/external/rvr_us_hospitalization_daily.csv" ] || cp "${REPO}/data/external/rvr_us_hospitalization_daily.csv" "${root}/data/external/" 2>/dev/null
  log "${name}: launching in ${root}  (skseq experiments ${cmd})"
  # The 'sweep-slot=<name>' token is only there so pgrep can find the process by slot.
  SKSEQ_PROJ_ROOT="${root}" setsid nohup bash -c "exec -a 'sweep-slot=${name}' poetry run skseq experiments ${cmd} --resume" \
    >> "${logfile}" 2>&1 < /dev/null &
  sleep 2
done

log "done. status: bash scripts/status_parallel.sh"
