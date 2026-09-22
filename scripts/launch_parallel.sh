#!/usr/bin/env bash
#
# Run the real-data sweeps as MANY processes on one GPU host, without training
# anything twice.
#
#     bash scripts/launch_parallel.sh                 # launch (idempotent; re-run to resume dead slots)
#     PARTS=10 bash scripts/launch_parallel.sh        # more slots per dataset (default 5)
#     MAX_ALIVE=6 bash scripts/launch_parallel.sh     # keep at most 6 processes running; re-run to top up
#     bash scripts/status_parallel.sh                 # progress per slot + GPU
#
# Concurrency: PARTS fixes how the work is CUT (cannot change once slots exist);
# MAX_ALIVE limits how many slots RUN at once (can change any time). Size MAX_ALIVE by
# the pod's CPU quota (cat /sys/fs/cgroup/cpu.max), not by the host's core count: each
# training process is CPU-launch-bound and needs about one full core. Run the GPU under
# NVIDIA MPS (scripts/mps.sh start) so the processes share it instead of time-slicing.
#
# How the work is split
#   For each dataset (owid, rvr_bed, rvr_flu) the script looks into the MAIN store
#   (${MAIN_ROOT}/mlruns.db) for the highest run index that already has runs there —
#   whatever the serial sweep or an earlier launch produced. Those run indices stay in
#   the main store and are resumed by ONE sequential "main" process (finished runs are
#   skipped, a half-done run is completed). The remaining run indices are cut into
#   PARTS contiguous ranges, each trained by its own process in its own root under
#   ${WORK_BASE} (own mlruns.db, data/, log; same code + venv via SKSEQ_PROJ_ROOT).
#   A run index therefore lives in exactly one store, so seed pairing (which is within
#   a run index) is never broken and no two processes write one SQLite file.
#
#   Slot roots encode their range (owid_r3-4). Re-running the script with the same
#   PARTS resumes them; with a different PARTS it refuses, because re-cutting ranges
#   that already started would duplicate run indices across stores.
#
# Sizing: total processes = 1 + 3*PARTS at most. Each needs ~0.6-1 GB of GPU memory
# and one CPU thread (OMP_NUM_THREADS is set from the core count). PARTS=10 gives one
# run index (36 trainings) per slot.
#
# Merge when everything is done:
#     bash scripts/status_parallel.sh --merge     # collect-results over every store
#
set -uo pipefail

REPO="${REPO:-$HOME/skewed-sequences}"
MAIN_ROOT="${MAIN_ROOT:-$REPO}"
WORK_BASE="${WORK_BASE:-$HOME/sweep_slots}"
N_RUNS="${N_RUNS:-10}"
PARTS="${PARTS:-5}"
MAX_ALIVE="${MAX_ALIVE:-0}"              # 0 = no limit
EXTRA="${EXTRA:-}"                       # extra runner flags, e.g. "--num-epochs 1" for a smoke test
DATASETS="${DATASETS:-owid rvr_bed rvr_flu}"
export MPLBACKEND=Agg

BED="average_inpatient_beds_occupied"
FLU="total_admissions_all_influenza_confirmed_past_7days"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
SETSID=$(command -v setsid >/dev/null 2>&1 && echo setsid || echo "")

cd "${REPO}" || exit 1
mkdir -p "${WORK_BASE}"

# ---- refuse to mix with the old fixed 2-way layout (owid_b, rvr_bed_a, ...) --------
old=$(ls -d "${WORK_BASE}"/{owid_a,owid_b,rvr_bed_a,rvr_bed_b,rvr_flu_a,rvr_flu_b} 2>/dev/null || true)
if [ -n "${old}" ] && [ "${IGNORE_OLD_LAYOUT:-0}" != "1" ]; then
  log "found slots from the old 2-way layout:"; echo "${old}"
  log "let them finish (status_parallel.sh) or move them away; set IGNORE_OLD_LAYOUT=1 to proceed anyway"
  exit 1
fi

# ---- per-dataset runner + experiment prefix -----------------------------------------
runner_for() {
  case "$1" in
    owid)    echo "run-owid main" ;;
    rvr_bed) echo "run-rvr main --time-series ${BED}" ;;
    rvr_flu) echo "run-rvr main --time-series ${FLU}" ;;
    *) echo "unknown dataset $1" >&2; exit 1 ;;
  esac
}
prefix_for() {
  case "$1" in
    owid) echo "covid-owid" ;; rvr_bed) echo "rvr-us-bed-occupancy" ;; rvr_flu) echo "rvr-us-influenza-cases" ;;
  esac
}
# highest run index with at least one run in a store (0 if none / no store)
max_run_in_store() {
  local store="$1" prefix="$2"
  [ -f "${store}" ] || { echo 0; return; }
  poetry run python - "${store}" "${prefix}" <<'PY' 2>/dev/null
import re, sqlite3, sys
db, prefix = sys.argv[1], sys.argv[2]
con = sqlite3.connect(db)
rows = con.execute(
    "select e.name from experiments e where e.name like ? and exists "
    "(select 1 from runs r where r.experiment_id = e.experiment_id)", (prefix + "_run_%",)
).fetchall()
idx = [int(m.group(1)) for (n,) in rows for m in [re.search(r"_run_(\d+)$", n)] if m]
print(max(idx) if idx else 0)
PY
}
alive() { local pf="$1"; [ -f "${pf}" ] && kill -0 "$(cat "${pf}")" 2>/dev/null; }

n_alive() {
  local n=0 pf
  for pf in "${MAIN_ROOT}/slot.pid" "${WORK_BASE}"/*/slot.pid; do alive "${pf}" && n=$((n + 1)); done
  echo "${n}"
}
slot_done() { tail -n 3 "$1" 2>/dev/null | grep -q 'SLOT EXIT=0'; }

launch() {  # name root cmd
  local name="$1" root="$2" cmd="$3" pf="$2/slot.pid" logfile="$2/sweep_$1.log"
  if alive "${pf}"; then log "${name}: already running — skipped"; return; fi
  if slot_done "${logfile}"; then log "${name}: finished earlier — skipped"; return; fi
  if [ "${MAX_ALIVE}" -gt 0 ] && [ "$(n_alive)" -ge "${MAX_ALIVE}" ]; then
    log "${name}: not started (MAX_ALIVE=${MAX_ALIVE} reached) — re-run later to top up"; return
  fi
  mkdir -p "${root}/data/processed" "${root}/data/external" "${root}/reports"
  [ -f "${root}/data/processed/dataset.npy" ] || cp "${REPO}/data/processed/dataset.npy" "${root}/data/processed/" 2>/dev/null
  [ -f "${root}/data/external/rvr_us_hospitalization_daily.csv" ] || cp "${REPO}/data/external/rvr_us_hospitalization_daily.csv" "${root}/data/external/" 2>/dev/null
  log "${name}: launching  (${cmd})"
  # setsid detaches the slot from this shell AND from a notebook kernel (Linux); macOS
  # has no setsid, so fall back to plain nohup there (local smoke tests only).
  SKSEQ_PROJ_ROOT="${root}" OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
    ${SETSID} nohup bash -c "echo \$\$ > '${pf}'; ${cmd}; echo \"SLOT EXIT=\$?\"" >> "${logfile}" 2>&1 < /dev/null &
  sleep 1
}

# ---- plan --------------------------------------------------------------------------
declare -a MAIN_CMDS=() SLOT_NAMES=() SLOT_ROOTS=() SLOT_CMDS=()
for ds in ${DATASETS}; do
  runner=$(runner_for "${ds}"); prefix=$(prefix_for "${ds}")
  done_upto=$(max_run_in_store "${MAIN_ROOT}/mlruns.db" "${prefix}")
  [ "${done_upto}" -gt "${N_RUNS}" ] && done_upto="${N_RUNS}"
  if [ "${done_upto}" -ge 1 ]; then
    MAIN_CMDS+=("poetry run skseq experiments ${runner} --n-runs ${done_upto} --resume ${EXTRA}")
  fi
  remaining=$((N_RUNS - done_upto))
  [ "${remaining}" -le 0 ] && { log "${ds}: runs 1-${done_upto} in the main store; nothing to split"; continue; }
  parts=$(( PARTS < remaining ? PARTS : remaining ))
  chunk=$(( (remaining + parts - 1) / parts ))
  # plan this dataset's ranges first ...
  planned=""
  a=$((done_upto + 1))
  while [ "${a}" -le "${N_RUNS}" ]; do
    b=$(( a + chunk - 1 )); [ "${b}" -gt "${N_RUNS}" ] && b="${N_RUNS}"
    name="${ds}_r${a}-${b}"
    planned="${planned} ${name}"
    SLOT_NAMES+=("${name}"); SLOT_ROOTS+=("${WORK_BASE}/${name}")
    SLOT_CMDS+=("poetry run skseq experiments ${runner} --first-run ${a} --n-runs ${b} --resume ${EXTRA}")
    a=$((b + 1))
  done
  # ... then refuse if a slot dir on disk is not one of them (a different cut would
  # duplicate run indices across stores). Deferred/missing planned slots are fine.
  for existing in "${WORK_BASE}/${ds}_r"*; do
    [ -d "${existing}" ] || continue
    case " ${planned} " in *" $(basename "${existing}") "*) ;; *)
      log "${ds}: slot $(basename "${existing}") on disk is not part of the PARTS=${PARTS} split (${planned# }); keep the PARTS you launched with or move it away"; exit 1 ;;
    esac
  done
done

total=$(( ${#SLOT_NAMES[@]} + (${#MAIN_CMDS[@]} > 0 ? 1 : 0) ))
cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
# honour a cgroup CPU quota (JupyterHub pods usually have one far below the host's cores)
if [ -r /sys/fs/cgroup/cpu.max ]; then
  read -r quota period < /sys/fs/cgroup/cpu.max
  [ "${quota}" != "max" ] && cores=$(( (quota + period - 1) / period ))
fi
concurrent=$(( MAX_ALIVE > 0 && MAX_ALIVE < total ? MAX_ALIVE : total ))
THREADS=$(( cores / (concurrent > 0 ? concurrent : 1) )); [ "${THREADS}" -lt 1 ] && THREADS=1
log "plan: ${total} processes (${#SLOT_NAMES[@]} slots + $(( ${#MAIN_CMDS[@]} > 0 ? 1 : 0 )) main), up to ${concurrent} at once, ${THREADS} CPU thread(s) each (${cores} CPUs available), PARTS=${PARTS}, N_RUNS=${N_RUNS}"

# ---- launch ------------------------------------------------------------------------
if [ "${#MAIN_CMDS[@]}" -gt 0 ]; then
  chain=$(printf '%s; ' "${MAIN_CMDS[@]}"); chain="${chain%; }"   # no trailing ';' (the wrapper appends its own)
  launch "main" "${MAIN_ROOT}" "${chain}"
fi
for i in "${!SLOT_NAMES[@]}"; do
  launch "${SLOT_NAMES[$i]}" "${SLOT_ROOTS[$i]}" "${SLOT_CMDS[$i]}"
done
log "launched. progress: bash scripts/status_parallel.sh"
