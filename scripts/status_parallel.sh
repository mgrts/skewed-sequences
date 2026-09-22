#!/usr/bin/env bash
# Progress of the parallel sweep slots (launch_parallel.sh).
#   bash scripts/status_parallel.sh           # table + GPU
#   bash scripts/status_parallel.sh --merge   # also collect-results over every store -> reports/experiment_results.csv
REPO="${REPO:-$HOME/skewed-sequences}"
MAIN_ROOT="${MAIN_ROOT:-$REPO}"
WORK_BASE="${WORK_BASE:-$HOME/sweep_slots}"
alive() { local pf="$1"; [ -f "${pf}" ] && kill -0 "$(cat "${pf}")" 2>/dev/null; }
printf '%-16s %-6s %9s %9s %9s  %s\n' slot state finished skipped failures "last line"
stores=()
for root in "${MAIN_ROOT}" "${WORK_BASE}"/*/; do
  root="${root%/}"; [ -d "${root}" ] || continue
  logfile=$(ls -t "${root}"/sweep_*.log 2>/dev/null | head -1); [ -f "${logfile}" ] || continue
  name=$(basename "${logfile}" .log); name=${name#sweep_}
  if alive "${root}/slot.pid"; then state=alive
  elif tail -n 3 "${logfile}" | grep -q 'SLOT EXIT=0'; then state=done
  elif tail -n 3 "${logfile}" | grep -q 'SLOT EXIT='; then state=failed
  else state=DIED; fi
  done=$(grep -c 'Training complete' "${logfile}")
  skipped=$(grep -c 'resume: skipping' "${logfile}")
  bad=$(grep -cE 'FAILED|Traceback' "${logfile}")
  last=$(grep -vE '^\s*$|SLOT EXIT' "${logfile}" | tail -n 1 | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-64)
  printf '%-16s %-6s %9s %9s %9s  %s\n' "${name}" "${state}" "${done}" "${skipped}" "${bad}" "${last}"
  [ -f "${root}/mlruns.db" ] && stores+=("sqlite:///${root}/mlruns.db")
done
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader | sed 's/^/GPU: /'
if [ "${1:-}" = "--merge" ]; then
  args=(); for s in "${stores[@]}"; do args+=(--tracking-uri "$s"); done
  ( cd "${REPO}" && MPLBACKEND=Agg poetry run skseq experiments collect-results main "${args[@]}" )
fi
