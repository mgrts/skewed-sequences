#!/usr/bin/env bash
# Progress of every parallel sweep slot launched by launch_parallel.sh.
REPO="${REPO:-$HOME/skewed-sequences}"
WORK_BASE="${WORK_BASE:-$HOME/sweep_slots}"
printf '%-10s %-8s %9s %9s  %s\n' slot state finished failures "last line"
for logfile in "${REPO}"/sweep_owid_a.log "${WORK_BASE}"/*/sweep_*.log; do
  [ -f "${logfile}" ] || continue
  name=$(basename "${logfile}" .log); name=${name#sweep_}
  if pgrep -f "sweep-slot=${name}" >/dev/null 2>&1; then state=alive; else state=GONE; fi
  done=$(grep -c 'Training complete' "${logfile}")
  bad=$(grep -cE 'FAILED|Traceback' "${logfile}")
  last=$(tail -n 1 "${logfile}" | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-70)
  printf '%-10s %-8s %9s %9s  %s\n' "${name}" "${state}" "${done}" "${bad}" "${last}"
done
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader | sed 's/^/GPU: /'
