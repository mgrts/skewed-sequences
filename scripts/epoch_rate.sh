#!/usr/bin/env bash
# Seconds per epoch per slot (mean of the last 6 epochs) and the aggregate speed-up over
# ONE process running alone (baseline: OWID ~12 s/epoch, RVR ~21 s/epoch on the L4).
#   bash scripts/epoch_rate.sh
REPO="${REPO:-$HOME/skewed-sequences}"
MAIN_ROOT="${MAIN_ROOT:-$REPO}"
WORK_BASE="${WORK_BASE:-$HOME/sweep_slots}"
python3 - "${MAIN_ROOT}" "${WORK_BASE}" <<'PY'
import glob, os, re, sys
from datetime import datetime
main_root, work_base = sys.argv[1], sys.argv[2]
logs = sorted(glob.glob(os.path.join(main_root, "sweep_main.log")) + glob.glob(os.path.join(work_base, "*", "sweep_*.log")))
ts_re = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+).*Epoch \d+/\d+")
total = 0.0; n_alive = 0
print(f"{'slot':16s} {'s/epoch':>8s} {'baseline':>9s} {'share of solo':>14s}")
for lf in logs:
    name = os.path.basename(lf)[len("sweep_"):-len(".log")]
    pid_file = os.path.join(os.path.dirname(lf), "slot.pid")
    try:
        alive = os.path.exists(pid_file) and os.kill(int(open(pid_file).read().strip()), 0) is None
    except OSError:
        alive = False
    stamps = []
    with open(lf, errors="ignore") as fh:
        for line in fh:
            line = re.sub(r"\x1b\[[0-9;]*m", "", line)
            m = ts_re.match(line)
            if m:
                stamps.append(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f"))
    base = 21.0 if name.startswith("rvr") else 12.0
    if len(stamps) < 2:
        print(f"{name:16s} {'-':>8s} {base:9.0f} {'(no epochs yet)':>14s}" + ("" if alive else "  [not running]"))
        continue
    last = stamps[-7:]
    gaps = [(b - a).total_seconds() for a, b in zip(last, last[1:])]
    gaps = [g for g in gaps if g < 3600]  # ignore the pause between runs
    if not gaps:
        continue
    mean = sum(gaps) / len(gaps)
    share = base / mean
    if alive:
        total += share; n_alive += 1
    print(f"{name:16s} {mean:8.1f} {base:9.0f} {share:14.2f}" + ("" if alive else "  [not running]"))
print(f"\n{n_alive} slot(s) running -> aggregate throughput = {total:.2f}x one solo process")
PY
