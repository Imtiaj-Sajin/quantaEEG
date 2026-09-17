#!/usr/bin/env bash
# Where did the revision run get to?
#
# Answers the question after an interruption (power cut, killed session, a
# reboot): which jobs finished, which are outstanding, did anything crash, and
# is there finished work sitting uncommitted.
#
#   bash scripts/revision_status.sh
#
# Read-only. Safe to run while the queue is still going.
set -u
cd "$(dirname "$0")/.."

blue() { printf '\n== %s ==\n' "$1"; }

blue "repo"
printf '  HEAD      %s\n' "$(git log -1 --format='%h %ad %s' --date=format:'%Y-%m-%d %H:%M')"
if git rev-parse --verify -q origin/main >/dev/null; then
    read -r behind ahead < <(git rev-list --left-right --count origin/main...HEAD)
    printf '  vs origin behind %s, ahead %s\n' "$behind" "$ahead"
fi
dirty=$(git status --porcelain | wc -l)
printf '  uncommitted files: %s\n' "$dirty"
if [ "$dirty" -gt 0 ]; then
    git status --porcelain | head -15 | sed 's/^/    /'
    uncommitted_results=$(git status --porcelain | grep -cE ' results/' || true)
    if [ "${uncommitted_results:-0}" -gt 0 ]; then
        echo "    WARNING: $uncommitted_results result file(s) above are finished"
        echo "    compute that is not yet committed. Commit before relaunching."
    fi
fi

blue "stage markers in results/run_revision.log"
if [ -f results/run_revision.log ]; then
    grep -E 'revision run start|STAGE1_DONE|STAGE2_DONE|stage . start' \
        results/run_revision.log | tail -10 | sed 's/^/  /'
else
    echo "  no results/run_revision.log on this machine"
fi

blue "stage 1: full-cohort benchmark batches"
python - <<'PY'
import pathlib, subprocess
jobs = subprocess.run(["python", "scripts/archive/full_cohort_queue.py"],
                      capture_output=True, text=True).stdout.splitlines()
done, todo = [], []
for line in jobs:
    p = line.split()
    if len(p) < 3:
        continue
    tag = p[2]
    (done if pathlib.Path(f"results/raw_folds_{tag}.csv").exists() else todo).append(tag)
print(f"  complete {len(done)} of {len(jobs)}")
if todo:
    print(f"  outstanding: {', '.join(todo)}")
    for tag in todo:
        part = pathlib.Path(f"results/raw_folds_{tag}.partial.csv")
        if part.exists():
            import pandas as pd
            n = pd.read_csv(part, usecols=["subject"]).subject.nunique()
            print(f"    {tag}: checkpoint present, {n} subjects done")
PY

blue "stage 2: transfer, diagnostics, calibration, four-class"
python - <<'PY'
import pathlib, re, subprocess
jobs = subprocess.run(["python", "scripts/archive/phase2_jobs.py"],
                      capture_output=True, text=True).stdout.splitlines()
done, todo = [], []
for line in jobs:
    m = re.search(r"if \[ -f (\S+) \]", line)
    t = re.search(r"echo skip (\S+);", line)
    if not m or not t:
        continue
    (done if pathlib.Path(m.group(1)).exists() else todo).append(t.group(1))
print(f"  complete {len(done)} of {len(jobs)}")
if todo:
    print(f"  outstanding ({len(todo)}): {', '.join(todo)}")
PY

blue "crashes"
crash=$(grep -l forrtl results/run_*.log 2>/dev/null || true)
if [ -n "$crash" ]; then
    echo "  Intel Fortran window-CLOSE crash in:"
    echo "$crash" | sed 's/^/    /'
    echo "  cause: the process had a console and the session was destroyed."
    echo "  fix:   relaunch via the scheduled task, which has no console."
else
    echo "  no forrtl window-CLOSE crashes found"
fi
nonzero=$(grep -h 'JOB_EXIT=' results/run_*.log 2>/dev/null | grep -v 'JOB_EXIT=0' | wc -l)
printf '  jobs recording a non-zero exit: %s\n' "$nonzero"

blue "scheduled task"
if command -v powershell.exe >/dev/null 2>&1; then
    powershell.exe -NoProfile -Command "
      try {
        \$t = Get-ScheduledTask -TaskName 'quantaEEG_revision' -ErrorAction Stop
        \$i = Get-ScheduledTaskInfo -TaskName 'quantaEEG_revision'
        '  state           ' + \$t.State
        '  last run        ' + \$i.LastRunTime
        '  last result     ' + \$i.LastTaskResult
      } catch { '  task not registered' }
    " 2>/dev/null
else
    echo "  powershell not available here"
fi

blue "what to do"
echo "  1. commit anything listed as uncommitted above, so it survives"
echo "  2. relaunch:  Start-ScheduledTask -TaskName 'quantaEEG_revision'"
echo "     finished jobs skip themselves and batches resume from checkpoints"
