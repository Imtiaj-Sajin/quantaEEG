#!/usr/bin/env bash
# The whole revision compute, in order, resumable:
#   stage 1  every PhysioNet within-subject suite on all 104 subjects
#   stage 2  transfer, Gram, concentration and shots on 104 subjects, then
#            few-trial calibration and four-class IV-2a
#
# Launch it detached from any editor or Claude Code session, so closing the
# session cannot kill it (that happened once, on 2026-09-16):
#
#   powershell: Invoke-CimMethod -ClassName Win32_Process -MethodName Create `
#       -Arguments @{CommandLine='"C:\Program Files\Git\bin\bash.exe" -lc "cd /g/codes/Ass/quantaEEG && bash scripts/run_revision.sh"'}
#
# Safe to relaunch at any time: finished jobs skip themselves and unfinished
# benchmark batches resume from their per-subject checkpoints.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

log=results/run_revision.log
echo "revision run start $(date)  pid $$" >> "$log"

python scripts/full_cohort_queue.py > results/full_cohort_jobs.txt
python scripts/phase2_jobs.py > results/phase2_jobs.txt

echo "stage 1 start $(date)" >> "$log"
xargs -d '\n' -P 10 -I{} bash -c "{}" < results/full_cohort_jobs.txt >> "$log" 2>&1
echo "STAGE1_DONE $(date)" >> "$log"

echo "stage 2 start $(date)" >> "$log"
xargs -d '\n' -P 10 -I{} bash -c "{}" < results/phase2_jobs.txt >> "$log" 2>&1
echo "STAGE2_DONE $(date)" >> "$log"

# Transfer, re-split across more processes once the rest of the queue drained
# and left half the machine idle. phase2_jobs.py omits the six chunk jobs while
# this file exists, so the two never run the same subject twice.
if [ -s results/resplit_jobs.txt ]; then
  echo "resplit start $(date)  $(wc -l < results/resplit_jobs.txt) jobs" >> "$log"
  xargs -d '\n' -P 12 -I{} bash -c "{}" < results/resplit_jobs.txt >> "$log" 2>&1
  echo "RESPLIT_DONE $(date)" >> "$log"
fi
