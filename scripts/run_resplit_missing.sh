#!/usr/bin/env bash
# Re-run the transfer jobs that died of out-of-memory, staggered.
#
# Twelve transfer processes at 104 subjects do not fit in 24 GB. Each one sits
# at about 1.3 GB steady, but during kernel precomputation it allocates several
# 4680x4680 Gram matrices at once and peaks far higher. With all twelve
# precomputing simultaneously on 2026-09-17, three died with
# numpy ArrayMemoryError before completing a single subject.
#
# The survivors are past their peak, so the dead ones can be restarted now
# provided their peaks do not coincide: this script leaves four minutes between
# launches, which is longer than a precompute takes.
#
# Launch through the scheduler, never from an editor or agent session:
#   schtasks.exe /create /tn "quantaEEG_missing" /f /sc once /st HH:MM \
#     /tr "'C:\Program Files\Git\bin\bash.exe' -lc \"cd /g/codes/Ass/quantaEEG && bash scripts/run_resplit_missing.sh\""
#   schtasks.exe /run /tn "quantaEEG_missing"
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

log=results/run_resplit_missing.log
jobs=results/resplit_missing_jobs.txt
echo "missing-job rerun start $(date)  $(wc -l < "$jobs") jobs" >> "$log"

n=0
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  n=$((n + 1))
  [ "$n" -gt 1 ] && sleep 240
  echo "launch $n $(date)" >> "$log"
  bash -c "$cmd" >> "$log" 2>&1 &
done < "$jobs"
wait
echo "MISSING_DONE $(date)" >> "$log"
