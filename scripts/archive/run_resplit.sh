#!/usr/bin/env bash
# Run the re-split transfer queue written by scripts/resplit_transfer.py.
#
# Launch it through the Windows scheduler, never from an editor or agent
# session: a process started from one is torn down with it, which is how two
# earlier attempts at this project's long runs died silently.
#
#   powershell:
#     $a = New-ScheduledTaskAction -Execute "C:\Program Files\Git\bin\bash.exe" `
#         -Argument '-lc "cd /g/codes/quantaEEG && bash scripts/run_resplit.sh"' `
#         -WorkingDirectory "G:\codes\Ass\quantaEEG"
#     Register-ScheduledTask -TaskName "quantaEEG_resplit" -Action $a -Settings $s -Force
#     Start-ScheduledTask -TaskName "quantaEEG_resplit"
#
# Safe to relaunch: each job writes its own tagged output and resumes from its
# own checkpoint, and resplit_transfer.py only ever queues subject-frame pairs
# that no existing file already contains.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

log=results/run_resplit.log
jobs=results/resplit_jobs.txt
echo "resplit start $(date)  pid $$  $(wc -l < "$jobs") jobs" >> "$log"
xargs -d '\n' -P 12 -I{} bash -c "{}" < "$jobs" >> "$log" 2>&1
echo "RESPLIT_DONE $(date)" >> "$log"
