#!/usr/bin/env bash
# Revision compute on a machine that does not yet hold the full data cache.
#
# The default queue (run_revision.sh) does stage 1 then stage 2, which would
# block 18 immediately-runnable jobs behind a 3-hour download. This ordering
# instead starts every job that needs no download first, and fetches the rest
# in the background while those run.
#
#   phase A  now      18 IV-2a jobs (that cache is already present here)
#            now      PhysioNet subjects 33-109 and Cho2017 download in parallel
#   phase B  on data  4 outstanding filter-bank batches + 12 PhysioNet stage-2
#   phase C  on data  8 Cho2017 calibration jobs
#
# Every job skips itself when its output exists and benchmark batches resume
# from their per-subject checkpoints, so this is safe to relaunch at any point.
#
# Launch it as a scheduled task, never from a shell or editor session: a job
# that owns a console dies when the session is destroyed, which is what the
# forrtl window-CLOSE crash on 2026-09-16 was.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=src

log=results/run_here.log
mkdir -p results/full104
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$log"; }

say "run_here start, pid $$, host ${COMPUTERNAME:-unknown}"

python scripts/full_cohort_queue.py > results/full_cohort_jobs.txt
python scripts/phase2_jobs.py      > results/phase2_jobs.txt

# ---------------------------------------------------------------- downloads
# A prefetch started by an earlier launch of this script survives a task stop,
# because it is detached. Starting a second one would have two processes
# fetching the same files, so attach to a live one instead. "Live" means its
# log was written within the last two minutes; a prefetch that is running
# always writes progress faster than that.
alive() {  # alive LOGFILE
    [ -f "$1" ] || return 1
    local age=$(( $(date +%s) - $(stat -c %Y "$1" 2>/dev/null || echo 0) ))
    [ "$age" -lt 120 ]
}

if alive results/prefetch_33_109.log; then
    say "PhysioNet prefetch already running, attaching to it"
else
    say "starting PhysioNet prefetch for subjects 33-109"
    nohup python prefetch.py 33 109 > results/prefetch_33_109.log 2>&1 &
fi

if alive results/prefetch_cho2017.log; then
    say "Cho2017 prefetch already running, attaching to it"
else
    say "starting Cho2017 prefetch (52 subjects, about 10 GB)"
    nohup python -u -c "
import logging, warnings
warnings.filterwarnings('ignore')
logging.getLogger('moabb').setLevel(logging.ERROR)
import mne; mne.set_log_level('ERROR')
from moabb.datasets import Cho2017
ds = Cho2017()
for s in ds.subject_list:
    try:
        ds.get_data([s]); print('ok', s, flush=True)
    except Exception as e:
        print('FAIL', s, type(e).__name__, e, flush=True)
print('cho2017 prefetch done', flush=True)
" > results/prefetch_cho2017.log 2>&1 &
fi

# Wait on the cache itself rather than on a PID: a relaunch has no handle on a
# prefetch that an earlier launch started.
wait_physionet() {
    local cache="$HOME/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0"
    while :; do
        local n
        n=$(find "$cache" -name '*.edf' 2>/dev/null \
            | sed 's/.*S\([0-9]*\)R.*/\1/' | sort | uniq -c \
            | awk '$1>=3' | wc -l)
        [ "$n" -ge 104 ] && { say "PhysioNet cache complete: $n subjects"; return; }
        say "PhysioNet cache at $n of 104 subjects, waiting"
        sleep 120
    done
}

wait_cho() {
    while :; do
        grep -q 'cho2017 prefetch done' results/prefetch_cho2017.log 2>/dev/null \
            && { say "Cho2017 cache complete"; return; }
        local n
        n=$(grep -c '^ok ' results/prefetch_cho2017.log 2>/dev/null || echo 0)
        say "Cho2017 at $n of 52 subjects, waiting"
        sleep 120
    done
}

# ---------------------------------------------------------------- phase A
# IV-2a is already cached, so those 18 need no download. The four filter-bank
# batches join them rather than waiting for the whole cache: cohort_job.sh
# blocks on its own eleven subjects, so each starts the moment its data lands.
# They are also the longest single jobs here at roughly 2.9 h, so starting them
# early is what shortens the critical path.
grep -E 'calib_bci_b|bci4_b' results/phase2_jobs.txt  > results/jobs_phaseA.txt
grep -E 'all_fb_b0[6-9]' results/full_cohort_jobs.txt >> results/jobs_phaseA.txt
say "phase A: $(wc -l < results/jobs_phaseA.txt) jobs (18 IV-2a + 4 filter-bank)"
xargs -d '\n' -P 22 -I{} bash -c "{}" < results/jobs_phaseA.txt >> "$log" 2>&1
say "PHASE_A_DONE"

# ---------------------------------------------------------------- phase B
# These read every subject at once and have no per-subject wait of their own,
# so they are gated on the full cache.
wait_physionet

grep -E 'tr104_c|gram104_|conc104|shots104' results/phase2_jobs.txt > results/jobs_phaseB.txt
say "phase B: $(wc -l < results/jobs_phaseB.txt) full-cohort PhysioNet jobs"
# Held at 12: the six transfer chunks take about 1.3 GB each at 104 subjects,
# and this box has about 17 GB free.
xargs -d '\n' -P 12 -I{} bash -c "{}" < results/jobs_phaseB.txt >> "$log" 2>&1
say "PHASE_B_DONE"

# ---------------------------------------------------------------- phase C
wait_cho

grep -E 'calib_cho_b' results/phase2_jobs.txt > results/jobs_phaseC.txt
say "phase C: $(wc -l < results/jobs_phaseC.txt) Cho2017 calibration jobs"
xargs -d '\n' -P 8 -I{} bash -c "{}" < results/jobs_phaseC.txt >> "$log" 2>&1
say "PHASE_C_DONE"

say "ALL_DONE. Check with: bash scripts/revision_status.sh"
