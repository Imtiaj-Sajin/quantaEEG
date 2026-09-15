#!/usr/bin/env bash
# One benchmark batch of the full PhysioNet cohort.
#
#   scripts/cohort_job.sh TAG SUBJECT_LIST [extra qeeg.benchmark args ...]
#
# Waits until every EDF file its subjects need is in the MNE cache (so it can
# be queued while prefetch.py is still downloading), skips itself if its
# output already exists (so the queue can be restarted safely), and runs with
# one BLAS thread per process (so ten batches share twelve cores cleanly).
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

tag="$1"; subs="$2"; shift 2
log="results/run_${tag}.log"

if [ -f "results/raw_folds_${tag}.csv" ]; then
    echo "skip ${tag}: already complete"
    exit 0
fi

cache="$HOME/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0"
while :; do
    missing=0
    for s in ${subs//,/ }; do
        n=$(printf "S%03d" "$s")
        for r in 04 08 12; do
            [ -f "$cache/$n/${n}R${r}.edf" ] || missing=1
        done
    done
    [ "$missing" = 0 ] && break
    sleep 60
done

echo "start $(date)" > "$log"
PYTHONPATH=src python -u -m qeeg.benchmark --subject-list "$subs" --tag "$tag" \
    --no-stats --resume "$@" >> "$log" 2>&1
echo "JOB_EXIT=$? $(date)" >> "$log"
