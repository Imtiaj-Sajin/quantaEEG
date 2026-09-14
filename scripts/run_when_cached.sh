#!/usr/bin/env bash
# Run one Cho2017 benchmark batch once the prefetch has cached every subject
# it needs. Waiting here (rather than letting the batch download on demand)
# avoids two processes fetching the same file at once, which corrupts the
# MOABB cache.
#
#   scripts/run_when_cached.sh <max_subject> <subject-list> <tag> [extra args]
#
# The prefetch log (results/run_cho_prefetch.log) prints "ok <subject> <sec>"
# in subject order, so the count of ok lines is the highest cached subject.
set -u
need="$1"; subjects="$2"; tag="$3"; shift 3
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
log="results/run_${tag}.log"
while :; do
    have=$(grep -c '^ok' results/run_cho_prefetch.log 2>/dev/null || echo 0)
    if [ "$have" -ge "$need" ]; then break; fi
    if grep -q '^CHO_DONE' results/run_cho_prefetch.log 2>/dev/null; then break; fi
    sleep 60
done
echo "start $(date) after prefetch reached $have (needed $need)" > "$log"
PYTHONPATH=src python -u -m qeeg.benchmark --dataset cho2017 --suite extended \
    --channels motor8 --subject-list "$subjects" --tag "$tag" --no-stats "$@" >> "$log" 2>&1
echo "BATCH_EXIT=$?" >> "$log"
