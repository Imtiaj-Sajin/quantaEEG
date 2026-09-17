#!/usr/bin/env bash
# Referee item 9: Table 3 has 15 rows while the caption and the Spearman
# correlation say 16. The cause is that quantum/QRE-RBF-SVM was added to the
# core suite after IV-2a had been run, so PhysioNet core carries 16 pipelines
# and IV-2a core carries 15.
#
# This re-runs the nine IV-2a subjects with the current 16-pipeline core suite,
# at the settings recorded in results/meta_bci2a_motor8_q4.json (seed 0,
# 5-fold x 3 repeats, motor8, 4 qubits), writing to a NEW tag.
#
# Nothing is adopted automatically. scripts/adopt_bci2a_core16.py diffs the 15
# overlapping pipelines against the published file first: if any published
# number moves, that is a reproducibility problem to understand, not a result
# to publish.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=src

log=results/run_bci2a_core16.log
echo "start $(date)" > "$log"

python -u -m qeeg.benchmark \
    --dataset bci2a --suite core --channels motor8 --qubits 4 \
    --splits 5 --repeats 3 --seed 0 \
    --tag bci2a_core16 --no-stats --resume >> "$log" 2>&1
echo "JOB_EXIT=$?" >> "$log"
echo "done $(date)" >> "$log"
