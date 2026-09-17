# scripts/archive: the compute that has already run

These produced the results in `results/` and are kept as the record of how,
not because anything still needs them. The outputs they generate are all
present and merged, so every one of them would now skip itself or refuse.

Do not run `adopt_full_cohort.py` again. See the note below.

| Script | What it did |
|---|---|
| `run_revision.sh` | the whole revision compute in order, resumable; the entry point the Windows scheduled task called |
| `full_cohort_queue.py` | wrote the job list that reran every PhysioNet suite on all 104 subjects |
| `cohort_job.sh` | ran one benchmark batch from that queue |
| `phase2_jobs.py` | job list for the second stage: Gram diagnostics, concentration, shots, transfer, calibration, four-class |
| `resplit_transfer.py` | re-split the remaining transfer work across more processes when the rest of the queue drained |
| `run_resplit.sh` | ran that re-split queue |
| `run_resplit_missing.sh` | restarted the three transfer jobs that died of out-of-memory, staggered so their allocation peaks did not coincide |
| `run_when_cached.sh` | ran a Cho2017 batch once the prefetch had cached its subjects |
| `run_here.sh` | the same compute on the second machine, which had no warm dataset cache |
| `adopt_full_cohort.py` | archived the 30-subject results into `results/n30/` and promoted the 104-subject ones to the canonical names |

## Why adopt_full_cohort.py must not be run again

It archives by moving `results/*.csv` into `n30/` and then re-merges the
batches. Running it a second time moves the already-adopted 104-subject files
on top of the 30-subject archive and regenerates them identically, so `n30/`
ends up holding the current cohort under the old cohort's name. That happened
on 2026-09-17 and the archive had to be restored from the commit before
adoption. The script now refuses to archive into a non-empty `n30/`, but the
safest thing is simply not to run it: the adoption is done.

## What is still live, one directory up

| Script | Why it stays |
|---|---|
| `use_local_datasets.py` | moves the 12 GB EEG cache into the gitignored `datasets/`; this is how a fresh machine is set up |
| `revision_status.sh` | read-only progress check, useful after any interruption |
| `calib_full.py`, `calib_levels.py`, `calib_trend.py` | the few-trial calibration analyses behind a reported result, not job plumbing |
