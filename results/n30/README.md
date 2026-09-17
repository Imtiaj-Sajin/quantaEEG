# results/n30: the 30-subject record

What the manuscript reported before the cohort was extended to all 104 eligible
PhysioNet subjects. Kept so the earlier numbers can be reproduced and the two
cohorts compared, not because anything in the current paper depends on it.

Nothing here is read by `paper/make_tables.py`, `run.py verify` or any figure.
The live results are one directory up.

## What changed between the two

| | n30 (this folder) | current |
|---|---|---|
| PhysioNet within-subject | 30 subjects, 6,750 fold rows | 104 subjects, 24,960 rows |
| Reference frame, 3 qubits | 30 subjects | 104 subjects |
| Reference frame, 4 qubits | 30 subjects | 104 subjects |
| Partition seeds 1 and 2 | 30 subjects | 104 subjects |
| Filter bank | 30 subjects | 104 subjects |
| Cross-subject transfer | 30 held-out subjects, 540 rows | 104 held-out, 1,872 rows |
| Finite shots | 30 subjects, 720 rows | 104 subjects, 2,496 rows |
| Gram and concentration diagnostics | 10 to 14 subjects | 104 subjects |

The conclusions did not change. They tightened: the frame effect grew, the
equivalence bounds narrowed, and one comparison that had been one-sided became
two-sided (see REVISION.md).

## A caution about this folder's history

`scripts/archive/adopt_full_cohort.py` archives by moving files here and then re-merges
the batches, so it is not safe to run twice. It was run twice on 2026-09-17,
which moved the already-adopted 104-subject files on top of this archive and
regenerated them identically; for a while this folder held the current cohort
under the old cohort's name. The genuine files were restored from commit
`76a6197`, the last commit before adoption, and the script now refuses to
archive into a non-empty `n30/`.

If you ever need to check this folder is what it claims to be:

```bash
python -c "import pandas as pd; print(pd.read_csv('results/n30/raw_folds_motor8_q4.csv').subject.nunique())"
# 30
```
