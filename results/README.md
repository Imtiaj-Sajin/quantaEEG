# results/

Every number in the manuscript comes from a file here. Nothing is typed by
hand: `paper/make_tables.py` regenerates all tables and inline macros from
these CSVs, and `python run.py verify` recomputes the headline claims from them
in a few seconds.

## Naming

```
raw_folds_<tag>.csv      one row per (subject, pipeline, fold): the primary record
summary_<tag>.csv        per-pipeline means over subjects
tests_vs_<ref>_<tag>.csv paired Wilcoxon against a reference pipeline, Holm corrected
meta_<tag>.json          what was run: suite, seeds, channels, timing
```

A `refstate_` tag is the reference frame; without it, the sensor frame.
A `_q4` suffix is the extended suite. `motor16` is the 16-channel, 4-qubit
register; `motor8` is the 8-channel, 3-qubit one.

## The files the paper reads

| File | Content | Used for |
|---|---|---|
| `raw_folds_motor8_q4.csv` | PhysioNet, 104 subjects, sensor frame | table 1, figures 5 and 6 |
| `raw_folds_refstate_motor8_q4.csv` | the same in the reference frame | the frame effect, the twin comparison |
| `raw_folds_refstate_motor8_q4_seed{1,2}.csv` | two further outer partitions | the robustness table |
| `raw_folds_refstate_motor16_q4.csv` | 16 channels, 4 qubits | the register comparison |
| `raw_folds_bci2a_motor8_q4.csv` + `refstate_` | BCI IV-2a, 9 subjects | the replication |
| `raw_folds_refstate_cho2017_motor8_q4.csv` | Cho2017, 52 subjects | the best-powered twin test |
| `raw_folds_refstate_bci2a4_motor8_q4.csv` | four-class IV-2a | the four-class section |
| `raw_folds_filterbank_motor8.csv` | FBCSP-class baselines, 5 qubits | the filter-bank section |
| `transfer_folds_motor8.csv` | leave-one-subject-out, both frames | cross-subject transfer |
| `crosssession_folds_bci2a_motor8.csv` | session to session within subject | cross-session transfer |
| `calib_folds_{bci2a,cho2017}.csv` | few-trial calibration sweep | the one setting that separates |
| `concentration_*.csv` | Gram variance against qubit count | the concentration section |
| `reference_gram_*.csv` | Gram statistics per frame and register | the frame diagnostic |
| `shots_folds_motor8.csv` | accuracy against shot budget | finite-shot estimation |
| `equivalence_twin.csv` | two one-sided tests against the twin | the equivalence table |

## Folders

| Folder | What it holds |
|---|---|
| `figures_paper/` | the figures the manuscript includes, rendered in paper mode |
| `figures/` | the same figures in standalone mode, plus PNGs for quick viewing |
| `n30/` | the 30-subject record from before the cohort was extended; see its own README |

## What is not here, on purpose

Per-batch intermediates. Long runs were split into batches
(`raw_folds_all_core_b01.csv` and so on) and merged into the canonical files
above. The merges were verified lossless, row for row, before the batches were
removed; they are build intermediates, and they are in git history if a
per-batch question ever arises. The `meta_*.json` provenance for each merged
tag is kept.

Logs are gitignored. So is `datasets/`, the 12 GB EEG cache, which
`scripts/use_local_datasets.py` manages.

## Checking a file is what it says

```bash
python -c "import pandas as pd; d=pd.read_csv('results/raw_folds_motor8_q4.csv'); print(d.subject.nunique(), 'subjects,', len(d), 'rows,', d.pipeline.nunique(), 'pipelines')"
# 104 subjects, 24960 rows, 16 pipelines
```
