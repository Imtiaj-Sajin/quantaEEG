# Revision plan and status

A referee report on the first submission (major revision, 7.5/10, "closer to
the minor end") listed five required items, three recommended ones and a set of
presentation fixes. This file tracks each one and is the skeleton of the
point-by-point response letter the referee asked for. Update the status column
when an item lands; do not delete items, the response letter needs them all.

Statuses: **done**, **running** (compute in flight), **blocked** (needs the
compute to finish), **decision** (needs the corresponding author).

## Tier 1: required for acceptance (7.5 to 8.5)

| # | Item | Status |
|---|------|--------|
| 1 | Run the full PhysioNet cohort (~104 subjects, not the first 30) at 3 and 4 qubits, and use one fixed subject set for every Gram-variance analysis | **running** |
| 2 | Publish the hyperparameter grids: C ranges, bandwidth multipliers, PCA dimensions, angle scales, shrinkage, grid-point counts | **done** |
| 3 | One enumerated definition of the core and extended suites; add QRE-RBF to the core table and figure; one fixed meaning of "best quantum" | **done** |
| 4 | Holm-correct the frame-effect table (was 20 uncorrected paired Wilcoxon tests) | **done** |
| 5 | Cut to journal length: move the heavy tables and figures to supplementary, compress the results section that re-argues the discussion | **partly done**, see below |

Item 1 detail. Stage 1 (every within-subject suite on all 104 eligible
subjects) is complete. Stage 2 is in flight: leave-one-subject-out transfer in
six chunks, plus the Gram diagnostics at four register sizes, the concentration
sweep and the finite-shot run, which are already done and skip themselves.
`scripts/adopt_full_cohort.py` archives the 30-subject files and promotes the
104-subject ones to the canonical names once every input exists; it refuses to
run while anything is missing. The Gram-variance analyses then all read the one
104-subject file, which also settles the n=14 versus n=10 inconsistency the
referee found.

Item 2 detail. `paper/grids_table.py` builds the grid table from the live
`make_grids()` objects rather than from a hand-written list, so it cannot
drift from what the code actually searches.

Item 3 detail. Core is now 16 sensor-frame pipelines (QRE-RBF moved in),
extended is 23 (core plus five reference-frame kernels and two SPD twins).
"Best quantum" means the best density-matrix kernel everywhere.

Item 5 detail, and an honest accounting. Four tables and three figures moved
to a separate supplementary document, which took the build from 26 pages to
24. The referee asked for 15 to 17. The remaining main text is 11 tables, 8
figures and about 9,900 words of prose, and closing a seven-page gap would
mean deleting material that answers the other referee items rather than
padding. The plan, to execute at the final rebuild:

* move `tab:seeds` (the two extra partition seeds, a robustness check by
  nature) to supplementary, and with it the last figure panel that only it
  needs;
* set the panel figures to 0.85 of text width, which costs nothing
  scientifically and recovers roughly a page and a half;
* compress the passage that reads the cross-dataset result twice, once where
  it is measured and again after the frame correction revises it. The second
  reading is the correct one; the first only needs a sentence, not a
  paragraph.

That lands near 20 pages. The response letter should say so plainly: the
document is shorter than the version reviewed despite carrying a cohort three
and a half times larger and two experiments the referee asked for, and we
would rather be told which section to cut than cut the cohort.

## Tier 2: recommended, takes it to 9

| # | Item | Status |
|---|------|--------|
| 6 | Few-trial calibration sweep (10, 20, 40, 80, 160 training trials), quantum kernel versus twin, on the dataset where the reference state is worst estimated | **running** |
| 7 | One more paradigm or class pair, e.g. four-class IV-2a, to show the invariance argument is not specific to left versus right | **queued** |
| 8 | State the proposition's scope: per-subject and per-session whitening extends it to subject- and session-specific congruence, and the residual is what the transfer sections test empirically | **done** |

Item 6 detail. `src/qeeg/calibration.py`. This is the experiment the paper's
own discussion named as the last place a quantum kernel could win, so the
result is worth having whichever way it goes. IV-2a is complete (9 subjects,
750 rows each); Cho2017 is running in eight batches.

Item 7 detail. Nine queued jobs, `bci4_b01` to `bci4_b09`, extended suite,
four-class IV-2a, one subject per process. They start as the transfer and
calibration jobs free up slots.

## Tier 3: presentation

| Item | Status |
|------|--------|
| Unify the confidence-interval convention between the twin figure and the equivalence table | **done**: 90 % everywhere, the interval that matches the two one-sided tests, and the count of intervals excluding zero is now a macro in both places |
| Fix the subtraction in the cross-session paragraph and the accuracy-range mismatch in the transfer paragraph | **done**: both sentences are generated from macros now, so neither number is typed by hand |
| Flag the all-trials reference state in the caption of the cross-session table, not only in the text | **done**: the caption now says the test session's mean uses all of its trials, that this uses test data, and why that is legitimate (no labels, and available to a deployed decoder) |
| Trim the abstract to the journal limit and lead with the methodological contribution | **done**: 296 words, recentring and the twin first, quantum framing second |
| Replace "reproduce their published values" with a reference to the baseline table | **done** |
| A less rhetorical title | **decision** |

Title. The referee suggests, without requiring, a title that states the finding
rather than dramatises it, and notes that the current one front-loads the
quantum part of a paper whose contribution is a control. Current title:

> Quantum Kernels for EEG: An Invariance Mismatch Masquerading as Advantage

The three the referee offered, in increasing plainness:

1. Quantum kernels for EEG motor-imagery decoding match their classical
   counterparts once invariance is matched
2. Recentring is a precondition for comparing quantum and classical covariance
   kernels in EEG decoding
3. No quantum advantage in EEG motor-imagery decoding: an invariance mismatch
   in density-matrix kernels

Only the corresponding author decides this one.

## What the referee did not ask for and we should not add

A fourth dataset. The report asks for breadth in paradigm (item 7), not in
dataset, and the discussion section already argues that mechanism and controls
are what change a reader's mind.

## After the compute finishes

```bash
python scripts/adopt_full_cohort.py          # promote the 104-subject results
PYTHONPATH=src python -m qeeg.equivalence    # TOST against the new numbers
python paper/make_tables.py                  # tables and macros
PYTHONPATH=src python -m qeeg.figures_reference --paper
python paper/check_tex.py
cd paper && latexmk -pdf supplementary.tex && latexmk -pdf main.tex
python paper/make_overleaf_zip.py --check
```

Then reread the results prose against the new numbers. Two paragraphs are
written around how many equivalence intervals exclude zero and which kernels
they belong to; the macros keep the figures right, but the sentence structure
("the first ... the second ...") assumes the count and has to be reread by a
human. Then read the rendered PDF end to end, which `check_tex.py` cannot do
for you.
