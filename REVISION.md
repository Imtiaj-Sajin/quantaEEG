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
| 1 | Run the full PhysioNet cohort (~104 subjects, not the first 30) at 3 and 4 qubits, and use one fixed subject set for every Gram-variance analysis | **done** |
| 2 | Publish the hyperparameter grids: C ranges, bandwidth multipliers, PCA dimensions, angle scales, shrinkage, grid-point counts | **done** |
| 3 | One enumerated definition of the core and extended suites; add QRE-RBF to the core table and figure; one fixed meaning of "best quantum" | **done** |
| 4 | Holm-correct the frame-effect table (was 20 uncorrected paired Wilcoxon tests) | **done** |
| 5 | Cut to journal length: move the heavy tables and figures to supplementary, compress the results section that re-argues the discussion | **partly done**, see below |

Item 1 detail. Stage 1 (every within-subject suite on all 104 eligible
subjects) is complete. Stage 2 is in flight: leave-one-subject-out transfer in
six chunks, plus the Gram diagnostics at four register sizes, the concentration
sweep and the finite-shot run, which are already done and skip themselves.
`scripts/archive/adopt_full_cohort.py` archives the 30-subject files and promotes the
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
| 6 | Few-trial calibration sweep (10, 20, 40, 80, 160 training trials), quantum kernel versus twin, on the dataset where the reference state is worst estimated | **done** |
| 7 | One more paradigm or class pair, e.g. four-class IV-2a, to show the invariance argument is not specific to left versus right | **done**, and written up as section 3.9 |
| 8 | State the proposition's scope: per-subject and per-session whitening extends it to subject- and session-specific congruence, and the residual is what the transfer sections test empirically | **done** |

Item 6 detail. `src/qeeg/calibration.py`. This is the experiment the paper's
own discussion named as the last place a quantum kernel could win, so the
result is worth having whichever way it goes. IV-2a is complete (9 subjects,
750 rows each); Cho2017 is running in eight batches.

**Early result, and it points the way the paper predicted.** The
quantum-minus-twin difference is largest at the smallest calibration set and
decays towards zero as training trials are added. On IV-2a at 10 training
trials the best reference-frame kernel leads the twin by +0.014 (8 of 9
subjects, p = 0.035 uncorrected) and the best classical pipeline by +0.024
(p = 0.027); by 40 trials and beyond the difference is within ±0.007 with
mixed signs.

Reporting this correctly matters more than the result itself, because a
"best kernel at the best training size" comparison is precisely the selection
effect the twin control was built to prevent. So the pre-specified quantity is
the *trend*: per subject and per kernel, the slope of (quantum minus twin)
against log2(training size), which is one test per kernel, Holm corrected over
the five. All five slopes are negative on both datasets:

| Dataset | slope per doubling | subjects negative | p (uncorrected) | Holm |
|---|---|---|---|---|
| IV-2a, n=9 | -0.0028 to -0.0044 | 6 to 8 of 9 | 0.039 to 0.164 | 0.20 to 0.39 |
| Cho2017, first 14 of 52 | -0.0058 to -0.0081 | 10 to 11 of 14 | 0.049 to 0.135 | 0.25 |

Nothing here survives correction yet, and the effect at its largest is inside
the ±0.02 equivalence margin. The negative control is what makes it worth
taking seriously: the twin's own advantage over the best classical pipeline
shows no such trend (IV-2a slope -0.002, p = 0.20; Cho2017 +0.0001, p = 0.86),
so this is not the generic convergence of every method as data grows. The
decisive test is the full 52-subject Cho2017 run now in flight. Write it up as
a size-dependent trend with the sign the paper predicted in advance, not as an
advantage, and quote the Holm-corrected numbers.

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
python scripts/archive/adopt_full_cohort.py          # promote the 104-subject results
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


## Added 2026-09-17, after the corresponding author read the rebuilt draft

Three criticisms, all fair, all acted on:

* **Recent literature was thin.** The submitted draft engaged with two
  quantum-EEG papers. The bibliography now carries seven more, and the
  introduction argues against two of them specifically rather than citing a
  list: QEEGNet (IEEE SiPS 2024), whose reported gain is 0.381 against EEGNet's
  0.377 on four-class IV-2a, over one classical architecture and far below what
  filter-bank and Riemannian pipelines reach; and Carter et al (Mayo Clinic
  Proceedings 2026), who reach 0.898 on binary motor imagery with digitised
  counterdiabatic quantum features and report, to their credit, that those
  features change only 244 of 30,590 classifications and do not reach
  significance over their own classical model overall. Both were read in full;
  the other five are cited as evidence of the literature's extent, which is all
  the text claims of them. The QEEGNet entry also cited the preprint rather
  than the published version, which is now fixed.

* **Tables read like a dump of the code.** Every pipeline name printed its
  registry prefix, so the first column of every table repeated the word that
  the Group column already carried. Names are now typeset (`CSP + LDA`, not
  `classical/CSP+LDA`), and the IV-2a table gets its Group column back, which
  had been dropped only because the prefixed names overflowed the text block.

* **A caption stated something false.** The pre-specified comparison table
  appended "the others are not" unconditionally, so with all three contrasts
  significant it named all three and then referred to others that do not exist.

### Still open from that review

* **Table 3 duplicates Table 5.** The three pre-specified comparisons on
  PhysioNet appear in full in the both-datasets table two pages later. One of
  them should go, and the early one is the candidate: its three numbers are
  already macros and can be stated in the prose. This removes a table, removes
  the duplication and shortens the paper, but it changes the structure of the
  results section, so it waits for the corresponding author.
* **The four-class IV-2a result is computed but not written up.** Nine subjects,
  extended suite, sitting in `results/raw_folds_bci4_b0?.csv`.
* **A full read of the rendered PDF.** The static checker cannot see rendering
  problems, and this draft has not yet been read end to end at its new length.


## Status as of 2026-09-17, after the full reviewer pass

Every Tier 1 and Tier 2 item is done and written into the manuscript. The
compute is finished; no job is queued or running and the scheduled tasks have
been removed.

Two items still need the corresponding author, and nothing else does.

1. **The title.** The referee suggested, without requiring, something less
   rhetorical than "An Invariance Mismatch Masquerading as Advantage". Three
   alternatives are listed above. Only the corresponding author decides this.
2. **`unding{}`** states that no specific grant was received. Correct it if
   that is wrong; IOP parse that section.

Item 5, length, is the one place we deliberately did not do what the referee
asked. The manuscript is 28 pages against a request for 15 to 17, and it is
longer than the reviewed version despite four tables and three figures moving
to supplementary, because the cohort grew three and a half times and two
requested experiments were added. The response letter should say so plainly and
invite the editor to name a section to cut rather than cutting the cohort.

### What the reviewer pass found and fixed

Recorded because these are the failure modes to check for again after any
future change:

* Four statements in the discussion and limitations still described the study
  as it stood before the four-class and calibration experiments existed,
  including one that called the calibration regime "untried".
* The abstract asserted that exactly one equivalence interval excluded zero and
  that it favoured a quantum kernel; three do, and one favours the twin.
* Methods said "the first 104 eligible subjects" when 104 is every eligible
  subject, which reads like the arbitrary selection the referee objected to.
* Figures 1 to 4 had never been regenerated for the 104-subject cohort, so the
  tables were at n=104 while the figures plotted n=30 from the same files.
* `submission.zip` shipped a zero-byte bibliography, and `--check` could not
  see it because latexmk reran BibTeX in the temp directory.
* Six numbers in the concentration section were typed by hand and had gone
  stale; they are macros now.
* `results/n30/` held the current cohort under the old cohort's name.

## Second review: minor revision, 8.5/10 (2026-09-17)

The referee read the rebuilt PDF and the repository and asked for a
consistency pass: accept on receipt if items 1 to 12 are addressed, no new
experiments required. Every item was checked against the source before it was
touched, and every one was real. Most were prose around generated numbers
that the rerun had changed underneath, so where a sentence asserted a fact
about the numbers, the fact is now generated too.

| # | Referee item | What changed | Status |
|---|---|---|---|
| 1 | Section 3.2 says neither control is significant, then reports p < 0.001 for both | The verdict is generated from the two p-values (`\ControlsVerdict`) and the paragraph rewritten | done |
| 2 | The twin is called the second-best pipeline; four are ahead of it | Rank, total and number ahead are generated (`\TwinRank`, `\TwinRankOf`, `\TwinAhead`) | done |
| 3 | "Widens from 0.028 to 0.022"; Table 8 caption "never in their favour" | Primary-partition bound is its own macro (0.018); the caption is generated per kernel family and labels its significance as uncorrected | done |
| 4 | Qualifications skip HS-RBF at 4 qubits; "does not survive its family" with no corrected p | All three off-zero intervals are listed with per-setting Holm p. Two survive (Fidelity, PhysioNet 0.015 and Cho2017 0.007), one does not (HS-RBF 4q, 0.242). The pooled 25-test family is also reported: Cho2017 survives (0.036), PhysioNet does not (0.070). The false "changing sign between partitions" argument is replaced by the fact that Fidelity is ahead in all three partitions, significant after Holm in one. "One dataset" is now two | done |
| 5 | Section 3.6 calls p = 0.097 "conventionally significant", "thirty subjects", "the same picture" | Paragraph rewritten: at four qubits the twin is ahead of every quantum kernel; verdicts and sign are generated | done |
| 6 | Section 3.10: 0.029 "inside" a 0.02 margin; correction for "11 in favour, 0 against" | 0.029 is the largest bound and is stated as outside the margin; the largest mean difference (+0.016) is the one inside. Counts are given three ways: 11 uncorrected, 7 after per-size Holm, 3 pooled over all 50, none against. "Only at the smallest training set" was also wrong (Cho2017 k = 10 and 20); now generated | done |
| 7 | Abstract says no kernel beats the best classical pipeline anywhere; Section 3.10 says otherwise | Abstract, introduction, discussion and conclusion now say "no quantum advantage of practical size" and name the few-trial exception. Abstract 297 words | done |
| 8 | Two "third candidate"s in Section 4.4 | The second is now the second; the section also names both datasets with the fidelity separation | done |
| 9 | Figure 7 axis says 30 subjects; Table 3 has 15 IV-2a rows against 16 claimed | Axis label is read from the result files. IV-2a rerun with the 16-pipeline core suite; `scripts/adopt_bci2a_core16.py` requires the 15 shared pipelines to reproduce the published accuracies exactly before adding the QRE-RBF rows. They did, in all 2,025 fold rows (four circuit-kernel folds differ in AUC by one tied pair); the 135 new rows were appended to the published bytes. Table 3 has 16 rows, Spearman 0.877 over 16 | done |
| 10 | Figure 2 caption: "one-second window", "five evaluation settings" | Each dataset's own window is described; the settings count is the generated macro | done |
| 11 | "p <= < 0.001", "up to 0 %", "pre-specified +-0.02 margin" | Max-p macros carry their own relation; the dip is found at whichever register step it occurs; the caption says not pre-registered. `check_tex.py` now fails on all of these (verified against the committed version, which it rejects on six counts) | done |
| 12 | References [6], [8], [9], [13] incomplete | Completed from Crossref and PubMed. QEEGNet was typed `@article` so its SiPS venue was silently dropped; Carter et al is ahead of print and marked in press with its DOI. The check also found wrong first names in three entries (initials unchanged) and quEEGNet misspelled as qEEGNet | done |
| 13 | Garg and Garg paragraph out of place in Results | Moved to Discussion, section 4.2 | done |

Found while fixing these, and fixed with them:

* **`submission.zip` did not contain the supplement.** It carried
  `supplementary.aux` so the cross-references resolved, but not the
  supplement's source, tables, PDF or its four figures. The referee
  guessed as much ("make sure the supplement actually arrived").
* The IV-2a rank-floor paragraph typed "ten of the fourteen" and multiplied the
  floor by a hard-coded 14 while quoting PhysioNet's family size. Generated now.
* Section 3.5 and the Discussion both claimed the reversal was "a publishable
  claim of quantum advantage at p = 0.095" with the effect "growing on the
  better-powered" dataset. At n = 104 it is not significant and is largest on
  PhysioNet. The duplicate in Section 3.5 is gone; the Discussion version is
  corrected.
* Negative numbers quoted in running text printed with a hyphen; the macros
  now carry a true minus.
* `use_local_datasets.py` crashed on a fresh clone outside Windows, and the
  README called it optional when it must run before the download to keep the
  data in `datasets/`.

Deliberately not done, and why (for the response letter):

* **The 15 % length cut.** Not an acceptance condition. The genuine duplication
  in Section 3.5 was removed instead.
* **Moving Figure 2 to the supplement.** The referee wrote "could"; it is the
  reader's only map of the design.
* **Removing CLAUDE.md.** It stays. An AI-use disclosure is the right response
  and needs the corresponding author's wording.

Still needing the corresponding author: the AI-use disclosure statement, the
title, `\funding{}`, and (suggested) a Zenodo DOI for a tagged release.
