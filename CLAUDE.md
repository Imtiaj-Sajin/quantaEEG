# CLAUDE.md: project context

Auto-loaded at the start of every Claude Code session. Keep it current; it is
what makes a fresh session on any machine immediately useful.

## What this project is

A **controlled benchmark of quantum and quantum-inspired kernels for EEG
decoding**. The goal is a publishable, methodologically rigorous answer to:
*does quantum structure actually help brain–computer interface classification?*

It is deliberately **not** an attempt to produce a big accuracy number. The
field is full of "PCA to 8 features → ZZFeatureMap → QSVM → 85%" papers with
weak baselines and no controls. The contribution here is rigour plus a
principled representation.

**Read [RESEARCH.md](RESEARCH.md) first**: it holds the feasibility verdict,
the literature review, the identified gap, all findings, and the publication
strategy. This file is only orientation.

## The core idea

An EEG trial's spatial covariance `C` is symmetric positive definite.
Normalised to unit trace, `ρ = C/tr(C)` **is a quantum density matrix** on
`log2(n_channels)` qubits. That is an identity, not an analogy, so quantum
information geometry (Uhlmann fidelity, Bures metric, von Neumann entropy)
applies to EEG natively, no lossy squeezing of features into rotation angles.
`tr(ρσ)` is exactly what a SWAP test estimates.

**Scope discipline: never overstate this.** At 8–64 channels everything here
is classically computable in O(n³). The claim is *quantum-information-geometric
modelling*, never a speedup. Any wording that implies quantum advantage is
wrong and will (rightly) get the paper rejected.

## Non-negotiable methodology

These exist because removing any one of them is how quantum-advantage claims
get manufactured. Do not weaken them to make results look better.

1. **Classical baselines are tuned with the same CV budget** as the quantum
   models (inner 4-fold `GridSearchCV`). Riemannian tangent-space and CSP are
   the real state of the art, beating a strawman proves nothing.
2. **Entanglement ablation** (`control/IQP-no-entangle`): the identical
   circuit with entanglers deleted. This isolates whether "quantumness"
   contributes anything (Bowles et al., arXiv:2403.07059).
3. **Dimension-matched controls** (`control/PCA-matched-*`): classical kernels
   on the *same* 4-D features the circuit kernels see, so results are not
   confounded with dimensionality reduction.
4. **Paired subject-wise statistics**: Wilcoxon signed-rank + Holm correction.
   Per-subject optimality varies a lot; mean rankings alone mislead.
5. **Report wall-clock cost.** The quantum kernels are far slower for no gain.
   Hiding that would be dishonest.
6. **Be willing to publish a negative result.** The expected outcome is that
   quantum kernels match or lose. That is a real contribution; p-hacking away
   from it is not.

## Layout

```
src/qeeg/
  data.py           PhysioNet EEGMMIDB + MOABB loaders, epoching, channel sets
  quantum.py        density matrices, HS/fidelity/Bures/QRE kernels,
                    reference_whitener, CircuitKernel
  pipelines.py      the pipeline registry; suite="core" (the published 15),
                    "extended" (+8: reference-frame kernels, QRE, SPD-kernel
                    controls)
  reference.py      the invariance proposition + its numerical check, and the
                    sensor-vs-reference concentration diagnostic
  transfer.py       leave-one-subject-out cross-subject transfer
  crosssession.py   within-subject cross-session transfer on IV-2a, both
                    directions, per-session reference states
  filterbank.py     FBCSP-class baselines and 5-qubit wide-register kernels
  shots.py          finite-shot SWAP-test estimation
  benchmark.py      nested-CV runner + paired statistics (--suite core |
                    extended | filterbank)
  equivalence.py    TOST equivalence tests against the classical twin
  concentration.py  kernel variance vs qubit count
  merge.py          combine batched runs
  circuits_qiskit.py  Qiskit feature maps, verified against the PennyLane ones
  figures.py          figures 1-4 (validated palette)
  figures_eeg.py      figure 0: scalp EEG -> covariance -> density matrix
  figures_circuits.py figure 5: the circuit diagrams, rendered from Qiskit
  figures_reference.py figures 6-10: invariance, frame effect, twin, transfer,
                    cross-session + register sweep
results/            CSV/JSON outputs + figures/
paper/              journal manuscript (see below)
RESEARCH.md         the actual research document
```

**The central finding, so a fresh session does not re-derive it.** The
density-matrix kernels were being evaluated in the *sensor* frame while every
strong classical baseline is invariant under congruence `C → ACAᵀ`, the group
EEG's nuisances actually generate. Referring states to a training-set
reference state (`reference_whitener`) makes them exactly affine-invariant,
which relieves concentration and lifts accuracy enough to reverse the headline
comparison, but the `control/riemann-kernel-SVM` twin matches every quantum
kernel to within noise, so the gain is the frame, not quantum structure. Read
RESEARCH.md §4.6–§4.10 before proposing new experiments.

## The manuscript

`paper/` holds a full draft targeting **Journal of Neural Engineering** (IOP,
Q1) on IOP's official `iopjournal` class, with JNE's required structured
abstract (*Objective / Approach / Main results / Significance*).

**The one rule: no number is ever typed into `main.tex` by hand.** Every figure
quoted in the prose is a LaTeX macro (`\PrimaryDelta`, `\BestClassicalAcc`, …)
and all eleven tables are generated from the result CSVs by
`python paper/make_tables.py`. After any new benchmark run, re-run it and the
manuscript is consistent by construction. This is deliberate: the study is
ongoing, numbers will change, and a hand-transcribed manuscript rots silently.

```bash
python paper/make_tables.py    # regenerate tables + inline macros
python paper/check_tex.py      # static checks (no LaTeX toolchain needed)
cd paper && latexmk -pdf main.tex
```

`check_tex.py` catches undefined macros, unresolved citations, dangling
cross-references, unbalanced environments and missing figures, the things that
would otherwise only surface on first compile.

**Compiles.** MiKTeX/`latexmk`: 20 pages, 0 warnings, 0 overfull boxes, 11
tables, 10 figures, 35 references. Run `bash paper/get_iop_class.sh` once
first: it downloads IOP's own `ioplatextemplate.zip` and extracts
`iopjournal.cls` + `orcid.pdf` (neither is on CTAN).

**The class was migrated from `iopart` to `iopjournal` on 2026-09-05.**
`iopart` is legacy and is not in IOP's current package at all; ours had come
from a third-party mirror that could not be verified against anything. Do not
revert. `iopjournal` defines none of the iopart-isms (`\sref`, `\eref`,
`\submitto`, `\ead`, `\address`, `indented`, `\br`/`\mr`) and adds
`\articletype`, `\orcid`, `\funding`, `\roles`, `\data`, plus an
`[anonymous]` option for double-anonymous review. Mapping table in
`paper/README.md`.

Note the `-outdir` BibTeX trap documented in `paper/README.md`: without
`BIBINPUTS` set, BibTeX silently produces an empty bibliography and the error
surfaces as a misleading `missing \item` from `main.bbl`.

`check_tex.py` is a pre-flight, not a substitute for compiling, and the
migration proved it three times over. It cannot see: an undefined macro from a
package the class does not load; a heading the class emits itself (our
`\section*{References}` printed twice); a bibliography style that *typesets*
the `note` field; or `$<$0.001` nested inside `$p=...$`, which closes math mode
and renders `<` as `¡`. All four shipped in a build with a clean exit code.
**Read the rendered PDF, not just the log.**

**Before submitting**, work through `paper/README.md`'s checklist. Reference
provenance now lives in a `verified` field, NOT `note`: `iopart-num`
*typesets* `note`, so bookkeeping there was printing into the bibliography of
the submitted PDF. No entry carries `[CHECK]` any more; two (`holm1979`,
`demsar2006`) are `[NO DOI]` because they genuinely have none and still want
a manual eyeball.

## How to run

```bash
pip install -r requirements.txt

# Main benchmark (~70 s/subject after data is cached)
PYTHONPATH=src python -u -m qeeg.benchmark --subjects 30 --splits 5 --repeats 3

# Kernel concentration vs qubit count (2 -> 6 qubits)
PYTHONPATH=src python -u -m qeeg.concentration --subjects 10

# Figures
PYTHONPATH=src python -m qeeg.figures
```

Long runs can be split and merged:

```bash
PYTHONPATH=src python -u -m qeeg.benchmark --subject-list 1,2,3,4 \
    --tag batch01 --no-stats
PYTHONPATH=src python -m qeeg.merge --pattern "raw_folds_batch*.csv"
```

## Environment gotchas (all hit during development)

- **scipy is pinned to `1.15.3`.** On this Windows machine, Application Control
  blocks scipy ≥1.16's `_batched_linalg` DLL, which breaks the entire stack with
  an opaque `ImportError`. If scipy gets upgraded and imports start failing,
  this is why.
- **Always use `python -u` for long runs.** Redirected stdout is block-buffered;
  without `-u` a multi-hour run shows zero progress and its output is lost if
  the process is killed. The runners also force `flush=True`.
- **PhysioNet downloads are slow** (~50 s/file, 3 files/subject). First run of
  30 subjects spends ~40 min downloading. `prefetch.py` warms the cache with a
  thread pool; data caches to `~/mne_data` and is reused thereafter.
- **Set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` for any
  parallel batch.** Each benchmark process otherwise spawns a full BLAS thread
  pool; five processes on twelve cores ran slower than one until this was set.
  `scripts/run_when_cached.sh` does it for the Cho2017 batches.
- **Check for duplicate runs.** `tasklist`/`ps` under Git Bash have returned
  empty output unreliably here; verify with PowerShell
  `Get-CimInstance Win32_Process` before concluding a process died. Two
  concurrent runs will corrupt each other's output files.
- Subjects 88, 89, 92, 100, 104 are excluded (documented EEGMMIDB defects).

## Datasets

Two, run under one protocol, which is what lets the paper separate a property
of the methods from a property of the data:

- **PhysioNet EEGMMIDB** (`--dataset physionet`), 30 subjects, 45 trials each.
  Downloads via MNE, slow first time, caches to `~/mne_data`.
- **BCI Competition IV-2a** (`--dataset bci2a`), 9 subjects, 288 trials each
  over two sessions, via MOABB. ~83 MB per subject, ~6 min per subject to
  evaluate. The two sessions are what `crosssession.py` uses.
- **Cho2017** (`--dataset cho2017`), 52 subjects, 64 channels, 200 trials,
  one session, via MOABB (~190 MB per subject, ~75 s to fetch). Added for
  statistical power on the frame/twin comparison, not as a dataset for its
  own sake.

The same 8 sensorimotor channels exist in all three montages, so the register
size is identical at 3 qubits and the datasets are directly comparable.

## Findings so far

1. **Quantum overlap kernels concentrate catastrophically on EEG**, pairwise
   `tr(ρσ)` ≈ 0.99 ± 0.01, Gram matrix effectively rank-one. This reproduces
   the Thanasilp et al. (*Nat. Commun.* 2024) pathology on real biological
   data. Remedy implemented: bandwidth-parameterised kernels
   `exp(−γ·d²_Bures)`, γ from the median heuristic on training splits only.
2. **Entanglement accelerates the collapse** (n=10). Entangled IQP kernels lose
   variance at 0.428× per qubit vs 0.716× with entanglers removed, 2.5× faster
   in log-slope, 29× vs 3.7× loss over 2→6 qubits. This supplies a *mechanism*
   for why deleting entanglement improves QML models.
3. **More data widens the gap in the sensor frame.** On IV-2a (288 trials) the
   classical-quantum gap is 0.079 versus 0.032 on PhysioNet (45 trials). This
   kills the "the quantum model was starved" defence. Ranking is stable across
   the two datasets (Spearman 0.882). **Reinterpreted in §4.10:** a kernel with
   the wrong invariance cannot use extra data, while the classical baselines
   can. Give it the right frame and the extra data becomes *more* valuable to
   the quantum kernels than to the classical ones, which is why the frame
   correction is worth 2-3× more on IV-2a.
4. **Two distinct concentration mechanisms.** Circuit kernels concentrate from
   qubit count; density-matrix kernels concentrate from the *data*. Most of
   the latter is an artefact of the sensor frame (variance rises 4.7-9.5× on
   recentring at 3 qubits). **Settled in §4.12:** the sweep to 6 qubits in
   both frames shows reference-frame variance never falls below its 3-qubit
   value (1.6-2.4× at 6q), so the "run it wider" corollary survives. Say
   "never falls below", not "rises monotonically": three kernels dip ~7 %
   from 5q to 6q.
5. **The frame is the whole effect (§4.6-§4.11).** Recentring reverses the
   headline comparison on both datasets, but the metric-matched classical twin
   matches every quantum kernel: TOST puts the two families within ±0.032
   accuracy across all 25 comparisons in five settings. The gain is the frame
   and the SPD-kernel-in-an-SVM formulation, neither of which is quantum.
6. **Cross-session transfer (§4.11) is parity too**, in the setting the paper
   had named as the most promising for a real quantum effect. Sensor-frame
   quantum kernels sit near chance (0.54-0.58) while classical baselines hold
   0.70-0.73; per-session recentring is worth +0.17 to +0.22 (9/9 subjects)
   and then quantum minus twin is within ±0.009, p ≥ 0.5.

## Where to take it next

Everything the argument needs is done: PhysioNet at 3 and 5 qubits, IV-2a,
cross-subject and cross-session transfer, the register sweep to 6 qubits in
both frames, filter-bank/FBCSP baselines, shot noise, and TOST equivalence.

Robustness runs launched 2026-09-14 (see RESEARCH.md §9 for how to harvest):
seeds 1 and 2 of the PhysioNet extended suite (does the twin comparison hold
under different fold splits?), the extended suite at 16 channels / 4 qubits
(`raw_folds_ref16*.csv`, merge with `qeeg.merge`), and Cho2017 (52 subjects,
`--dataset cho2017`) for a better-powered frame/twin comparison. None of these
is needed for the argument; they close the "one seed, two datasets" objection.

Outstanding, in order:

1. **Author confirmation.** Order, and the CRediT `\roles{}` draft in
   `main.tex`, are marked NOT FINAL. Only the corresponding author can settle
   who did what.
2. **`\funding{}`** currently states no specific grant. Correct it if that is
   wrong; IOP parse that section.
3. **Read the PDF end to end** after each rebuild. `check_tex.py` cannot see
   rendering problems.

Do **not** add more datasets beyond Cho2017. Mechanism and controls change a
reviewer's mind; a fourth dataset does not.
