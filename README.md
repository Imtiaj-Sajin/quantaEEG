# quantaEEG

A controlled benchmark of **quantum and quantum-inspired kernels for EEG
decoding**, built to answer one question honestly: *does quantum structure help
brain-computer interface classification, or does it only look like it does?*

The short answer is that almost all of the apparent benefit is a coordinate
frame, not quantum geometry, and this repository contains the control that
shows it. There is one exception, in the regime the study named in advance as
the last place a quantum kernel could win, and it is reported at its real size
rather than inflated.

```bash
python run.py verify     # recheck every headline claim from the committed data, in seconds
python run.py status     # what results this checkout contains
```

`verify` needs no GPU, no downloads and no reruns. It reads the result files in
`results/`, recomputes the claims, and prints what it found beside what the
paper says. Start there: it is faster than reading this file.

## What was found

**1. The comparison everyone runs is rigged, and not in a subtle way.**
An EEG trial's spatial covariance `C` is symmetric positive definite, so
`rho = C / tr(C)` *is* a quantum density matrix on `log2(n_channels)` qubits.
That is an identity, not an analogy. But the density-matrix kernels built on it
are invariant only under *orthogonal* congruence, while EEG's nuisances
(electrode gain, head geometry, skull conductivity, source mixing) act by the
*full* congruence group `C -> A C A'`, which is exactly what the classical
Riemannian baselines are invariant to. Comparing them in the sensor frame
measures invariance groups and reports the answer as though it were about
quantum geometry. `run.py verify` demonstrates this numerically: apply one
random congruence and the sensor-frame kernels move by 0.17 to 0.35, while the
reference-frame ones move by 1e-15.

**2. Fixing the frame produces a large, real gain, and it is not quantum.**
Referring every state to a label-free reference state makes the quantum kernels
exactly congruence-invariant (proposition and machine-precision check in
`src/qeeg/reference.py`). Accuracy rises by 0.064 to 0.102 on PhysioNet (all five kernels, n = 104)
and the headline comparison reverses: the quantum kernels now appear to beat the
classical baselines. Published as-is, that would be a quantum-advantage paper.

**3. The control kills it.** `control/riemann-kernel-SVM` is a classical
Riemannian kernel fed the same covariances, in the same support vector machine,
with the same tuning budget, in the same frame. It differs from the quantum
kernels in one respect: the metric. It matches them everywhere. Two one-sided
tests put the two families inside a small equivalence margin across datasets,
register sizes, and within-subject, cross-subject and cross-session evaluation.
The gain was the frame and the kernel-in-an-SVM formulation, neither of which
is quantum.

**4. Entanglement makes things worse, measurably.** Deleting the entanglers
from the circuit kernels *slows* the concentration collapse: entangled IQP
kernels lose Gram variance 2.7 times faster per qubit than the product version
(slopes -0.86 against -0.32 per qubit, n = 104).
This supplies a mechanism for a result other groups have reported empirically.

**5. One regime does favour the quantum side.** With few calibration trials,
where the reference state is poorly estimated, reference-frame quantum kernels
beat the metric-matched twin: at 40 training trials on 52 subjects, all five
kernels lead by about +0.015 and every one survives Holm correction. The effect
disappears by 80 trials and reverses by 160. Against the *best classical
pipeline* rather than the twin, the lead survives only at the smallest training
size. It is one accuracy point, it sits inside the equivalence margin, and the
paper says so in those words.

**6. The quantum kernels cost about three times the wall clock** for this, and
the paper reports that rather than omitting it.

## Why you can believe the negative result

A negative result is only worth reading if the thing being tested was given
every chance. These are the controls, and removing any one of them is how
quantum-advantage claims get manufactured:

| Control | What it rules out |
|---|---|
| Classical baselines tuned with the identical inner CV budget | beating a strawman |
| `control/riemann-kernel-SVM`, a metric-matched twin | crediting the frame or the SVM to quantum geometry |
| `control/IQP-no-entangle`, the same circuit with entanglers deleted | crediting "quantumness" for what the encoding does |
| `control/PCA-matched-*` on the same 4-D features | confounding with dimensionality reduction |
| Paired per-subject Wilcoxon, Holm corrected within pre-specified families | multiplicity |
| Two one-sided tests, not just a non-significant difference | reading absence of evidence as evidence of absence |
| Exact state-vector simulation, no gate noise | an idealisation that can only help the quantum side |

And the scope is stated plainly rather than left ambiguous: at 8 to 64 channels
every kernel here is classically computable in O(n^3). This is
quantum-information-*geometric* modelling. **No speedup is claimed anywhere.**

## Verify the claims yourself

```bash
python run.py verify
```

Reads `results/*.csv` and reprints the paper's claims with live numbers. It
exits non-zero if any claim fails, so it works in CI.

Every number in the manuscript is a LaTeX macro regenerated from these same
CSVs by `paper/make_tables.py`. No figure is typed into the text by hand, so
the paper cannot drift from the data. `paper/check_tex.py` enforces it.

| Claim | File to look at |
|---|---|
| Invariance proposition | `src/qeeg/reference.py`, `check_invariance()` |
| Frame effect | `results/raw_folds_motor8_q4.csv` vs `..._refstate_...` |
| Twin equivalence (TOST) | `results/equivalence_twin.csv` |
| Few-trial calibration | `results/calib_folds_calib_{cho,bci}_b*.csv` |
| Concentration and entanglement | `results/concentration_decay.csv` |
| Cross-subject, cross-session transfer | `results/transfer_folds_*.csv`, `crosssession_folds_*.csv` |
| Per-fold scores for every pipeline | `results/raw_folds_*.csv` |

## Reproduce

```bash
python run.py setup      # dependencies, plus the IOP class files (not on CTAN)
python run.py data       # download the three datasets (several GB, slow once)
python run.py reproduce  # show the compute queue; --run to execute it
python run.py figures    # regenerate every figure
python run.py paper      # tables, macros, static checks, then the PDF
```

Datasets download to MNE's cache. To keep them inside the project instead of on
the system drive:

```bash
python scripts/use_local_datasets.py --check
python scripts/use_local_datasets.py --move
```

`datasets/` is gitignored. The move is hardlink-aware: MOABB builds a
BIDS-style view whose files are hardlinks into the same data, so a naive copy
turns 12 GB into 22 GB.

Long runs are resumable. Every job skips itself if its output exists and
continues from a per-subject checkpoint if it was interrupted, which on this
project has mattered more than once.

## Datasets

Three, run under one protocol, which is what lets the paper separate a property
of the methods from a property of the data. The same eight sensorimotor
channels exist in all three montages, so the register is identical at 3 qubits
and the datasets are directly comparable.

| Dataset | Subjects | Trials | Why it is here |
|---|---|---|---|
| PhysioNet EEGMMIDB | 104 | 45 | the main cohort |
| BCI Competition IV-2a | 9 | 288, two sessions | more trials per subject; cross-session transfer |
| Cho2017 | 52 | 200 | statistical power on the frame and twin comparisons |

## Layout

```
run.py              one entry point: verify, status, setup, data, figures, paper
src/qeeg/
  data.py           loaders, epoching, channel sets
  quantum.py        density matrices, HS/fidelity/Bures/QRE kernels, whitening
  pipelines.py      the pipeline registry: core (16) and extended (23) suites
  reference.py      the invariance proposition and its numerical check
  benchmark.py      nested-CV runner and paired statistics
  transfer.py       cross-subject, leave-one-subject-out
  crosssession.py   within-subject across recording sessions
  calibration.py    few-trial calibration sweep
  filterbank.py     FBCSP-class baselines, 5-qubit registers
  shots.py          finite-shot SWAP-test estimation
  concentration.py  kernel variance against qubit count
  equivalence.py    two one-sided tests against the twin
  figures*.py       every figure, including the study-design overview
results/            per-fold CSVs, summaries and figures: the scientific record
paper/              the manuscript (IOP, Journal of Neural Engineering)
scripts/            job queues, dataset management, analysis helpers
RESEARCH.md         the full research document: literature, gap, findings
REVISION.md         referee items and their status
CLAUDE.md           orientation for a fresh session on any machine
```

## Reading order

1. `python run.py verify`, for the claims and their evidence.
2. [RESEARCH.md](RESEARCH.md), for the argument, the literature and every
   finding in detail.
3. `paper/build/main.pdf`, for the manuscript.
4. [REVISION.md](REVISION.md), for what a referee asked and what was done.

## Environment notes

- `scipy` is pinned to `1.15.3`. On some Windows machines, Application Control
  blocks scipy 1.16's `_batched_linalg` DLL and breaks the whole stack.
- Use `python -u` for long runs; redirected stdout is block-buffered otherwise.
- Set `OMP_NUM_THREADS=1` (and the OpenBLAS and MKL equivalents) before running
  batches in parallel. Five processes each spawning a full BLAS pool ran slower
  than one process until this was set.
