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
  data.py           PhysioNet EEGMMIDB loader, epoching, channel sets
  quantum.py        density matrices, HS/fidelity/Bures kernels, CircuitKernel
  pipelines.py      the 15 pipelines (classical / quantum / control)
  benchmark.py      nested-CV runner + paired statistics
  concentration.py  kernel variance vs qubit count
  merge.py          combine batched runs
  figures.py        publication figures (validated palette)
results/            CSV/JSON outputs + figures/
paper/              journal manuscript (see below)
RESEARCH.md         the actual research document
```

## The manuscript

`paper/` holds a full draft targeting **Journal of Neural Engineering** (IOP,
Q1) in `iopart` format with JNE's required structured abstract.

**The one rule: no number is ever typed into `main.tex` by hand.** Every figure
quoted in the prose is a LaTeX macro (`\PrimaryDelta`, `\BestClassicalAcc`, …)
and all four tables are generated from the result CSVs by
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

**Not yet compiled.** No LaTeX toolchain on this machine; static checks pass but
the draft has never been run through `iopart.cls`. Expect minor first-compile
fixes.

**Before submitting**, work through `paper/README.md`'s checklist. The most
important item: `refs.bib` marks each entry `[VERIFIED]` (checked against the
publisher record during this study) or `[CHECK]` (canonical work cited from
standing knowledge, the paper is right, the volume/page metadata was not
re-checked). Every `[CHECK]` needs a DOI lookup.

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
- **Check for duplicate runs.** `tasklist`/`ps` under Git Bash have returned
  empty output unreliably here; verify with PowerShell
  `Get-CimInstance Win32_Process` before concluding a process died. Two
  concurrent runs will corrupt each other's output files.
- Subjects 88, 89, 92, 100, 104 are excluded (documented EEGMMIDB defects).

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
3. **Two distinct concentration mechanisms.** Circuit kernels concentrate from
   qubit count; density-matrix kernels concentrate from the *data* (EEG
   covariances are intrinsically similar), dimension-independent, and actually
   *relieved* by adding channels. Corollary: the density-matrix route should
   use more channels, not fewer.

## Where to take it next

Highest value first (see RESEARCH.md §6, §9):

- **Cross-subject transfer**, not within-subject accuracy. Hypothesis: quantum
  distances (Bures, quantum relative entropy) are more robust to inter-subject
  covariance shift than affine-invariant Riemannian ones. This targets the
  field's actual bottleneck (the calibration problem).
- Add MOABB + BCI Competition IV-2a for comparability with published claims.
- Shot-noise simulation: everything is currently infinite-shot.
- Quantum relative entropy `S(ρ‖σ)` as an additional divergence.
