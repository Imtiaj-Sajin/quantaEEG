# quantaEEG

A controlled benchmark of **quantum and quantum-inspired kernels for EEG
decoding**, built to answer one question honestly: *does quantum structure help
brain–computer interface classification, or does it only look like it does?*

The scientific framing, literature review, and findings live in
[RESEARCH.md](RESEARCH.md). Read that first.

## The core idea

An EEG trial's spatial covariance matrix `C` is symmetric positive definite.
Normalised to unit trace, `ρ = C / tr(C)` **is a quantum density matrix** on
`log2(n_channels)` qubits. That is an identity, not an analogy, so quantum
information geometry applies to EEG directly, with no PCA-into-rotation-angles
step:

| Quantum object | EEG meaning | Hardware primitive |
|---|---|---|
| `tr(ρσ)` | Hilbert–Schmidt overlap of two trials | SWAP test |
| `F(ρ,σ)` | Uhlmann fidelity | SWAP test on purifications |
| `d_Bures` | Bures / Bures–Wasserstein metric |, |
| `S(ρ)` | von Neumann entropy, spatial mixedness |, |

**Scope, stated plainly:** at 8–64 channels all of this is classically
computable in O(n³). This is quantum-information-*geometric* modelling, not a
speedup claim.

## Install

```bash
pip install -r requirements.txt
```

> `scipy` is pinned to `1.15.3`. On some Windows machines, Application Control
> blocks scipy ≥ 1.16's `_batched_linalg` DLL, which breaks the whole stack.

## Run

```bash
# Main benchmark: nested CV, 30 subjects, 15 pipelines
PYTHONPATH=src python -m qeeg.benchmark --subjects 30 --splits 5 --repeats 3

# Kernel concentration vs qubit count (2 -> 6 qubits) on real EEG
PYTHONPATH=src python -m qeeg.concentration --subjects 10
```

Data (PhysioNet EEGMMIDB) downloads automatically via MNE on first run and is
cached in `~/mne_data`.

## What is in the benchmark

15 pipelines in three groups. The **controls** are the point: without them, a
quantum result cannot be interpreted.

**Classical baselines** (tuned with the same CV budget as everything else)
`logvar+LDA` · `CSP+LDA` · `MDM` · `TS+LR` · `TS+RBF-SVM`

**Quantum**
`HS-overlap-SVM` · `Fidelity-SVM`, raw overlaps, kept to document the
concentration pathology
`HS-RBF-SVM` · `Bures-RBF-SVM`, bandwidth-corrected quantum-geometric kernels
`IQP-kernel-SVM` · `CNOT-kernel-SVM`, parameterised circuit embedding kernels

**Controls**
`IQP-no-entangle`: identical circuit with every entangler deleted. If this
matches the entangled version, "quantumness" contributed nothing. (The ablation
from [Bowles et al. 2024](https://arxiv.org/abs/2403.07059).)
`PCA-matched-RBF` · `PCA-matched-linear`: classical kernels on the *same*
4-dimensional features the circuit kernels see, so results are not confounded
with dimensionality reduction.
`logeuclid-TS+LR`: the classical geometry twin of the density-matrix kernels.

## Methodology

- **Nested CV.** Outer 5-fold × 3 repeats for generalisation; inner 4-fold
  `GridSearchCV` for hyperparameters. Applied identically to every pipeline,
  including all classical baselines.
- **Leakage discipline.** Every fitted transform (tangent-space reference mean,
  scalers, PCA, kernel bandwidth) is refit inside each training split.
- **Paired statistics.** Wilcoxon signed-rank across subjects with
  Holm–Bonferroni correction, plus paired Cohen's *d*.
- **Cost is reported.** Wall-clock per pipeline is in the summary table.

## Layout

```
src/qeeg/
  data.py            PhysioNet EEGMMIDB loader, epoching, preprocessing
  quantum.py         density matrices, HS/fidelity/Bures kernels, circuit kernels
  pipelines.py       the 15 pipelines and their controls
  benchmark.py       nested-CV runner + paired statistics
  concentration.py   kernel variance vs qubit count
results/             CSV/JSON outputs (raw folds, summary, tests, metadata)
RESEARCH.md          literature review, findings, publication strategy
```

## Data

PhysioNet **EEG Motor Movement/Imagery** (EEGMMIDB), 109 subjects, 64 channels,
160 Hz. Task: left-fist vs right-fist motor imagery (runs 4/8/12), 8–30 Hz,
0.5–3.5 s post-cue, ~45 trials per subject. Subjects 88, 89, 92, 100 and 104
are excluded for documented annotation/sampling defects.
