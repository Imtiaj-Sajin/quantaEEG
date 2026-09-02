# Quantum Machine Learning for EEG: Research Log

**Status:** living document. Started 2026-09-02.
**Question being answered:** is "quantum + EEG" a real research programme or a fascination?

---

## 0. Verdict up front

**The idea is feasible and publishable, but not in the form most papers in this
area take it.**

Two things are true at once, and keeping both in view is the whole game:

1. **The naive version is a dead end.** "Take EEG, PCA it down to 4–8 features,
   shove them into a `ZZFeatureMap`, call it a QSVM, report 85% accuracy" is a
   saturated, low-credibility genre. Dozens of such papers exist. They almost
   never include the controls that would tell you whether the quantum part did
   anything, and the field's own large-scale benchmarks say it usually doesn't.
2. **There is a genuinely underexplored, principled bridge.** EEG's dominant
   modern representation is the **spatial covariance matrix**, which is
   symmetric positive definite. A trace-normalised SPD matrix **is a quantum
   density matrix**. That is not an analogy: it is an identity. It means the
   entire apparatus of quantum information geometry (Uhlmann fidelity, Bures
   metric, von Neumann entropy, quantum relative entropy) applies to EEG
   *natively*, with no lossy feature-squeezing step. This connection is
   visible in the literature but, as far as this review found, **has not been
   developed into a kernel method and benchmarked properly.**

So: publishable, yes, if the paper's contribution is *rigour plus a principled
representation*, not a leaderboard number. See §6 for what that paper looks
like and §7 for the risk that the honest answer is negative.

**One caution to internalise now:** the most likely outcome of a well-controlled
study here is that quantum methods **match or underperform** strong classical
baselines. That is still a publishable result, a well-executed negative
benchmark in a hyped area has real value, but you should decide up front that
you are willing to publish it, because the alternative is p-hacking your way
into the low-credibility genre described above.

**Update: that caution has now been confirmed empirically** (§4.4, n = 30,
nested CV). The best classical pipeline beats the best quantum kernel by
+0.032 accuracy (p = 0.002, dz = 0.58, better in 24/30 subjects), no quantum
kernel beat the reference at any significance level, and quantum kernels are
2–270× more expensive.

**Second update (2026-09-03), and it changes the paper.** §4.4 was measuring
something it did not intend to. Every density-matrix kernel there was evaluated
in the **sensor frame**, while every strong classical baseline in the same
suite is invariant under congruence `C → ACAᵀ` — the group that EEG's nuisance
transformations actually generate. Referring the states to a training-set
reference state makes the quantum kernels *exactly* affine-invariant (§4.6,
proved, verified to 1e-15), relieves concentration 4.7–9.5×, and lifts
within-subject accuracy enough to **reverse the sign** of the headline
comparison (+0.052 classical-favouring → −0.018 quantum-favouring, p = 0.059).

That reversal is not a quantum win, and the new `control/riemann-kernel-SVM` is
what proves it: the classical Riemannian kernel in the same SVM, same frame,
same budget, scores second of 23, and **no quantum kernel differs from it**
(Δ −0.015 to +0.008, all p > 0.18). The gain belongs to the frame and the
kernel formulation, not to quantum structure. Cross-subject transfer agrees
with power to spare — all nine geometries tie, spread 0.014 (§4.7).

So the paper remains an **Option A/B negative result (§6), but for a better
reason and with a much sharper mechanism**: not "quantum kernels concentrate"
but "quantum information geometry has the wrong invariance group for EEG;
supply the missing invariance and the quantum kernels become indistinguishable
from their classical twins." That also diagnoses, concretely, how the
optimistic literature manufactures wins — compare a whitened quantum kernel
against an unwhitened classical baseline and a spurious advantage appears.

---

## 1. Why EEG is an unusually good substrate for this question

Most QML application papers pick a dataset arbitrarily. EEG has structural
properties that make the quantum question *natural* rather than decorative:

| Property | Why it matters for QML |
|---|---|
| Modern SOTA = covariance matrices on a Riemannian manifold | Trace-normalised covariance **is** a density matrix. Quantum geometry applies directly. |
| Low channel counts (8–64) | 8 channels = 3 qubits, 64 channels = 6 qubits under amplitude encoding. Fits NISQ scale honestly. |
| Small trial counts (~45–288 per subject) | Kernel methods are the right tool; deep nets overfit. Quantum kernels are at least in the correct algorithmic family. |
| High inter-subject variability | Creates a real, unsolved problem (calibration/transfer) that a better *geometry* could plausibly address. |
| Non-stationary, low SNR | Robustness and regularisation matter more than raw capacity, an argument for structured kernels. |

That fourth row is the strategically important one. **The unsolved problem in
BCI is not within-subject accuracy, it is cross-subject/cross-session
transfer.** Chasing within-subject accuracy puts you in a crowded field where
classical methods are already excellent. Chasing transfer puts you where the
actual pain is. See §6.

---

## 2. Literature landscape

### 2.1 The optimistic strand (claims of quantum benefit on EEG)

- **[EEG-based motor imagery classification with quantum algorithms](https://www.sciencedirect.com/science/article/abs/pii/S0957417424002197)**
  Olvera, Ross & Rubio, *Expert Systems with Applications* **247**:123354
  (2024). Two NISQ-executable approaches (a quantum genetic algorithm for
  feature selection, and an end-to-end variant). Reports 83.82% / 85.56% /
  73.73% on **BCI Competition IV-2b** for subject-dependent cross-validation,
  subject-dependent hold-out, and leave-one-subject-out respectively. *(An
  earlier draft of this review recorded "84.86% on IV-2a" from a search
  summary; that was wrong: corrected against the source record.)*
- **[QEEGNet: Quantum Machine Learning for Enhanced EEG Encoding](https://arxiv.org/abs/2407.19214)**
  (2024). Inserts a variational quantum layer into EEGNet. Claims it
  "consistently outperforms traditional EEGNet on most subjects" and is more
  noise-robust. Notably, **the authors themselves hedge**: results "might not
  always surpass traditional methods but it shows its potential."
- **[Advancing motor imagery EEG classification by quantum feature integration and QSVM](https://www.sciencedirect.com/science/article/abs/pii/S0003682X25004487)**
  (Applied Acoustics, 2025).
- **[Performance Analysis of Quantum-Enhanced Kernel Classifiers Based on Feature Maps: A Case Study on EEG-BCI Data](https://link.springer.com/chapter/10.1007/978-981-96-6579-2_25)**
  (Springer, 2025).
- **[A hybrid spiking neural network–quantum framework for spatio-temporal data classification: a case study on EEG](https://link.springer.com/article/10.1140/epjqt/s40507-025-00443-1)**
  (EPJ Quantum Technology, 2025).
- **[Addressing the Current Challenges of QML through Multi-Chip Ensembles](https://arxiv.org/html/2505.08782v1)**
  (2025). A multi-chip QCNN reportedly beats both a single-chip QCNN **and a
  matched classical CNN** on a PhysioNet EEG task, with reduced overfitting.
  This is one of the few that includes a matched classical control.
- **[QuantumNeuroXAI](https://www.nature.com/articles/s41598-026-47627-y)**
  (Scientific Reports, 2026) and
  **[Quantum-inspired wavelet and Fourier feature fusion for EEG epilepsy detection](https://www.nature.com/articles/s41598-025-31219-3)**
  (Sci Rep, 2025; 92.3% Bonn, 91.0% CHB-MIT), note these are **quantum-
  *inspired*** (classical algorithms borrowing quantum formalism), a
  distinction routinely blurred in abstracts.

**Recurring methodological weaknesses across this strand** (this is your
opening):
1. Classical baselines are weak or untuned, often plain SVM/LDA on band power,
   rather than the Riemannian tangent-space or FBCSP methods that actually
   define the state of the art.
2. No entanglement ablation, so "quantumness" is never isolated.
3. No dimension-matched control: the quantum model sees 4–8 PCA features and
   is compared against a classical model that saw something else entirely.
4. Small subject counts, single splits, no paired statistics across subjects.
5. Hyperparameters tuned for the quantum model only.

### 2.2 The skeptical strand (and why you must engage with it)

- **[Better than classical? The subtle art of benchmarking QML models](https://arxiv.org/abs/2403.07059)**
  Bowles, Ahmed & Schuld (2024). 12 QML models × 6 tasks × 160 datasets.
  Findings: **out-of-the-box classical models outperform the quantum
  classifiers**, and **removing entanglement often leaves performance equal or
  better**. This is the single most important paper to reckon with; a reviewer
  who knows the field *will* ask whether you ran their ablation.
- **[Exponential concentration in quantum kernel methods](https://www.nature.com/articles/s41467-024-49287-w)**
  Thanasilp, Wang, Cerezo & Holmes, *Nature Communications* 15 (2024).
  Quantum kernel values concentrate exponentially in qubit count toward a fixed
  value, yielding a trivial model. **We reproduced exactly this pathology on
  EEG data, see §4.1.** Engaging with it explicitly is a credibility marker.
- **[Quantum Kernel Methods under Scrutiny: A Benchmarking Study](https://arxiv.org/pdf/2409.04406)**
  (2024) and **[Benchmarking QML kernel training for classification](https://arxiv.org/abs/2408.10274)**
  (2024), quantum kernel *training* (QKT) often fails to justify its extra
  cost over plain quantum kernel estimation.
- **[Limitations of Amplitude Encoding on Quantum Classification](https://arxiv.org/pdf/2503.01545)** (2025).
- Dequantization results generally: claimed exponential speedups collapse to
  polynomial under sampling-access assumptions.

### 2.3 The bridge that already exists: and where it stops

- **[pyRiemann-qiskit](https://github.com/pyRiemann/pyRiemann-qiskit)**
  ([RIO Journal paper](https://riojournal.com/article/101006/), 2023), the
  most relevant prior art. Implements QSVC, VQC and a Nearest Convex Hull
  classifier on top of pyRiemann, with a standard pipeline
  (`QuantumClassifierWithDefaultRiemannianPipeline`) for binary brain-wave
  classification. **It is explicitly framed as a "sandbox," not a claim of
  advantage.** Crucially, its quantum step still operates on *tangent-space
  vectors*, the covariance is flattened to a Euclidean vector before the
  quantum model sees it. The density-matrix structure is discarded.
- **[A Unified SPD Token Transformer Framework for EEG Classification](https://arxiv.org/html/2601.21521)** (2026), notes that the
  **Bures–Wasserstein** distance "offers an alternative via matrix square root,
  with better gradient conditioning but **no prior systematic evaluation for
  EEG**." The Bures–Wasserstein distance on trace-normalised SPD matrices *is*
  the Bures metric of quantum information geometry, i.e. the geodesic distance
  induced by Uhlmann fidelity. **This is the gap.**
- pyRiemann ships `wasserstein` and `kullback` **distances**, but its kernel
  module (`pyriemann.geometry.kernel`) offers only `euclid`, `logeuclid`,
  `riemann`. **There is no fidelity/Bures kernel in the standard EEG toolbox.**

### 2.4 What strong classical baselines actually look like

You must clear these bars, not strawmen:

- **[Friedman–Nemenyi benchmark of MI-BCI decoders](https://arxiv.org/abs/2606.24394)** (2026) and
  **[Subject-Level Heterogeneity in EEG MI Decoding](https://arxiv.org/abs/2607.22778)** (2026):
  across PhysionetMI (109 subjects) and others, **covariance tangent-space
  projection and CSP consistently define the strongest methodological
  families**, in near-equilibrium on PhysionetMI. Average rankings *mask
  per-subject optimality*: different subjects have different best decoders,
  which is why paired, subject-wise statistics are mandatory.
- Standard MOABB protocol: `LeftRightImagery`, 8–15 Hz and 8–30 Hz bands,
  within-session evaluation.

---

## 3. The gap, stated precisely

> EEG spatial covariance matrices, normalised to unit trace, are quantum density
> matrices. Quantum information geometry therefore supplies a family of kernels
>Hilbert–Schmidt overlap, Uhlmann fidelity, Bures metric, that apply to EEG
> **without any dimensionality reduction or ad-hoc angle encoding**, and that
> map onto hardware-native primitives (the SWAP test). No published work
> evaluates these as kernels for EEG decoding against properly tuned Riemannian
> baselines with entanglement ablations and dimension-matched controls.

Why this framing is defensible:

- **Principled, not arbitrary.** The encoding is forced by the mathematics, not
  chosen to make a circuit fit.
- **Hardware-meaningful.** `tr(ρσ)` is exactly what a SWAP test estimates. The
  method has a real quantum implementation path (via purifications), even if
  we simulate it classically today.
- **Falsifiable.** It has clean classical controls: log-Euclidean and
  affine-invariant Riemannian kernels are the natural null hypotheses.
- **Honest about scope.** At 8–64 channels these kernels are classically
  computable in O(n³). This is *quantum-information-geometric* modelling, not a
  speedup claim, and saying so plainly is what separates a credible paper from
  the genre in §2.1.

---

## 4. What we built and what we found

Code: [src/qeeg/](src/qeeg/): `data.py`, `quantum.py`, `pipelines.py`,
`benchmark.py`. Dataset: PhysioNet EEG Motor Movement/Imagery (EEGMMIDB),
left-fist vs right-fist imagery (runs 4/8/12), 8 sensorimotor channels,
8–30 Hz, 0.5–3.5 s window, ~45 trials/subject.

Protocol: **nested CV**: outer 5-fold × 3 repeats for generalisation, inner
4-fold `GridSearchCV` for hyperparameters, applied *identically to every
pipeline including all classical baselines*. Paired Wilcoxon signed-rank across
subjects with Holm correction.

### 4.1 Finding 1: quantum overlap kernels concentrate catastrophically on EEG

Measured on subject S001, 45 trials, 8 channels (off-diagonal Gram entries after
cosine normalisation):

| Kernel | mean | std | min | max |
|---|---|---|---|---|
| Hilbert–Schmidt `tr(ρσ)` | 0.99082 | 0.01008 | 0.9216 | 0.9997 |
| Uhlmann fidelity `F(ρ,σ)` | 0.98807 | 0.00991 | 0.9400 | 0.9996 |
| `sqrt(F)` | 0.99400 | 0.00501 | 0.9695 | 0.9998 |

Every pair of trials looks ~99% identical. The Gram matrix is effectively
rank-one and the SVM has nothing to separate. **This is the Thanasilp et al.
concentration pathology, reproduced on real biological data**, and it is a
concrete, quantitative contribution in its own right, because that paper's
examples are largely synthetic.

For the IQP circuit kernel the failure is the mirror image, over-dispersion:

| Circuit kernel | off-diag mean | std |
|---|---|---|
| IQP, entanglement ON | 0.13433 | 0.15743 |
| IQP, entanglement OFF | 0.37967 | 0.25065 |

Entanglement pushes states toward mutual orthogonality, i.e. toward a kernel
that memorises the training set. Both regimes are useless for the same
underlying reason: no tunable length scale.

### 4.1b Finding 2: entanglement *accelerates* the collapse (new result)

We swept register size on real EEG by exploiting the identity directly: a
d-channel covariance is a density matrix on log2(d) qubits, so sweeping
channels over powers of two sweeps qubits with **no change of method**, 
4/8/16/32/64 channels = 2/3/4/5/6 qubits. Statistic: variance of off-diagonal
Gram entries (the quantity whose collapse defines concentration). Fitted decay
per added qubit, **n = 10 subjects** (`results/concentration_decay.csv`):

| Kernel | variance factor per qubit | variance 2→6 qubits |
|---|---|---|
| **IQP, entanglement ON** | **0.428** | 0.0782 → 0.0027 (29× loss) |
| IQP, entanglement OFF (product) | 0.716 | 0.0982 → 0.0263 (3.7× loss) |
| Bures-RBF | 0.753 | 0.0951 → 0.0317 |
| HS-RBF | 0.835 | 0.0932 → 0.0463 |
| Fidelity (raw) | 1.447 | 0.000264 → 0.001248 |
| HS-overlap (raw) | 1.520 | 0.000506 → 0.003556 |

Two results here, and the second is the more interesting one.

**(a) Entanglement is what kills the circuit kernel.** The entangled IQP kernel
loses variance at 0.428 per qubit, it more than *halves* with every qubit
added, while the identical circuit with entanglers deleted decays at 0.716.
In log-variance slope the entangled kernel concentrates **2.5× faster**
(−0.850 vs −0.334 per qubit); over the 2→6 qubit range that is a 29× loss of
kernel variance versus 3.7×. This supplies a *mechanism* for the Bowles et al.
observation that deleting entanglement often improves QML models: entanglement
pushes embedded states toward mutual orthogonality, driving the Gram matrix
toward the identity, which is a memorising, non-generalising model. That is a
testable, quantitative claim on real data and, to our knowledge, is not in the
literature.

Practical consequence for hardware: kernel variance of 0.0027 means resolving
typical kernel differences needs O(1/variance) ≈ 3.7×10² shots *per entry* at
one standard deviation, and the requirement grows exponentially with qubits.
The `shots_for_1sigma` column in `results/concentration_raw.csv` tracks this.

**(b) The two kernel families concentrate for *different reasons*.** The raw
HS/fidelity kernels do **not** follow the qubit-count law, their variance
*rises* with dimension (factor 1.53–1.99). They are already saturated at 2
qubits (variance ~1e-4, mean 0.99) because **EEG covariance matrices are
intrinsically similar to one another**, not because of register size. So there
are two distinct pathologies wearing the same name:

- *Circuit kernels*: concentration driven by qubit count and entanglement
  (the Thanasilp et al. mechanism).
- *Density-matrix overlap kernels*: concentration driven by the **data
  distribution**, dimension-independent, and in fact *relieved* by adding
  channels.

The practical corollary is encouraging for the density-matrix route: unlike
circuit kernels, it should be run at **more** channels, not fewer. That is the
opposite of what the standard "PCA down to 4 features" recipe does.

### 4.2 The remedy we implemented

Keep the quantum *geometry*, restore a length scale, exponentiate the induced
quantum distance with a tunable bandwidth, exactly as an RBF kernel does for
Euclidean distance:

- `K_HS(ρ,σ) = exp(−γ · tr[(ρ−σ)²])`
- `K_Bures(ρ,σ) = exp(−γ · d²_Bures)`, where `d²_Bures = 2(1 − √F(ρ,σ))`

with γ set by the median heuristic on the **training split only** (no leakage)
and a multiplier tuned in the inner CV. Circuit kernels get the analogous
treatment via a tunable angle `scale`.

This is the methodological core of the paper: *quantum kernels for EEG are not
usable off the shelf; they require explicit bandwidth control, and reporting
them without it produces exactly the flat, uninformative results seen in the
literature.*

### 4.3 The benchmark suite (15 pipelines, 3 groups)

**Classical baselines**: the bar to clear:
`logvar+LDA`, `CSP+LDA`, `MDM`, `TS+LR`, `TS+RBF-SVM`

**Quantum:**
`HS-overlap-SVM`, `Fidelity-SVM` (raw, to document concentration),
`HS-RBF-SVM`, `Bures-RBF-SVM` (bandwidth-corrected),
`IQP-kernel-SVM`, `CNOT-kernel-SVM`

**Controls**: what makes this a real study:
- `IQP-no-entangle`: identical circuit, entanglers deleted (the Bowles et al.
  ablation).
- `PCA-matched-RBF`, `PCA-matched-linear`, classical kernels on the *same*
  4-dimensional PCA features the circuit kernels see. Without this, any circuit
  result is confounded with dimensionality reduction.
- `logeuclid-TS+LR`: the classical geometry twin of the density-matrix kernels.

### 4.4 Results: n = 30 subjects, nested CV

**Completed run.** 30 subjects × 15 pipelines × 15 outer folds = 6 750 fold
scores. Sources: `results/summary_motor8_q4.csv`,
`results/tests_vs_classical-TS+LR_motor8_q4.csv`, `results/meta_motor8_q4.json`.
Figures: `results/figures/fig2_benchmark.*`, `fig3_paired_differences.*`.

| Pipeline | Group | Acc | SD | AUC | s/subj |
|---|---|---|---|---|---|
| classical/CSP+LDA | classical | **0.6114** | 0.154 | 0.636 | 2.0 |
| classical/TS+LR | classical | 0.5946 | 0.174 | 0.625 | 2.7 |
| classical/TS+RBF-SVM | classical | 0.5899 | 0.168 | 0.605 | 6.3 |
| control/logeuclid-TS+LR | control | 0.5825 | 0.149 | 0.611 | 2.6 |
| control/PCA-matched-linear | control | 0.5820 | 0.156 | 0.588 | 2.4 |
| quantum/CNOT-kernel-SVM | quantum | 0.5790 | 0.148 | 0.593 | 9.3 |
| control/PCA-matched-RBF | control | 0.5775 | 0.153 | 0.576 | 6.6 |
| control/IQP-no-entangle | control | 0.5763 | 0.147 | 0.594 | 8.3 |
| classical/MDM | classical | 0.5751 | 0.161 | 0.594 | 0.23 |
| quantum/IQP-kernel-SVM | quantum | 0.5657 | 0.134 | 0.588 | 10.2 |
| classical/logvar+LDA | classical | 0.5585 | 0.134 | 0.585 | **0.046** |
| quantum/Bures-RBF-SVM | quantum | 0.5521 | 0.109 | 0.567 | 12.4 |
| quantum/HS-RBF-SVM | quantum | 0.5494 | 0.101 | 0.567 | 5.8 |
| quantum/HS-overlap-SVM | quantum | 0.5321 | 0.076 | 0.578 | 2.1 |
| quantum/Fidelity-SVM | quantum | 0.5232 | 0.082 | 0.557 | 4.7 |

Quantum kernels occupy **five of the bottom six** positions. Absolute
accuracies are modest because PhysioNet MI contains many near-chance subjects;
this is expected and is why paired subject-wise testing is mandatory.

**Primary comparison: best classical vs best quantum:**

| Comparison | Δacc | p (Wilcoxon) | Cohen's dz | better in |
|---|---|---|---|---|
| **CSP+LDA vs CNOT-kernel-SVM** | **+0.0323** | **0.0023** | **+0.576** | **24/30** |
| CSP+LDA vs Fidelity-SVM | +0.0881 | 0.0006 | +0.665 | 24/30 |

**The best classical pipeline significantly outperforms the best quantum
kernel**, a medium effect size, consistent across 80 % of subjects.

**Family test vs the `classical/TS+LR` reference** (14 comparisons, Holm
corrected): **0 of 14 significant after correction.** Uncorrected, only two
reach p < 0.05, `HS-overlap-SVM` (p = 0.048) and `Fidelity-SVM` (p = 0.013), 
and in both the quantum model is *worse*. **No quantum kernel beat the
reference, at any significance level.**

**Ablations: the controls, which are the point of the study:**

| Ablation | Δacc | p | dz | better in |
|---|---|---|---|---|
| Remove entanglement (`IQP-no-entangle` vs `IQP-kernel-SVM`) | +0.0106 | 0.309 | +0.236 | 17/30 |
| Dimension-matched (`PCA-matched-linear` vs `IQP-kernel-SVM`) | +0.0163 | 0.173 | +0.275 | 18/30 |

Both point the same way as §2.2 but **neither is significant on accuracy at
n = 30**. State this precisely and resist overclaiming: deleting entanglement
did not *hurt*, and numerically helped, but the accuracy evidence alone cannot
carry the claim. The entanglement effect is established much more strongly on
**kernel variance** (§4.1b, 0.428 vs 0.716 per qubit) than on accuracy, which
is itself the interesting point: the mechanism is visible in the kernel long
before it shows up in a downstream score that is dominated by EEG noise.

**Cost.** `classical/logvar+LDA` reaches 0.5585 in **0.046 s/subject**;
`quantum/Bures-RBF-SVM` reaches 0.5521 in 12.4 s, **270× slower and less
accurate**. Every quantum kernel is 2–270× more expensive than a classical
baseline that matches or beats it. At infinite shots, on a simulator, with no
hardware noise: i.e. under conditions maximally favourable to the quantum side.

**Honest summary of what this run shows.** Against the specific `TS+LR`
reference, differences are within noise. Against the *best* classical pipeline,
quantum kernels lose significantly. Nothing here supports a quantum advantage
for within-subject MI decoding, and the density-matrix kernels, the
principled, hardware-mappable formulation, performed **worst of all**, which
is the most scientifically interesting negative result in the set, because
§4.1b explains exactly why (data-driven concentration, not qubit count).

---

### 4.5 Replication on BCI Competition IV-2a, and what changes with more data

The single-dataset benchmark in §4.4 has an obvious weakness, which the paper
had to concede as a limitation: PhysioNet EEGMMIDB gives only 45 trials per
subject, so "quantum kernels lose" might really be "quantum kernels lose when
starved of data". BCI Competition IV-2a is the natural test, for two reasons.
It has **288 trials per subject**, more than six times as many, on cleaner
recordings. And it is the dataset the quantum-EEG literature actually reports
on, so a result here is directly comparable to published claims.

Identical protocol, identical 8 sensorimotor channels (all present in the 2a
montage), identical nested cross-validation, identical tuning budget. Only the
data changed. 9 subjects, 2 592 trials, 56 minutes of compute.

**Result: quantum kernels do not merely lose again, they lose by much more.**

| Pipeline | Group | Acc (2a) | Acc (PhysioNet) |
|---|---|---|---|
| classical/TS+LR | classical | **0.7631** | 0.5946 |
| classical/TS+RBF-SVM | classical | 0.7554 | 0.5899 |
| classical/CSP+LDA | classical | 0.7507 | 0.6114 |
| control/logeuclid-TS+LR | control | 0.7506 | 0.5825 |
| classical/MDM | classical | 0.7226 | 0.5751 |
| classical/logvar+LDA | classical | 0.6972 | 0.5585 |
| control/PCA-matched-linear | control | 0.6900 | 0.5820 |
| control/PCA-matched-RBF | control | 0.6874 | 0.5775 |
| quantum/CNOT-kernel-SVM | quantum | 0.6845 | 0.5790 |
| control/IQP-no-entangle | control | 0.6821 | 0.5763 |
| quantum/IQP-kernel-SVM | quantum | 0.6721 | 0.5657 |
| quantum/Bures-RBF-SVM | quantum | 0.6002 | 0.5521 |
| quantum/HS-RBF-SVM | quantum | 0.5951 | 0.5494 |
| quantum/HS-overlap-SVM | quantum | 0.5862 | 0.5321 |
| quantum/Fidelity-SVM | quantum | 0.5799 | 0.5232 |

#### Finding 3: the gap widens with better data

| Dataset | Best classical | Best quantum | Gap | Gap to worst quantum |
|---|---|---|---|---|
| PhysioNet (45 trials) | 0.6114 | 0.5790 | 0.032 | 0.088 |
| BCI IV-2a (288 trials) | 0.7631 | 0.6845 | **0.079** | **0.183** |

Classical methods gained roughly 0.15 accuracy from the extra data. The
density-matrix kernels gained about 0.05 and stayed near chance in relative
terms. **More and cleaner data made the quantum kernels relatively worse, not
better.** This directly refutes the most natural defence of a negative QML
result, that the quantum model was starved. It is exactly what §4.1 predicts:
a kernel whose off-diagonal entries all sit at 0.99 has no extra structure to
exploit no matter how many trials you hand it, while the classical methods
have plenty.

#### Finding 4: the ranking is stable across radically different datasets

**Spearman rank correlation between the two datasets: 0.882** over 15
pipelines, despite one having 30 subjects with 45 trials each and the other 9
subjects with 288. More striking, the **four bottom positions are identical**:
Bures-RBF (12), HS-RBF (13), HS-overlap (14), Fidelity (15) on both datasets,
with zero rank shift. The ordering is a property of the methods, not of the
dataset.

#### Statistics, and a floor that must be reported honestly

Pre-specified comparisons, with Fisher's method combining the two independent
datasets:

| Comparison | PhysioNet | BCI IV-2a | Fisher combined |
|---|---|---|---|
| Best classical vs best quantum | +0.032, p=0.0023, 24/30 | +0.066, p=0.0039, **9/9** | **p = 0.0001** |
| Best classical vs density-matrix | +0.088, p=0.0006, 24/30 | +0.171, p=0.0039, **9/9** | **p = 0.00002** |
| Entanglement ablation | +0.011, p=0.309, 17/30 | +0.010, p=0.203, 7/9 | p = 0.237 |
| Dimension-matched control | +0.016, p=0.173, 18/30 | +0.018, p=0.129, 7/9 | p = 0.107 |

On 2a the best classical pipeline beat the best quantum kernel in **all nine
subjects**, with dz = 1.14; against the density-matrix kernels, dz = 1.45.

**The floor caveat.** With n pairs the two-sided Wilcoxon signed-rank test
cannot return a p below 2^(1-n). At n = 9 that floor is **0.0039**, so Holm
correction across 14 comparisons cannot produce anything below **0.0547**,
regardless of effect size. Ten of the fourteen 2a comparisons sit exactly at
that floor. Reporting "0 of 14 survive Holm correction" without this caveat
would badly misrepresent the data: the test is saturated, not null. This is
why the pre-specified comparisons above, which need no family correction, are
the right inference. `crossdataset.py` computes and prints the floor so it
cannot be forgotten.

#### What is still not established

The entanglement ablation now has four independent signals pointing the same
way: kernel variance decay (§4.1b, a large and unambiguous effect), accuracy on
PhysioNet, accuracy on 2a, and the Fisher combination. It is still **not
significant on accuracy** (combined p = 0.237). The honest statement remains
that entanglement clearly damages the *kernel*, and that its accuracy cost is
directionally consistent but below the resolution of these sample sizes. Do
not overstate it.

---

### 4.6 The frame problem: the benchmark was not comparing like with like

**Status: complete, n = 30, PhysioNet.** `--suite extended --tag
refstate_motor8_q4`, 23 pipelines, identical nested-CV protocol.
Code: `src/qeeg/reference.py`, `src/qeeg/quantum.py`, `src/qeeg/pipelines.py`.

**Regression check.** The 15 core pipelines in that run reproduce
`results/summary_motor8_q4.csv` to **max |Δ| = 0.0000**, so everything below is
purely additive and the published numbers stand unchanged. (This also clears
the PennyLane 0.38 → 0.45 upgrade: the circuit kernels are bit-identical.)

#### The asymmetry

Every density-matrix kernel in §4.4 was evaluated in the **sensor frame**:
`ρ = C/tr(C)`, with `C` as the electrodes deliver it. The classical baselines
were not. `pyriemann.tangentspace.TangentSpace.fit` estimates a reference mean
`M` and `transform` maps `C ↦ log(M^{-1/2} C M^{-1/2})` — verified in the
library source, not inferred. MDM compares affine-invariant distances; CSP
solves a generalised eigenproblem. **All three strong classical pipelines are
congruence-invariant and all four quantum kernels are not.**

This matters because the nuisance group of EEG *is* congruence,
`C → A C Aᵀ`: electrode gain, impedance, choice of reference electrode,
volume conduction, and the change from one subject or session to the next all
act this way. The affine-invariant Riemannian metric is invariant under that
group, which is why tangent-space decoding is the classical state of the art
and why it transfers. `tr(ρσ)`, Uhlmann fidelity, the Bures distance and the
quantum relative entropy are invariant only under the **orthogonal subgroup**.

So part of what §4.4 measured was an invariance mismatch, not a property of
quantum geometry. That is a hole a referee can drive a truck through, and it
has to be closed whichever way the numbers then fall.

#### The fix, and why it is still quantum

Measure each state relative to a **reference state** `M`, the Fréchet mean of
the *training* covariances: `ρ̃ = W C W / tr(W C W)` with `W = M^{-1/2}`.

> **Proposition.** Under `C_i → A C_i Aᵀ` the affine-invariant mean is
> equivariant, `M → A M Aᵀ`. Put `B = (A M Aᵀ)^{-1/2} A`. Then `B M Bᵀ = I`
> and `W M Wᵀ = I`, so `U := B M^{1/2}` is orthogonal and `B = U W`. Every
> whitened matrix therefore transforms as `W C_i W → U (W C_i W) Uᵀ`, by one
> common orthogonal `U`. All four quantum quantities are invariant under a
> common unitary, hence **in the reference frame they are exactly
> affine-invariant.**

Verified numerically (`python -m qeeg.reference --check`), nuisance with
cond(A) = 25.8:

| Kernel | max change, sensor frame | max change, reference frame |
|---|---|---|
| HS overlap | 2.52e-01 | 1.53e-15 |
| Fidelity | 1.65e-01 | 4.44e-15 |
| Bures d² | 1.77e-01 | 4.89e-15 |
| QRE | 3.53e-01 | 1.02e-14 |

Put beside the concentration numbers, the sensor-frame column is damning: the
entire off-diagonal spread of the sensor-frame HS kernel is std ≈ 0.022, and a
routine nuisance congruence moves kernel entries by 0.25. **The nuisance is
about an order of magnitude larger than the whole discriminative signal.**

The construction is not a classical pre-processing hack smuggled in.
`ρ ↦ WρW/tr(WρW)` is a filtering (Lüders) operation with Kraus operator `W`
followed by renormalisation — a legitimate quantum operation, so the
SWAP-test implementation path is untouched. It is state preparation.

#### Consequence 1: most of the concentration was the frame (n = 14)

`python -m qeeg.reference --gram --subjects 14`:

| Kernel | mean (sensor) | mean (reference) | var (sensor) | var (reference) | variance gain |
|---|---|---|---|---|---|
| HS overlap | 0.98087 | 0.87133 | 0.00049 | 0.00471 | **9.5×** |
| Fidelity | 0.98248 | 0.93079 | 0.00033 | 0.00155 | 4.7× |
| Bures d² | 0.01771 | 0.07108 | 0.00035 | 0.00178 | 5.2× |
| QRE | 0.03912 | 0.14671 | 0.00158 | 0.00827 | 5.2× |

This revises §4.1b(b). The data-driven concentration is real, but it is not
mostly intrinsic to EEG covariances: it is largely an artefact of measuring
them in the sensor frame. Since the shot budget to resolve a kernel entry
scales as 1/variance, the reference frame cuts the hardware cost of these
kernels by the same 4.7–9.5×.

#### Consequence 2: the frame is worth more than the geometry (n = 30)

Every density-matrix kernel improves significantly when referred to a
reference state:

| Kernel | sensor frame | reference frame | Δ | p | dz | better in |
|---|---|---|---|---|---|---|
| Fidelity | 0.5232 | 0.6296 | **+0.1064** | **0.0003** | +0.77 | 23/30 |
| HS overlap | 0.5321 | 0.6210 | +0.0889 | 0.0022 | +0.64 | 21/30 |
| Bures-RBF | 0.5521 | 0.6106 | +0.0585 | 0.0089 | +0.53 | 20/30 |
| HS-RBF | 0.5494 | 0.6064 | +0.0570 | 0.0491 | +0.43 | 19/30 |
| QRE-RBF | 0.5590 | 0.6094 | +0.0504 | 0.0267 | +0.45 | 20/30 |

And the headline comparison **reverses sign**:

| Frame | best classical − best quantum | p | classical better in |
|---|---|---|---|
| Sensor (as published) | **+0.0523** | **0.0159** | 20/30 |
| Reference | **−0.0183** | 0.0587 | 10/30 |

In the sensor frame CSP+LDA beats the best quantum kernel significantly. In
the reference frame `Fidelity-ref` (0.6296) is *ahead* of CSP+LDA (0.6114),
p = 0.059. Read carelessly, that is a quantum win.

#### It is not a quantum win. The new control says so.

`control/riemann-kernel-SVM` — the affine-invariant Riemannian kernel in an
SVM, i.e. the density-matrix kernels' **exact classical twin**: same input,
same classifier, same tuning budget, same reference frame, only the geometry
differs — scores **0.6215, second of 23**. Paired against it:

| Quantum kernel (reference frame) | Δ vs Riemannian twin | p |
|---|---|---|
| Fidelity-ref | +0.0081 | 0.230 |
| HS-overlap-ref | −0.0005 | 0.873 |
| Bures-RBF-ref | −0.0109 | 0.333 |
| QRE-RBF-ref | −0.0121 | 0.284 |
| HS-RBF-ref | −0.0151 | 0.183 |

**Not one differs from its classical twin.** Meanwhile the twin itself beats
`TS+LR` by +0.0269 (p = 0.0098). So the entire gain is attributable to two
things that have nothing to do with quantum structure: measuring in the
reference frame, and using an SPD kernel in an SVM rather than a tangent-space
projection. The quantum geometry adds nothing on top.

This is exactly what the control was built for. Without it, this run would
have supported "quantum kernels beat tuned classical baselines" at p = 0.059 —
and that claim would have been false.

#### Why this matters beyond our own result

It supplies a concrete mechanism for how the optimistic quantum-EEG literature
generates wins. Compare a quantum kernel that is (perhaps inadvertently)
working in a whitened/aligned frame against a classical baseline that is not,
and a win appears that is entirely attributable to the frame. The remedy is
the dimension-matched control's geometric analogue: **the same geometry
formulation, in the same frame, differing only in the metric.**

#### What this does to the paper

- It removes the most obvious referee objection ("you compared an
  affine-invariant baseline against a non-invariant kernel").
- It upgrades §4.1's diagnosis into a *mechanism*: these kernels have the
  wrong invariance group for EEG, and concentration is a symptom.
- The headline survives, better supported than before: quantum kernels do not
  beat classical ones. But the *reason* changes from "they concentrate" to
  "once you remove the frame confound they are indistinguishable from their
  classical twins", which is a sharper and more defensible claim.
- It corrects the Option C hypothesis in §6, see the note there.

---

### 4.7 Cross-subject transfer: the Option C question, answered

**Status: complete, n = 30, PhysioNet.** `src/qeeg/transfer.py`, results in
`results/transfer_{folds,summary,frame_tests}_motor8.csv`. Leave-one-subject-out,
train on the pooled trials of the other 29. Hyperparameters tuned by
subject-grouped inner CV on training subjects only, identical budget for every
method. `recenter=True` whitens each subject by its own Fréchet mean, which
uses no labels and for the held-out subject is exactly the unlabelled data a
real deployment would have.

#### Result 1: the frame effect is large, and larger for the quantum kernels

| Pipeline | sensor | reference | Δ | p | better in |
|---|---|---|---|---|---|
| quantum/HS-overlap | 0.5519 | 0.6467 | **+0.0948** | 0.00002 | 24/30 |
| classical/MDM | 0.5526 | 0.6326 | +0.0800 | 0.0009 | 19/30 |
| quantum/Bures-RBF | 0.5615 | 0.6400 | +0.0785 | 0.0002 | 22/30 |
| quantum/Fidelity | 0.5593 | 0.6356 | +0.0763 | 0.0009 | 21/30 |
| quantum/HS-RBF | 0.5637 | 0.6348 | +0.0711 | 0.0020 | 20/30 |
| quantum/QRE-RBF | 0.5756 | 0.6415 | +0.0659 | 0.0022 | 21/30 |
| control/logeuclid-kernel-SVM | 0.5763 | 0.6333 | +0.0570 | 0.0012 | 20/30 |
| control/riemann-kernel-SVM | 0.5874 | 0.6348 | +0.0474 | 0.0037 | 19/30 |
| classical/TS+LR | 0.6044 | 0.6444 | +0.0400 | 0.0092 | 18/30 |

Every method improves and every improvement is significant. But the ordering is
the point: **mean gain +0.077 for the five quantum kernels versus +0.056 for the
four classical/control ones**, with the largest gain going to the raw
Hilbert-Schmidt overlap (the most concentration-crippled kernel) and the
smallest to `TS+LR` (which already whitened by the reference mean internally).

One caveat that must be stated so the mechanism is not oversold. Per-subject
recentring is *not* a no-op even for an affine-invariant method, because each
subject receives a **different** whitening; affine invariance covers a common
congruence, not a per-domain one. So the classical gains are the known
Riemannian-recentring effect from the transfer literature, not a contradiction.
The invariance account predicts specifically that the non-invariant methods
gain *more* and that the sensor-frame quantum kernels sit *below* the
sensor-frame classical ones. Both hold.

#### Result 2: in the reference frame, every geometry is the same

Paired against `classical/TS+LR` in the reference frame, over 30 held-out
subjects:

| Pipeline | acc | Δ vs TS+LR | p |
|---|---|---|---|
| quantum/HS-overlap | 0.6467 | −0.0022 | 0.744 |
| quantum/QRE-RBF | 0.6415 | +0.0030 | 0.749 |
| quantum/Bures-RBF | 0.6400 | +0.0044 | 0.544 |
| quantum/Fidelity | 0.6356 | +0.0089 | 0.251 |
| control/riemann-kernel-SVM | 0.6348 | +0.0096 | 0.144 |
| quantum/HS-RBF | 0.6348 | +0.0096 | 0.367 |
| control/logeuclid-kernel-SVM | 0.6333 | +0.0111 | 0.133 |
| classical/MDM | 0.6326 | +0.0119 | 0.314 |

**Nothing is significant, uncorrected or Holm-corrected. The total spread
across all nine geometries is 0.0141.** And this is a genuine null rather than
a saturated test: at n = 30 the two-sided Wilcoxon floor is 1.9 × 10⁻⁹, so the
test has ample resolution. `HS-overlap` nominally tops the table by 0.0022 over
`TS+LR`, which is noise.

#### What this settles

The Option C hypothesis is **answered, and answered negatively, exactly at the
ceiling the algebra predicted**. §4.6 established that referred to a reference
state the quantum kernels are precisely as affine-invariant as AIRM and can
therefore at best match it. They match it. They do not beat it.

That is a much stronger result than a bare negative, because the theory
predicted the ceiling *before* the experiment and the experiment landed on it.
Three predictions have now been confirmed:

1. concentration relieves in the reference frame (4.7–9.5×, §4.6);
2. within-subject accuracy rises to parity, not beyond (§4.6, preliminary);
3. transfer accuracy rises to parity, not beyond (this section, n = 30).

The remaining open direction is what the invariance argument does *not* cover:
shift that is not a congruence. Nothing here rules out a quantum divergence
being better on non-congruence shift, and the spectral-weighting argument in §6
is still the reason to look. But on inter-subject transfer in motor imagery,
the answer is parity.

---

### 4.8 Shot noise: what the kernels would cost on hardware

**Status: complete, n = 30.** `src/qeeg/shots.py`,
`results/shots_{folds,summary}_motor8.csv`. `tr(ρσ)` is exactly the SWAP-test
observable, so the ancilla gives `P(0) = (1+k)/2` and `S` shots yield an
unbiased estimate with variance `(1−k²)/S`. We sample the upper triangle
binomially (a device estimates each unordered pair once), mirror, and project
back to the PSD cone.

| Kernel | frame | 10² | 10³ | 10⁴ | 10⁵ | 10⁶ | ∞ |
|---|---|---|---|---|---|---|---|
| HS-overlap | sensor | 0.5096 | 0.5069 | 0.5321 | 0.5383 | 0.5272 | 0.5323 |
| HS-overlap | reference | 0.4914 | 0.5175 | 0.5652 | 0.5894 | 0.6072 | **0.6180** |
| HS-RBF | sensor | 0.4919 | 0.5037 | 0.5111 | 0.5380 | 0.5331 | 0.5543 |
| HS-RBF | reference | 0.4817 | 0.5294 | 0.5644 | 0.5862 | 0.5901 | **0.6089** |

**A prediction of §4.6 was wrong and is corrected here.** We expected the
4.7–9.5× variance gain to translate into a 5–9× shot saving. It does not. The
reference frame needs *more* shots to approach its own ceiling, not fewer,
because it has a higher ceiling and therefore finer distinctions to resolve.
The "shots to reach 99 % of own ceiling" statistic is actively misleading here:
the sensor frame converges fast only because it converges to near-chance.

The meaningful comparison is cross-frame and absolute:

> **At 10⁴ shots per Gram entry the reference frame already beats the sensor
> frame at infinite shots** — 0.5652 vs 0.5323 for HS-overlap, 0.5644 vs
> 0.5543 for HS-RBF — and the margin widens from there.

Below 10³ shots both frames collapse to chance, so the frame does not rescue a
starved estimator; it raises the ceiling that a well-fed one can reach.

**The honest cost statement for the paper.** Near-ceiling accuracy needs
10⁵–10⁶ shots *per Gram entry*. A 45-trial subject has ~10³ unordered pairs, so
one Gram matrix costs 10⁸–10⁹ shots, and IV-2a's 288 trials cost ~40× more.
This is the sentence that keeps the scope claim honest: the reference frame
makes these kernels *work*, it does not make them *cheap*.

Caveat to carry: some fits hit the `max_iter` cap (2×10⁶) at the noisiest
budgets. The cap is identical across frames, shot levels and hyperparameters,
so it cannot bias the comparison, but it bounds a tail rather than resolving
it. Note also that the reference state here is fitted on all of a subject's
trials — it is label-free, and this study asks about estimation cost rather
than generalisation, so shot levels are comparable to each other but the
absolute values are not comparable to the nested-CV benchmark.

---

## 5. Datasets worth using

| Dataset | Access | Size | Why |
|---|---|---|---|
| **PhysioNet EEGMMIDB** | `mne.datasets.eegbci`, works, no auth | 109 subjects, 64 ch, 160 Hz | Large subject count for paired stats. **Currently used.** |
| **BCI Competition IV-2a** | MOABB `BNCI2014_001` | 9 subjects, 22 ch, 4 classes | The genre's default benchmark. **Done (§4.5).** |
| **BCI Competition IV-2b** | MOABB `BNCI2014_004` | 9 subjects, 3 ch | Multi-session, good for transfer experiments. |
| **Cho2017** | MOABB | 52 subjects | Large MI dataset, good for cross-subject. |
| **CHB-MIT** | PhysioNet | 23 patients | Seizure detection; heavy class imbalance. |
| **TUH EEG** | Registration required | Thousands of clinical recordings | Largest clinical corpus; the credible route to "scale." |

Recommendation: **PhysioNet MI (breadth) + BCI IV-2a (comparability)**. Add
Cho2017 if the cross-subject story in §6 becomes the centre of the paper.

---

## 6. What the publishable paper actually is

Do **not** write "Quantum SVM achieves 85% on EEG." Write one of these:

**Option A: the rigorous negative/neutral benchmark (safest, still valuable).**
*"Do quantum kernels help EEG decoding? A controlled benchmark with
entanglement ablations."* Contribution = methodology + honest answer + open
code. Precedent: Bowles et al. is highly cited precisely for this. Venue:
*Journal of Neural Engineering*, *EPJ Quantum Technology*, *Quantum Machine
Intelligence*.

**Option B: the principled-representation paper (higher ceiling).**
*"EEG covariance matrices as quantum states: fidelity and Bures kernels for
brain–computer interfaces."* Contribution = the density-matrix formulation, the
concentration diagnosis on real data, the bandwidth remedy, and a fair
benchmark. Works **even if the quantum kernels only match** classical ones,
because the framework and the diagnosis are the contribution. Venue: *Journal
of Neural Engineering*, *IEEE TNSRE*, *Quantum Machine Intelligence*.

**Option C: the one with the highest scientific upside (hardest).**
Target **cross-subject transfer**, not within-subject accuracy. Still the right
target: it addresses the field's actual bottleneck (the calibration problem).

**But the hypothesis as originally stated is wrong, and §4.6 is why.** The
original wording was: *quantum-information distances (Bures, quantum relative
entropy) are more robust to the covariance shift between subjects than
affine-invariant Riemannian distances.* At the level of group invariance that
cannot hold. Inter-subject shift is dominated by congruence `C → A C Aᵀ`. The
affine-invariant Riemannian distance is invariant under the whole congruence
group; Bures and QRE are invariant only under its orthogonal subgroup in the
sensor frame, and, once referred to a reference state, **exactly** as invariant
as AIRM — never more. So the quantum distances can at best match, and in the
sensor frame they must be strictly worse. Running the experiment without
noticing this would have produced a result that looked empirical but was
algebraically forced.

The restated hypothesis, which is still open and still worth testing:
congruence is not all of the shift. Subjects differ in conditioning, effective
rank, and how spiky their spatial spectrum is, and those survive whitening.
Bures and QRE weight the eigenvalue spectrum differently from AIRM — QRE
penalises support mismatch, Bures compresses large ratios — so **within the
reference frame**, where all three share an invariance group, they may still
order transfer differently. That is a fair test of geometry against geometry
and it is the version to run.

**Non-negotiables for any of them:**
- Tuned classical baselines (Riemannian TS, FBCSP), same CV budget.
- Entanglement ablation.
- Dimension-matched controls.
- Paired subject-wise statistics + multiple-comparison correction.
- Report wall-clock cost. Quantum kernels here are 5–50× slower for no gain so
  far; hiding that would be dishonest.
- Say plainly that simulated quantum kernels at this scale offer **no speedup**.
- Pre-register the analysis, or at minimum fix the protocol before looking at
  test results.

---

## 7. Honest risk assessment

| Risk | Severity | Mitigation |
|---|---|---|
| Quantum methods simply lose | **CONFIRMED (§4.4)** | Framed as Option A/B. The negative result now carries two mechanistic findings (§4.1, §4.1b) that explain it, which is what lifts it above a bare leaderboard table. |
| Reviewer: "no quantum advantage, so what?" | High | Pre-empt: the paper is a benchmark + representation contribution, not an advantage claim. Cite Bowles et al. as precedent. |
| Reviewer: "this is classically simulable" | Certain | Concede immediately and precisely. 8 ch = 3 qubits; everything here is classically computable. The claim is *geometric*, not computational. |
| Kernel concentration kills everything | **CONFIRMED (§4.1)** | Documented, quantified, and remedied (§4.1–4.2). Now the paper's strongest contribution rather than a threat. |
| Small trials/subject (~45) | Medium | Many subjects + repeated nested CV + paired stats; add BCI IV-2a (288 trials/subject). |
| Field is crowded with low-quality papers | Medium | Rigour *is* the differentiator. |

**Bottom line:** the fascination is legitimate and it has a real mathematical
core. But the value you can add is **methodological rigour on a principled
representation**, not a bigger accuracy number. Aim at Option C, be ready to
land on Option B, and be willing to publish A.

---

## 8. Reproducing

```bash
pip install -r requirements.txt          # scipy pinned <1.16 (see note in file)
PYTHONPATH=src python -m qeeg.benchmark --subjects 30 --splits 5 --repeats 3
```

Outputs land in `results/`: per-fold raw scores, per-pipeline summary, paired
tests vs the chosen reference, and a metadata JSON recording the exact protocol.

---

## 9. Open threads

### Where we are (2026-09-03)

The study now has a **mechanism, a fix, and the control that stops the fix
being oversold**, which is a materially stronger position than the negative
benchmark of §4.4.

The one-paragraph version: every density-matrix kernel was being evaluated in
the *sensor* frame while every strong classical baseline was congruence
invariant, so the benchmark was partly measuring an invariance mismatch. Refer
the states to a training-set reference state and the quantum kernels become
exactly affine invariant (§4.6, proved and verified to 1e-15), concentration
relieves 4.7-9.5x, and within-subject accuracy jumps enough to *reverse* the
headline comparison. But the new Riemannian-kernel control -- same input, same
SVM, same budget, same frame, only the metric differs -- matches every quantum
kernel to within noise. So the gain is the frame plus the kernel formulation,
not quantum structure. Cross-subject transfer says the same thing with power
to spare: all nine geometries tie (§4.7).

**Settled, n = 30, do not re-litigate**

- [x] Within-subject benchmark, 15 pipelines, nested CV (§4.4).
- [x] Kernel concentration on real EEG, 2->6 qubits (§4.1, §4.1b).
- [x] Entanglement ablation and dimension-matched controls (§4.3).
- [x] BCI IV-2a replication, core suite (§4.5).
- [x] Reference-state formulation + invariance proposition (§4.6).
- [x] Extended suite at n = 30: frame effect significant for all five kernels;
      headline reverses; **Riemannian-kernel control shows no quantum kernel
      beats its classical twin** (§4.6). Core-15 rows reproduce the published
      run to max |delta| = 0.0000, so all of this is additive.
- [x] Cross-subject transfer, LOSO n = 30: parity across all nine geometries,
      spread 0.0141, genuine null (§4.7). Option C answered.
- [x] Shot-noise study, n = 30 (§4.8). Corrects a wrong prediction of §4.6.
- [x] Manuscript compiles (17 pp, 0 warnings); all refs DOI-verified.

**In flight**

- [ ] **Filter-bank / FBCSP suite at 5 qubits**, `--suite filterbank`, batches
      `fbdone01-05` + `fbrest01-05`, merge with `qeeg.merge`. This is the
      load-bearing one: with quantum kernels now at the top of the table, the
      first referee question is whether parity survives against FBCSP.
- [ ] **IV-2a extended replication**, `--tag refstate_bci2a_motor8_q4`. Slow
      (2592 trials x 23 pipelines). Cross-dataset confirmation of §4.6.

**Next, in priority order**

1. **Rewrite the manuscript around the new story.** This is now the critical
    path, not more experiments. The paper is currently the §4.4 negative
    benchmark; it needs to become "the frame was the confound, here is the
    proof, here is what survives it". Retitle. Concretely:
    `sec:reference` (Methods) is written; Results needs a reference-frame
    section, a transfer section and a shot-noise section, and the abstract and
    conclusion need rewriting.
2. **Extend `paper/make_tables.py`** to emit macros/tables from
    `reference_gram_motor8.csv`, `raw_folds_refstate_motor8_q4.csv`,
    `transfer_*_motor8.csv` and `shots_*_motor8.csv`. The project's one rule
    is that no number is typed by hand; the new sections must obey it before
    they are written, not after.
3. **Figures** for the reference frame (Gram histograms sensor vs reference),
    transfer (paired per-subject), and the shot curve.
4. Re-run the channel-scaling sweep (§4.1b) **in the reference frame**. The
    claim that density-matrix concentration is dimension-independent was
    measured in the sensor frame and may not survive; §4.6 suggests it will
    not.
5. Scale to 64 channels (6 qubits) for the direct Thanasilp et al. test.
6. Cross-session transfer on IV-2b (5 sessions), where the shift is milder and
    the geometry comparison is cleaner than cross-subject.

**Do not bother with**

- More datasets for their own sake. Two datasets with Spearman 0.882 already
  separate "property of the method" from "property of the data"; a third
  changes no reviewer's mind. Mechanism and controls do.
- Chasing a higher accuracy number. The controls say the ceiling is the
  classical twin's score, and that is the finding.
