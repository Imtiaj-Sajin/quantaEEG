# Quantum Machine Learning for EEG — Research Log

**Status:** living document. Started 2026-09-02.
**Question being answered:** is "quantum + EEG" a real research programme or a fascination?

---

## 0. Verdict up front

**The idea is feasible and publishable — but not in the form most papers in this
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
   density matrix**. That is not an analogy — it is an identity. It means the
   entire apparatus of quantum information geometry (Uhlmann fidelity, Bures
   metric, von Neumann entropy, quantum relative entropy) applies to EEG
   *natively*, with no lossy feature-squeezing step. This connection is
   visible in the literature but, as far as this review found, **has not been
   developed into a kernel method and benchmarked properly.**

So: publishable, yes — if the paper's contribution is *rigour plus a principled
representation*, not a leaderboard number. See §6 for what that paper looks
like and §7 for the risk that the honest answer is negative.

**One caution to internalise now:** the most likely outcome of a well-controlled
study here is that quantum methods **match or underperform** strong classical
baselines. That is still a publishable result — a well-executed negative
benchmark in a hyped area has real value — but you should decide up front that
you are willing to publish it, because the alternative is p-hacking your way
into the low-credibility genre described above.

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
| Non-stationary, low SNR | Robustness and regularisation matter more than raw capacity — an argument for structured kernels. |

That fourth row is the strategically important one. **The unsolved problem in
BCI is not within-subject accuracy — it is cross-subject/cross-session
transfer.** Chasing within-subject accuracy puts you in a crowded field where
classical methods are already excellent. Chasing transfer puts you where the
actual pain is. See §6.

---

## 2. Literature landscape

### 2.1 The optimistic strand (claims of quantum benefit on EEG)

- **[EEG-based motor imagery classification with quantum algorithms](https://www.sciencedirect.com/science/article/abs/pii/S0957417424002197)**
  (Expert Systems with Applications, 2024). Reports ~84.86% mean accuracy on
  BCI Competition IV-2a, SD 7.1 across 9 subjects.
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
  (Sci Rep, 2025; 92.3% Bonn, 91.0% CHB-MIT) — note these are **quantum-
  *inspired*** (classical algorithms borrowing quantum formalism), a
  distinction routinely blurred in abstracts.

**Recurring methodological weaknesses across this strand** (this is your
opening):
1. Classical baselines are weak or untuned — often plain SVM/LDA on band power,
   rather than the Riemannian tangent-space or FBCSP methods that actually
   define the state of the art.
2. No entanglement ablation, so "quantumness" is never isolated.
3. No dimension-matched control — the quantum model sees 4–8 PCA features and
   is compared against a classical model that saw something else entirely.
4. Small subject counts, single splits, no paired statistics across subjects.
5. Hyperparameters tuned for the quantum model only.

### 2.2 The skeptical strand (and why you must engage with it)

- **[Better than classical? The subtle art of benchmarking QML models](https://arxiv.org/abs/2403.07059)**
  — Bowles, Ahmed & Schuld (2024). 12 QML models × 6 tasks × 160 datasets.
  Findings: **out-of-the-box classical models outperform the quantum
  classifiers**, and **removing entanglement often leaves performance equal or
  better**. This is the single most important paper to reckon with; a reviewer
  who knows the field *will* ask whether you ran their ablation.
- **[Exponential concentration in quantum kernel methods](https://www.nature.com/articles/s41467-024-49287-w)**
  — Thanasilp, Wang, Cerezo & Holmes, *Nature Communications* 15 (2024).
  Quantum kernel values concentrate exponentially in qubit count toward a fixed
  value, yielding a trivial model. **We reproduced exactly this pathology on
  EEG data — see §4.1.** Engaging with it explicitly is a credibility marker.
- **[Quantum Kernel Methods under Scrutiny: A Benchmarking Study](https://arxiv.org/pdf/2409.04406)**
  (2024) and **[Benchmarking QML kernel training for classification](https://arxiv.org/abs/2408.10274)**
  (2024) — quantum kernel *training* (QKT) often fails to justify its extra
  cost over plain quantum kernel estimation.
- **[Limitations of Amplitude Encoding on Quantum Classification](https://arxiv.org/pdf/2503.01545)** (2025).
- Dequantization results generally: claimed exponential speedups collapse to
  polynomial under sampling-access assumptions.

### 2.3 The bridge that already exists — and where it stops

- **[pyRiemann-qiskit](https://github.com/pyRiemann/pyRiemann-qiskit)**
  ([RIO Journal paper](https://riojournal.com/article/101006/), 2023) — the
  most relevant prior art. Implements QSVC, VQC and a Nearest Convex Hull
  classifier on top of pyRiemann, with a standard pipeline
  (`QuantumClassifierWithDefaultRiemannianPipeline`) for binary brain-wave
  classification. **It is explicitly framed as a "sandbox," not a claim of
  advantage.** Crucially, its quantum step still operates on *tangent-space
  vectors* — the covariance is flattened to a Euclidean vector before the
  quantum model sees it. The density-matrix structure is discarded.
- **[A Unified SPD Token Transformer Framework for EEG Classification](https://arxiv.org/html/2601.21521)** (2026) — notes that the
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
  per-subject optimality* — different subjects have different best decoders,
  which is why paired, subject-wise statistics are mandatory.
- Standard MOABB protocol: `LeftRightImagery`, 8–15 Hz and 8–30 Hz bands,
  within-session evaluation.

---

## 3. The gap, stated precisely

> EEG spatial covariance matrices, normalised to unit trace, are quantum density
> matrices. Quantum information geometry therefore supplies a family of kernels
> — Hilbert–Schmidt overlap, Uhlmann fidelity, Bures metric — that apply to EEG
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
  speedup claim — and saying so plainly is what separates a credible paper from
  the genre in §2.1.

---

## 4. What we built and what we found

Code: [src/qeeg/](src/qeeg/) — `data.py`, `quantum.py`, `pipelines.py`,
`benchmark.py`. Dataset: PhysioNet EEG Motor Movement/Imagery (EEGMMIDB),
left-fist vs right-fist imagery (runs 4/8/12), 8 sensorimotor channels,
8–30 Hz, 0.5–3.5 s window, ~45 trials/subject.

Protocol: **nested CV** — outer 5-fold × 3 repeats for generalisation, inner
4-fold `GridSearchCV` for hyperparameters, applied *identically to every
pipeline including all classical baselines*. Paired Wilcoxon signed-rank across
subjects with Holm correction.

### 4.1 Finding 1 — quantum overlap kernels concentrate catastrophically on EEG

Measured on subject S001, 45 trials, 8 channels (off-diagonal Gram entries after
cosine normalisation):

| Kernel | mean | std | min | max |
|---|---|---|---|---|
| Hilbert–Schmidt `tr(ρσ)` | 0.99082 | 0.01008 | 0.9216 | 0.9997 |
| Uhlmann fidelity `F(ρ,σ)` | 0.98807 | 0.00991 | 0.9400 | 0.9996 |
| `sqrt(F)` | 0.99400 | 0.00501 | 0.9695 | 0.9998 |

Every pair of trials looks ~99% identical. The Gram matrix is effectively
rank-one and the SVM has nothing to separate. **This is the Thanasilp et al.
concentration pathology, reproduced on real biological data** — and it is a
concrete, quantitative contribution in its own right, because that paper's
examples are largely synthetic.

For the IQP circuit kernel the failure is the mirror image — over-dispersion:

| Circuit kernel | off-diag mean | std |
|---|---|---|
| IQP, entanglement ON | 0.13433 | 0.15743 |
| IQP, entanglement OFF | 0.37967 | 0.25065 |

Entanglement pushes states toward mutual orthogonality, i.e. toward a kernel
that memorises the training set. Both regimes are useless for the same
underlying reason: no tunable length scale.

### 4.1b Finding 2 — entanglement *accelerates* the collapse (new result)

We swept register size on real EEG by exploiting the identity directly: a
d-channel covariance is a density matrix on log2(d) qubits, so sweeping
channels over powers of two sweeps qubits with **no change of method** —
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
loses variance at 0.428 per qubit — it more than *halves* with every qubit
added — while the identical circuit with entanglers deleted decays at 0.716.
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
HS/fidelity kernels do **not** follow the qubit-count law — their variance
*rises* with dimension (factor 1.53–1.99). They are already saturated at 2
qubits (variance ~1e-4, mean 0.99) because **EEG covariance matrices are
intrinsically similar to one another**, not because of register size. So there
are two distinct pathologies wearing the same name:

- *Circuit kernels*: concentration driven by qubit count and entanglement
  (the Thanasilp et al. mechanism).
- *Density-matrix overlap kernels*: concentration driven by the **data
  distribution** — dimension-independent, and in fact *relieved* by adding
  channels.

The practical corollary is encouraging for the density-matrix route: unlike
circuit kernels, it should be run at **more** channels, not fewer. That is the
opposite of what the standard "PCA down to 4 features" recipe does.

### 4.2 The remedy we implemented

Keep the quantum *geometry*, restore a length scale — exponentiate the induced
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

**Classical baselines** — the bar to clear:
`logvar+LDA`, `CSP+LDA`, `MDM`, `TS+LR`, `TS+RBF-SVM`

**Quantum:**
`HS-overlap-SVM`, `Fidelity-SVM` (raw, to document concentration),
`HS-RBF-SVM`, `Bures-RBF-SVM` (bandwidth-corrected),
`IQP-kernel-SVM`, `CNOT-kernel-SVM`

**Controls** — what makes this a real study:
- `IQP-no-entangle` — identical circuit, entanglers deleted (the Bowles et al.
  ablation).
- `PCA-matched-RBF`, `PCA-matched-linear` — classical kernels on the *same*
  4-dimensional PCA features the circuit kernels see. Without this, any circuit
  result is confounded with dimensionality reduction.
- `logeuclid-TS+LR` — the classical geometry twin of the density-matrix kernels.

### 4.4 Results

> Full run in progress (30 subjects, nested CV). Table to be inserted here from
> `results/summary_motor8_q4.csv` and `results/tests_vs_*.csv`.

Preliminary 3-subject smoke run (5-fold, 1 repeat — **not** inferential, shown
only for order of magnitude): classical `CSP+LDA` 0.696, `MDM` 0.674,
`TS+LR` 0.667 vs quantum `IQP-kernel-SVM` 0.600, `Bures-RBF-SVM` 0.578,
`HS-overlap-SVM` 0.548. Direction of travel is consistent with §2.2.

---

## 5. Datasets worth using

| Dataset | Access | Size | Why |
|---|---|---|---|
| **PhysioNet EEGMMIDB** | `mne.datasets.eegbci` — works, no auth | 109 subjects, 64 ch, 160 Hz | Large subject count for paired stats. **Currently used.** |
| **BCI Competition IV-2a** | MOABB `BNCI2014_001` | 9 subjects, 22 ch, 4 classes | The genre's default benchmark — needed for comparability with §2.1 papers. |
| **BCI Competition IV-2b** | MOABB `BNCI2014_004` | 9 subjects, 3 ch | Multi-session — good for transfer experiments. |
| **Cho2017** | MOABB | 52 subjects | Large MI dataset, good for cross-subject. |
| **CHB-MIT** | PhysioNet | 23 patients | Seizure detection; heavy class imbalance. |
| **TUH EEG** | Registration required | Thousands of clinical recordings | Largest clinical corpus; the credible route to "scale." |

Recommendation: **PhysioNet MI (breadth) + BCI IV-2a (comparability)**. Add
Cho2017 if the cross-subject story in §6 becomes the centre of the paper.

---

## 6. What the publishable paper actually is

Do **not** write "Quantum SVM achieves 85% on EEG." Write one of these:

**Option A — the rigorous negative/neutral benchmark (safest, still valuable).**
*"Do quantum kernels help EEG decoding? A controlled benchmark with
entanglement ablations."* Contribution = methodology + honest answer + open
code. Precedent: Bowles et al. is highly cited precisely for this. Venue:
*Journal of Neural Engineering*, *EPJ Quantum Technology*, *Quantum Machine
Intelligence*.

**Option B — the principled-representation paper (higher ceiling).**
*"EEG covariance matrices as quantum states: fidelity and Bures kernels for
brain–computer interfaces."* Contribution = the density-matrix formulation, the
concentration diagnosis on real data, the bandwidth remedy, and a fair
benchmark. Works **even if the quantum kernels only match** classical ones,
because the framework and the diagnosis are the contribution. Venue: *Journal
of Neural Engineering*, *IEEE TNSRE*, *Quantum Machine Intelligence*.

**Option C — the one with the highest scientific upside (hardest).**
Target **cross-subject transfer**, not within-subject accuracy. Hypothesis:
quantum-information distances (Bures, quantum relative entropy) are more robust
to the covariance shift between subjects than affine-invariant Riemannian
distances. If true, it is a real result with a real mechanism, addressing the
field's actual bottleneck (the calibration problem). If false, it folds back
into Option A/B. **This is where I would aim.**

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
| Quantum methods simply lose | **High — expect it** | Frame as Option A/B, where a rigorous negative result is the contribution. Decide *now* that you will publish it. |
| Reviewer: "no quantum advantage, so what?" | High | Pre-empt: the paper is a benchmark + representation contribution, not an advantage claim. Cite Bowles et al. as precedent. |
| Reviewer: "this is classically simulable" | Certain | Concede immediately and precisely. 8 ch = 3 qubits; everything here is classically computable. The claim is *geometric*, not computational. |
| Kernel concentration kills everything | **Already happened** | Documented and remedied (§4.1–4.2). Now an asset, not a threat. |
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

- [ ] Add MOABB + BCI IV-2a for comparability with §2.1 claims.
- [ ] Cross-subject / transfer evaluation (Option C).
- [ ] Quantum relative entropy `S(ρ‖σ)` as an additional divergence.
- [ ] Riemannian-kernel controls (`riemann`, `logeuclid` from pyRiemann) —
      currently only the tangent-space classifiers stand in for these.
- [ ] Shot-noise simulation: everything here is infinite-shot. Real hardware
      estimation of `tr(ρσ)` needs O(1/ε²) shots — quantify the degradation.
- [ ] Scale the density-matrix kernels to 64 channels (6 qubits) and test
      whether concentration *worsens* with qubit count, as theory predicts.
      **This is a direct empirical test of Thanasilp et al. on real data and is
      probably the single highest-value remaining experiment.**
