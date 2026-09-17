# Figure 1 caption

The figure carries the structure; everything quantitative lives here.

**Study design and the control that decides the result.**

**(a) EEG.** Three public motor-imagery datasets are reduced to one common
representation: PhysioNet EEGMMIDB (104 subjects, 45 trials each), BCI
Competition IV 2a (9 subjects, 288 trials, two sessions) and Cho2017 (52
subjects, 200 trials). Each is band-pass filtered to 8 to 30 Hz and cut to its
dataset's standard motor-imagery window (0.5 to 3.5 s after the cue on
PhysioNet) over the same eight sensorimotor electrodes (FC3, FCz, FC4,
C3, Cz, C4, CP3, CP4, filled in the montage), so the register size is identical
across datasets and the comparisons are like for like.

**(b) State.** Each epoch becomes an 8 by 8 spatial covariance matrix *C*.
Divided by its trace it is a density matrix *ρ* on log₂8 = 3 qubits. This is an
identity rather than an analogy: there is no dimensionality reduction and no
embedding of features into rotation angles, and the Hilbert-Schmidt overlap
tr(*ρσ*) is exactly the quantity a SWAP test estimates.

**(c) Frame.** EEG's nuisance factors, electrode gain, head geometry, skull
conductivity and source mixing, act on the covariance by congruence,
*C* → *ACA*ᵀ. In the sensor frame the density-matrix kernels are invariant only
under the orthogonal subgroup *OCO*ᵀ, while every strong classical baseline is
invariant under the full congruence group, so a comparison made there measures
invariance groups rather than geometry. Referring each state to a reference
state, *W* = *M*^(−1/2) with the Fréchet mean *M* estimated label-free from
training trials only, restores exact invariance under *ACA*ᵀ. Every suite in
the study is run in both frames, changing nothing else.

**(d) Kernels.** Four families are evaluated under one protocol: tuned
classical baselines (CSP+LDA, tangent space with logistic regression, MDM,
FBCSP), the metric-matched classical twin (a Riemannian SPD kernel, with the
log-Euclidean kernel reported alongside), the quantum density-matrix kernels
(Hilbert-Schmidt overlap, Uhlmann fidelity, Bures, quantum relative entropy)
and the circuit kernels with their entanglement ablation and dimension-matched
PCA control. The bracket marks the comparison the paper turns on: the twin and
the quantum kernels consume the same covariances, sit in the same support
vector machine, receive the same tuning budget and are expressed in the same
frame, so the only thing that differs between them is the metric.

**(e) Evaluation.** Nested cross-validation throughout: an outer loop of 5
folds by 3 repeats for generalisation, an inner 4-fold grid search for
hyperparameters, with the identical search budget for every pipeline, quantum
and classical alike. Inference is paired per subject by Wilcoxon signed-rank
with Holm correction inside each pre-specified family, and equivalence against
the twin is tested by two one-sided tests. The settings span within-subject
decoding, cross-subject transfer (leave one subject out), cross-session
transfer, few-trial calibration, and registers from 3 to 6 qubits.

## Files

| File | What it is |
|---|---|
| `figures/architecture.drawio` | **the source**, authored in diagrams.net by the corresponding author; edit this |
| `figures/architecture.pdf` | vector, the version the manuscript includes |
| `figures/architecture.svg` | intermediate, also usable directly |
| `figures/architecture_preview.png` | raster, for checking the output |
| `figures/architecture.pdf` | exported from diagrams.net; what the manuscript includes |
| `figures/src/drawio_to_pdf.py` | fallback renderer, for when the export paginates |
| `figures/src/scene.py` | the drawing primitives and the SVG exporter both paths share |
| `figures/versions/architecture_tworow.*` | an alternative layout, kept in case it is wanted back |
| `figures/src/make_architecture.py` | builds that alternative |

After editing the figure in diagrams.net, regenerate the PDF with

```bash
python figures/src/drawio_to_pdf.py paper/eeg_quantum_architecture_v3.drawio.xml     --out figures/architecture
```

The editable source also ships inside `paper/build/submission.zip` as
`architecture_drawio.xml`, so a co-author or a later reader can change the
figure rather than being stuck with a flat PDF. IOP's upload rules allow only
letters, digits and underscore in file names, which is why it is renamed.

## How the PDF is produced

From diagrams.net's own export, which for v4 is correct: one page, 590 by
523 pt, 276 vector drawings, no raster images, and Liberation Sans and DejaVu
Sans both subset and embedded. That is publication-grade, so it is used as it
comes.

`figures/src/drawio_to_pdf.py` remains as a fallback. It reads a `.drawio`
directly, resolves group offsets, crops to the drawing's bounding box and emits
one page. It was written because the v3 file declared a 1560 by 400 page while
the drawing spanned about 840 by 750 at an offset, so that export came out as
four page fragments with the figure cut across them. If a future edit
reintroduces that, run:

```bash
python figures/src/drawio_to_pdf.py figures/architecture.drawio     --out figures/architecture
```

Note that the fallback substitutes a superscript italic *T* for U+22A4 and a
plain arrow for U+21A6, because Arial carries neither; diagrams.net's own
export renders both correctly through Liberation Sans, which is one reason to
prefer it.

**After editing the figure, re-export the PDF.** The `.drawio` and the `.pdf`
are two files, and nothing in the build checks that they agree.

## Why two rows

A single row of five panels composes better on a screen and is illegible on the
page. At IOP's single-column width of 160 mm a 1560-unit canvas shrinks to
about 41 mm tall and a 10-unit label renders at 2.9 pt, well below the 6 pt
most journals require. The same five panels on an 880-unit canvas in two rows
put body labels at 6.7 pt and panel titles near 8.7 pt.

## Two portability notes

Cairo is not available on this machine, so the SVG is converted to PDF with
svglib and reportlab rather than cairosvg, and the preview is rasterised from
the PDF with PyMuPDF. Going through the PDF has the side benefit that the
preview shows exactly what the published vector shows.

Arial carries neither U+22A4 (⊤) nor U+21A6 (↦), and the DejaVu fallback does
not survive svglib's font resolution, so both rendered as empty boxes. The
transpose is set as a superscript italic *T*, which is the standard notation in
any case, and the maps-to is a plain right arrow.
