"""Text-free panels for the architecture figure, computed from real EEG.

An experiment: the output goes to figures/experiments/architecture_parts/ and
nothing in the paper build reads it until the panels are placed in the
diagram by hand.

The study-design figure (architecture.drawio) has five small graphics: the
EEG traces, the covariance matrix, the density matrix, the kernel matrix and
the SVM panel. They were placeholders. This script computes each one from a
real PhysioNet EEGMMIDB recording, with no axes, ticks, labels or titles, so
they can be dropped into diagrams.net where the figure's own text lives.

    python figures/src/make_architecture_parts.py              # subject 1
    python figures/src/make_architecture_parts.py --subject 7

Subject 1 is the default because it is the subject of Figure 0, so the two
figures show the same person. It decodes at an ordinary level (fidelity
kernel 0.74 in the benchmark), so the panels are not a best case.

What each panel is, exactly:

1  one left-hand trial, eight sensorimotor channels (FC3, FCz, FC4, C3, Cz,
   C4, CP3, CP4 from top to bottom), band-passed 8 to 30 Hz, 0.5 to 3.5 s.
   The trial is the subject's cleanest left-hand trial (smallest peak to
   standard-deviation ratio), so an artefact does not dominate a small panel
2  that trial's OAS covariance C, diverging colour map centred on zero
3  the same trial as a density matrix in the reference frame,
   rho = W C W / tr(W C W), W = M^(-1/2), M the Riemannian mean of the
   subject's trials; same colour map, centred on zero
4  the reference-frame Bures-RBF kernel matrix over all of the subject's
   trials, sorted left then right, values in [0, 1] with the bandwidth from
   the median heuristic, as the pipeline uses
5  the trials of panel 4 in the first two kernel principal components of that
   kernel matrix, coloured by class, with a linear SVM fitted in that plane
   and its margins. The real classifier is an SVM on the full precomputed
   kernel; the plane is a two-dimensional view of it for the diagram.

In the pipeline the reference state and bandwidth are estimated on training
folds only. For a single illustrative subject they are estimated on all of
that subject's trials, which changes nothing visible.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qeeg.data import MOTOR_8, load_subject  # noqa: E402
from qeeg.figures import BLUE, ORANGE  # noqa: E402
from qeeg.quantum import (bures_distance_sq, median_bandwidth,  # noqa: E402
                          reference_whitener, to_density_matrices)

TRACE = "#3a78c9"        # the blue of the placeholder traces
TRACE_MARGIN = 0.6       # blank space above the first and below the last trace
TRACE_SIZE = (2.4, 2.6)  # inches; width / height is used by the diagram builder
DPI = 600


def _circuits(out: Path, rhos: np.ndarray) -> None:
    """The circuits the study used, drawn by Qiskit and checked first.

    6  the IQP feature map of the circuit kernels at the benchmark's width
       (4 qubits), one of its two layers. The same construction with the
       coupling block replaced by a CNOT ring is the ring CNOT kernel, and with
       it deleted is the entanglement ablation. It is verified gate for gate
       against the PennyLane circuit the experiments ran before drawing.
    7  the SWAP test that estimates tr(rho sigma), the Hilbert-Schmidt overlap
       kernel, on two 3-qubit registers. It is verified on two real density
       matrices: P(ancilla = 0) must equal (1 + tr(rho sigma)) / 2.
    """
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.quantum_info import DensityMatrix

    from qeeg.circuits_qiskit import build_feature_map, verify_against_pennylane

    worst = max(verify_against_pennylane(n_qubits=4, n_layers=2).values())
    if worst > 1e-8:
        raise RuntimeError(f"Qiskit and PennyLane circuits differ by {worst:.1e}")

    def _draw(qc, name):
        fig = qc.draw("mpl", fold=-1, initial_state=False,
                      style={"backgroundcolor": "#ffffff"})
        for ext in ("png", "svg", "pdf"):
            fig.savefig(out / f"{name}.{ext}", dpi=400, bbox_inches="tight",
                        facecolor="white")
        plt.close(fig)

    _draw(build_feature_map(None, n_qubits=4, n_layers=1, parameterised=True,
                            entangle=True, entangler="iqp"), "6_circuit_iqp")

    a = QuantumRegister(1, "a")
    r = QuantumRegister(3, "ρ")
    s = QuantumRegister(3, "σ")
    swap = QuantumCircuit(a, r, s)
    swap.h(a[0])
    for i in range(3):
        swap.cswap(a[0], r[i], s[i])
    swap.h(a[0])

    rho, sigma = rhos[0], rhos[1]
    state = (DensityMatrix(sigma).tensor(DensityMatrix(rho))
             .tensor(DensityMatrix.from_label("0")))
    p0 = float(state.evolve(swap).probabilities([0])[0])
    expect = 0.5 * (1.0 + float(np.trace(rho @ sigma)))
    if abs(p0 - expect) > 1e-10:
        raise RuntimeError(f"SWAP test gives P(0) = {p0}, expected {expect}")
    print(f"  SWAP test on two real trials: P(0) = {p0:.6f}, "
          f"(1 + tr rho sigma)/2 = {expect:.6f}")

    c = ClassicalRegister(1, "c")
    swap.add_register(c)
    swap.measure(a[0], c[0])
    _draw(swap, "7_swap_test")


def _bare(ax) -> None:
    ax.set_axis_off()
    ax.margins(0)


def _save(fig, out: Path, name: str) -> None:
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=DPI, transparent=True,
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _heatmap(M: np.ndarray, out: Path, name: str, cmap: str,
             vmin: float, vmax: float) -> None:
    fig, ax = plt.subplots(figsize=(2.4, 2.4))
    ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    _bare(ax)
    _save(fig, out, name)


def _colorbar(out: Path, name: str, cmap: str) -> None:
    fig, ax = plt.subplots(figsize=(0.22, 2.4))
    ax.imshow(np.linspace(1, 0, 256)[:, None], cmap=cmap, aspect="auto")
    _bare(ax)
    _save(fig, out, name)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "figures" / "experiments"
                                         / "architecture_parts"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import mne
    from pyriemann.estimation import Covariances
    from sklearn.decomposition import KernelPCA
    from sklearn.svm import SVC

    mne.set_log_level("ERROR")
    ep = load_subject(args.subject, channels=MOTOR_8)
    X, y = ep.X, ep.y
    assert list(ep.ch_names) == MOTOR_8, ep.ch_names

    # Trials sorted left (0) then right (1), so the kernel shows its blocks.
    order = np.argsort(y, kind="stable")
    X, y = X[order], y[order]
    # The left-hand trial shown in panels 1 to 3: the one with the smallest
    # peak-to-typical amplitude ratio. The first left trial of subject 1 has
    # two blink-like spikes across every channel that swamp the rhythm.
    left = np.flatnonzero(y == 0)
    peak = np.abs(X[left]).max(axis=(1, 2)) / X[left].std(axis=(1, 2))
    trial = int(left[np.argmin(peak)])

    # 1  traces. Fixed layout, no tight crop, so a diagram can align channel
    # labels to the traces: with margin m (in trace spacings), trace i of 8
    # is centred at (m + i) / (7 + 2m) of the image height from the top.
    x = X[trial]
    step = 5.0 * np.std(x)
    m = TRACE_MARGIN
    t = np.arange(x.shape[-1]) / ep.sfreq
    # 2.4 by 2.6 in: the proportions of the slot in the pipeline diagram, so
    # draw.io never has to stretch the traces to fit.
    fig = plt.figure(figsize=TRACE_SIZE)
    ax = fig.add_axes([0, 0, 1, 1])
    for i in range(x.shape[0]):
        ax.plot(t, x[i] - i * step, color=TRACE, lw=0.8)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-(7 + m) * step, m * step)
    ax.set_axis_off()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"1_eeg_traces.{ext}", dpi=DPI, transparent=True)
    plt.close(fig)

    # 2  covariance
    covs = Covariances(estimator="oas").fit_transform(X)
    C = covs[trial]
    a = np.abs(C).max()
    _heatmap(C, out, "2_covariance", "RdBu_r", -a, a)

    # 3  density matrix in the reference frame
    W = reference_whitener(covs)
    rhos = to_density_matrices(np.einsum("ij,njk,kl->nil", W, covs, W))
    R = rhos[trial]
    a = np.abs(R).max()
    _heatmap(R, out, "3_density_matrix", "RdBu_r", -a, a)

    # 4  kernel matrix, Bures-RBF in the reference frame
    D = bures_distance_sq(rhos)
    K = np.exp(-median_bandwidth(D) * D)
    _heatmap(K, out, "4_kernel_matrix", "Blues", 0.0, 1.0)

    # 5  SVM view: kernel PCA plane of the same kernel, linear SVM in it
    Z = KernelPCA(n_components=2, kernel="precomputed").fit_transform(K)
    Z = Z / np.abs(Z).max()
    svm = SVC(kernel="linear", C=1.0).fit(Z, y)
    fig, ax = plt.subplots(figsize=(2.8, 2.4))
    lo, hi = Z.min(axis=0) - 0.12, Z.max(axis=0) + 0.12
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 400),
                         np.linspace(lo[1], hi[1], 400))
    f = svm.decision_function(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    ax.contourf(gx, gy, f, levels=[-1e9, 0, 1e9], colors=[BLUE, ORANGE],
                alpha=0.08)
    ax.contour(gx, gy, f, levels=[-1, 0, 1], colors=["#7a7a7a"] * 3,
               linewidths=[0.7, 1.2, 0.7], linestyles=["--", "-", "--"])
    ax.scatter(*Z[y == 0].T, s=18, color=BLUE, edgecolor="white", lw=0.4,
               zorder=3)
    ax.scatter(*Z[y == 1].T, s=18, facecolor="white", edgecolor=ORANGE,
               lw=1.1, zorder=3)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    _bare(ax)
    _save(fig, out, "5_svm")

    _colorbar(out, "colorbar_diverging", "RdBu_r")
    _colorbar(out, "colorbar_kernel", "Blues")
    _circuits(out, rhos)

    acc = SVC(kernel="precomputed").fit(K, y).score(K, y)
    print(f"subject {args.subject}: {len(y)} trials "
          f"({int((y == 0).sum())} left, {int((y == 1).sum())} right); "
          f"trial {trial} shown; kernel range [{K.min():.2f}, {K.max():.2f}]; "
          f"2-D view separates {svm.score(Z, y):.2f} of trials, "
          f"full-kernel SVM training accuracy {acc:.2f}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
