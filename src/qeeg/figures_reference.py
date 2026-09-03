"""Figures for the reference-frame results, and the two schematics the paper needs.

Companion to `figures.py`, reusing its validated palette and style helpers so
the whole figure set reads as one system. Every figure writes 300-dpi PNG plus
vector PDF, and each builder returns False (rather than raising) when its
inputs are absent, so a partial checkout still produces what it can.

    PYTHONPATH=src python -m qeeg.figures_reference --paper

Figures
-------
fig5_circuits   the three circuit feature maps, including the entanglement
                ablation -- a quantum paper should show its circuits
fig6_reference  why the sensor frame is the wrong one: schematic, the
                invariance check to machine precision, and the measured
                relief of concentration
fig7_frame      what the correction is worth, per subject, on both datasets
fig8_twin       the decisive control: quantum kernels against a classical
                kernel differing only in the metric, in four settings
fig9_transfer   cross-subject transfer, and the finite-shot budget
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from . import figures as F  # noqa: E402
from .figures import (  # noqa: E402
    AQUA, BLUE, GREEN, GRID, INK, INK_2, INK_MUTED, MAGENTA, ORANGE, YELLOW,
)

SENSOR_C, REF_C = ORANGE, BLUE

FRAME_PAIRS = [
    ("quantum/Fidelity-SVM", "quantum/Fidelity-ref-SVM", "Fidelity"),
    ("quantum/HS-overlap-SVM", "quantum/HS-overlap-ref-SVM", "HS overlap"),
    ("quantum/HS-RBF-SVM", "quantum/HS-RBF-ref-SVM", "HS-RBF"),
    ("quantum/Bures-RBF-SVM", "quantum/Bures-RBF-ref-SVM", "Bures-RBF"),
    ("quantum/QRE-RBF-SVM", "quantum/QRE-RBF-ref-SVM", "QRE-RBF"),
]


def _per(df):
    return df.groupby(["pipeline", "subject"])["accuracy"].mean().unstack("pipeline")


def _read(res: Path, name: str):
    p = res / name
    return pd.read_csv(p) if p.exists() else None


# ==========================================================================
# Figure 5: the circuit feature maps
# ==========================================================================

def _gate(ax, x, y, label, color=BLUE, w=0.44, h=0.44):
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor="white", edgecolor=color, lw=1.3, zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=7.4,
            color=color, zorder=4, fontweight="bold")


def _cnot(ax, x, yc, yt, color=INK_2, r=0.13):
    """Standard notation: filled control dot, open target with a cross."""
    ax.plot([x, x], [yc, yt], color=color, lw=1.1, zorder=2)
    ax.plot([x], [yc], marker="o", ms=5.0, color=color, zorder=4)
    ax.add_patch(plt.Circle((x, yt), r, facecolor="white", edgecolor=color,
                            lw=1.2, zorder=4))
    ax.plot([x - r, x + r], [yt, yt], color=color, lw=1.1, zorder=5)
    ax.plot([x, x], [yt - r, yt + r], color=color, lw=1.1, zorder=5)


def fig_circuits(out: Path) -> bool:
    """Delegate to the Qiskit renderer.

    This used to draw the circuits by hand in matplotlib. That is a hazard: a
    hand-drawn diagram can silently stop matching the circuit the experiments
    run. The figure is now produced by `figures_circuits`, which builds the
    circuit in Qiskit and verifies it against the executed PennyLane circuit
    before drawing anything. Kept as a thin wrapper so the existing entry point
    cannot regenerate the old, unverified version.
    """
    from .figures_circuits import fig_circuits as _qiskit_circuits

    return _qiskit_circuits(out)


def _schematic(ax):
    """States clustered about the mean, versus spread about the identity."""
    rng = np.random.default_rng(3)
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.9)

    # Sensor frame: a tight cluster far from the origin, plus a nuisance arrow.
    cx, cy = 2.35, 2.35
    pts = rng.normal(size=(26, 2)) * 0.22 + np.array([cx, cy])
    ax.scatter(*pts.T, s=16, color=SENSOR_C, alpha=0.75, zorder=3, lw=0)
    ax.plot([cx], [cy], marker="*", ms=13, color=INK, zorder=4)
    ax.text(cx, cy + 0.42, "$M$", fontsize=9, ha="center", color=INK)
    ax.annotate("", xy=(4.05, 3.35), xytext=(cx + 0.5, cy + 0.25),
                arrowprops=dict(arrowstyle="-|>", color=INK_MUTED, lw=1.2))
    ax.text(4.15, 3.55, "nuisance\n$C\\mapsto ACA^{\\mathsf{T}}$", fontsize=7.6,
            color=INK_MUTED, va="center", linespacing=1.5)
    pts2 = pts @ np.array([[1.35, 0.42], [-0.3, 0.78]]) + np.array([0.3, 0.55])
    ax.scatter(*pts2.T, s=16, color=SENSOR_C, alpha=0.28, zorder=2, lw=0)
    ax.text(2.35, 0.62, "Sensor frame", fontsize=9, fontweight="bold",
            color=INK, ha="center")
    ax.text(2.35, 0.18, "cluster moves under $A$", fontsize=7.8,
            color=INK_MUTED, ha="center")

    ax.annotate("", xy=(6.0, 2.35), xytext=(5.2, 2.35),
                arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=1.4))
    ax.text(5.6, 2.62, "$W=M^{-1/2}$", fontsize=8, ha="center", color=INK_2)

    # Reference frame: spread about the identity, invariant up to a rotation.
    cx2, cy2 = 8.0, 2.35
    ang = rng.uniform(0, 2 * np.pi, 26)
    rad = 0.42 + rng.normal(scale=0.13, size=26)
    p3 = np.stack([cx2 + rad * np.cos(ang), cy2 + rad * np.sin(ang)], 1)
    ax.scatter(*p3.T, s=16, color=REF_C, alpha=0.8, zorder=3, lw=0)
    ax.plot([cx2], [cy2], marker="*", ms=13, color=INK, zorder=4)
    ax.text(cx2 - 0.30, cy2 - 0.06, "$I$", fontsize=9, ha="right",
            va="center", color=INK)
    th = np.linspace(0.5, 2.5, 60)
    ax.plot(cx2 + 0.86 * np.cos(th), cy2 + 0.86 * np.sin(th), color=INK_MUTED,
            lw=1.0, ls=(0, (3, 2)))
    ax.annotate("", xy=(cx2 + 0.86 * np.cos(th[-1]), cy2 + 0.86 * np.sin(th[-1])),
                xytext=(cx2 + 0.86 * np.cos(th[-6]), cy2 + 0.86 * np.sin(th[-6])),
                arrowprops=dict(arrowstyle="-|>", color=INK_MUTED, lw=1.0))
    ax.text(cx2, 4.55, "$A$ becomes a rotation $U$;\nunitary invariants "
            "are unchanged", fontsize=7.6, color=INK_MUTED, va="top",
            ha="center", linespacing=1.5)
    ax.text(cx2, 0.62, "Reference frame", fontsize=9, fontweight="bold",
            color=INK, ha="center")
    ax.text(cx2, 0.18, "spread, and invariant", fontsize=7.8,
            color=INK_MUTED, ha="center")


def fig_reference(res: Path, out: Path) -> bool:
    gram = _read(res, "reference_gram_motor8.csv")
    if gram is None:
        return False
    F._style()
    fig = plt.figure(figsize=(9.4, 5.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.02, 1.0], hspace=0.38,
                          wspace=0.42)

    ax0 = fig.add_subplot(gs[0, :])
    _schematic(ax0)
    ax0.text(0.0, 4.15, "a", fontsize=11, fontweight="bold", color=INK)

    # (b) invariance verified numerically, log scale.
    ax1 = fig.add_subplot(gs[1, 0])
    from .reference import check_invariance
    r = check_invariance()
    r.pop("cond_A", None)
    names = list(r)
    xs = np.arange(len(names))
    ax1.bar(xs - 0.19, [r[k]["sensor"] for k in names], width=0.36,
            color=SENSOR_C, label="Sensor")
    ax1.bar(xs + 0.19, [max(r[k]["reference"], 1e-17) for k in names],
            width=0.36, color=REF_C, label="Reference")
    ax1.set_yscale("log")
    ax1.set_ylim(1e-17, 5)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(["HS", "Fid.", "Bures", "QRE"], fontsize=8)
    ax1.set_ylabel("max change under $A$")
    ax1.axhline(1e-15, color=INK_MUTED, lw=0.8, ls=":")
    ax1.set_xlim(-0.62, len(names) - 0.38)
    ax1.text(-0.55, 6e-15, "machine precision", fontsize=7,
             color=INK_MUTED, ha="left", va="bottom")
    ax1.legend(fontsize=7.5, loc="upper left")
    F._despine(ax1)
    ax1.text(-0.22, 1.06, "b", transform=ax1.transAxes, fontsize=11,
             fontweight="bold", color=INK)

    # (c) concentration relief: Gram variance per frame.
    ax2 = fig.add_subplot(gs[1, 1])
    # Labels derive from the data, so a rename in reference.py cannot silently
    # desynchronise the tick labels from the bars.
    short = {"HS-overlap": "HS", "Fidelity": "Fid.", "Bures-d2": "Bures",
             "QRE": "QRE"}
    order = [k for k in ("HS-overlap", "Fidelity", "Bures-d2", "QRE")
             if k in set(gram.kernel)]
    sen = [gram[(gram.frame == "sensor") & (gram.kernel == k)]["var"].mean()
           for k in order]
    ref = [gram[(gram.frame == "reference") & (gram.kernel == k)]["var"].mean()
           for k in order]
    xs = np.arange(len(order))
    ax2.bar(xs - 0.19, sen, width=0.36, color=SENSOR_C)
    ax2.bar(xs + 0.19, ref, width=0.36, color=REF_C)
    for i, (a, b) in enumerate(zip(sen, ref)):
        ax2.text(i + 0.19, b * 1.12, f"{b / a:.1f}$\\times$", ha="center",
                 fontsize=7.4, color=INK_2)
    ax2.set_yscale("log")
    ax2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_xticks(xs)
    ax2.set_xticklabels([short.get(k, k) for k in order], fontsize=8)
    ax2.set_ylabel("Gram variance")
    F._despine(ax2)
    ax2.text(-0.24, 1.06, "c", transform=ax2.transAxes, fontsize=11,
             fontweight="bold", color=INK)

    # (d) where the overlaps actually sit.
    ax3 = fig.add_subplot(gs[1, 2])
    for frame, col in (("sensor", SENSOR_C), ("reference", REF_C)):
        sub = gram[(gram.frame == frame) & (gram.kernel == "HS-overlap")]
        ax3.errorbar(sub["mean"], np.arange(len(sub)),
                     xerr=sub["std"], fmt="o", ms=2.6, lw=0.8, color=col,
                     alpha=0.85, label=frame.capitalize())
    ax3.set_xlabel(r"$\mathrm{tr}(\rho\sigma)$, off-diagonal")
    ax3.set_ylabel("subject")
    ax3.set_xlim(0.5, 1.02)
    ax3.legend(fontsize=7.5, loc="lower left")
    F._despine(ax3)
    ax3.text(-0.26, 1.06, "d", transform=ax3.transAxes, fontsize=11,
             fontweight="bold", color=INK)

    F._caption(fig, (
        "(a) EEG covariances cluster tightly about their mean, and the whole "
        "cluster moves under a nuisance congruence.\n"
        "Whitening by the mean spreads the states and reduces the nuisance to "
        "a rotation, which the quantum\ninvariants do not see. (b) That "
        "invariance verified numerically. (c, d) The measured consequence on "
        "real EEG,\nn = 14 subjects."))
    F._save(fig, out, "fig6_reference")
    return True


# ==========================================================================
# Figure 7: what the correction is worth
# ==========================================================================

def fig_frame(res: Path, out: Path) -> bool:
    phys = _read(res, "raw_folds_refstate_motor8_q4.csv")
    if phys is None:
        return False
    bci = _read(res, "raw_folds_refstate_bci2a_motor8_q4.csv")
    blocks = [("PhysioNet", _per(phys))]
    if bci is not None:
        blocks.append(("BCI IV-2a", _per(bci)))

    F._style()
    fig, axes = plt.subplots(1, len(blocks) + 1, figsize=(9.4, 3.5),
                             gridspec_kw={"width_ratios": [1] * len(blocks) + [1.2]})
    axes = np.atleast_1d(axes)

    for ax, (name, per) in zip(axes, blocks):
        # Per-subject lines for one kernel only. Overlaying all five gives 150
        # crossing lines that say nothing; Fidelity carries the largest effect
        # and is representative of the direction.
        a0, b0, lab0 = FRAME_PAIRS[0]
        if a0 in per.columns and b0 in per.columns:
            for s in per.index:
                ax.plot([0, 1], [per.loc[s, a0], per.loc[s, b0]],
                        color=INK_MUTED, lw=0.5, alpha=0.3, zorder=1)
        entries = []
        for a, b, lab in FRAME_PAIRS:
            if a not in per.columns or b not in per.columns:
                continue
            ax.plot([0, 1], [per[a].mean(), per[b].mean()], color=REF_C,
                    lw=2.0, marker="o", ms=4.5, zorder=3)
            entries.append((lab, 1.0, per[b].mean(), INK_2))
        ax.axhline(0.5, color=INK_MUTED, lw=0.8, ls=":")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Sensor", "Reference"], fontsize=8.5)
        # The end labels are drawn 9pt right of x=1 in axes coordinates, so the
        # panel needs headroom past the "Reference" tick or they overrun it.
        ax.set_xlim(-0.22, 2.45)
        ax.set_ylabel("accuracy" if ax is axes[0] else "")
        ax.set_title(f"{name}  ($n={len(per)}$)", fontsize=9.5, color=INK,
                     loc="left", pad=8)
        F._despine(ax)
        # The five means sit within a few accuracy points of one another, so
        # the labels must be pushed apart or they overprint.
        F._place_end_labels(ax, entries)

    # Effect size per kernel per dataset.
    ax = axes[-1]
    width = 0.36
    for bi, (name, per) in enumerate(blocks):
        vals, labs = [], []
        for a, b, lab in FRAME_PAIRS:
            if a in per.columns and b in per.columns:
                vals.append(per[b].mean() - per[a].mean())
                labs.append(lab)
        ys = np.arange(len(vals))
        ax.barh(ys + (bi - 0.5) * width, vals, height=width,
                color=[SENSOR_C, REF_C][bi], label=name)
    ax.set_yticks(np.arange(len(labs)))
    ax.set_yticklabels(labs, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color=INK_2, lw=0.9)
    ax.set_xlabel(r"$\Delta$ accuracy, reference $-$ sensor")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.14),
              ncol=2)
    F._despine(ax)

    F._caption(fig, (
        "Thin lines are individual subjects, shown for the Fidelity kernel "
        "only; overlaying all five would give 150\ncrossing lines. Every "
        "kernel improves on both datasets, and on IV-2a every kernel improves "
        "in every\nsubject. The correction is worth two to three times more on "
        "IV-2a, where the reference state is estimated\nfrom six times as many "
        "trials."))
    F._save(fig, out, "fig7_frame")
    return True


# ==========================================================================
# Figure 8: the decisive control
# ==========================================================================

def _ci(d):
    d = np.asarray(d, float)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), 1.96 * se


def fig_twin(res: Path, out: Path) -> bool:
    settings = []
    phys = _read(res, "raw_folds_refstate_motor8_q4.csv")
    if phys is not None:
        settings.append(("PhysioNet, 3 qubits", _per(phys),
                         [b for _, b, _ in FRAME_PAIRS],
                         "control/riemann-kernel-SVM"))
    fb = _read(res, "raw_folds_filterbank_motor8.csv")
    if fb is not None:
        settings.append(("PhysioNet, 5 qubits", _per(fb),
                         [b.replace("quantum/", "quantum/FB-")
                          for _, b, _ in FRAME_PAIRS],
                         "control/FB-logeuclid-kernel-SVM"))
    tr = _read(res, "transfer_folds_motor8.csv")
    if tr is not None:
        per_t = (tr[tr.frame == "reference"]
                 .groupby(["pipeline", "subject"])["accuracy"]
                 .mean().unstack("pipeline"))
        settings.append(("Transfer (LOSO)", per_t,
                         ["quantum/Fidelity", "quantum/HS-overlap",
                          "quantum/HS-RBF", "quantum/Bures-RBF",
                          "quantum/QRE-RBF"],
                         "control/riemann-kernel-SVM"))
    bci = _read(res, "raw_folds_refstate_bci2a_motor8_q4.csv")
    if bci is not None:
        settings.append(("BCI IV-2a, 3 qubits", _per(bci),
                         [b for _, b, _ in FRAME_PAIRS],
                         "control/riemann-kernel-SVM"))
    if not settings:
        return False

    F._style()
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    labels, ypos, y = [], [], 0.0
    groups = []                       # (name, y_top, y_bottom) for banding
    for name, per, kernels, twin in settings:
        if twin not in per.columns:
            continue
        y_top = y
        for k in kernels:
            if k not in per.columns:
                continue
            d = (per[k] - per[twin]).dropna()
            m, h = _ci(d)
            col = REF_C if abs(m) < h else ORANGE
            ax.errorbar(m, y, xerr=h, fmt="o", ms=4.6, lw=1.3, color=col,
                        capsize=2.4, zorder=3)
            labels.append(k.split("/")[-1].replace("-SVM", "")
                          .replace("FB-", "").replace("-ref", ""))
            ypos.append(y)
            y -= 1.0
        groups.append((name, y_top, y + 1.0, len(per)))
        y -= 1.0

    xmax = 0.043
    # Group name sits in the dead space to the right, so it never collides
    # with the kernel labels on the left.
    for gi, (name, y_top, y_bot, n) in enumerate(groups):
        if gi % 2 == 0:
            ax.axhspan(y_bot - 0.45, y_top + 0.45, color=GRID, alpha=0.35,
                       zorder=0)
        ax.text(0.0245, (y_top + y_bot) / 2, f"{name}\n$n={n}$", fontsize=8.4,
                fontweight="bold", color=INK, va="center", ha="left",
                linespacing=1.5)

    ax.axvline(0, color=INK, lw=1.1, zorder=2)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(r"$\Delta$ accuracy, quantum kernel $-$ classical twin"
                  "\n(95% CI; same data, same SVM, same frame, only the metric differs)")
    ax.set_xlim(-0.038, xmax)
    ax.set_xticks([-0.03, -0.02, -0.01, 0.0, 0.01, 0.02])
    ax.set_ylim(y + 0.6, 1.0)
    F._despine(ax)

    F._caption(fig, (
        "Every interval crosses zero: no quantum kernel is distinguishable "
        "from a classical SPD kernel that differs from it\nonly in the metric, "
        "in any of the four settings. Note the axis range: the whole plot spans "
        "eight accuracy\npoints, against a frame effect of up to 19."))
    F._save(fig, out, "fig8_twin")
    return True


# ==========================================================================
# Figure 9: transfer and shot budget
# ==========================================================================

def fig_transfer_shots(res: Path, out: Path) -> bool:
    tr = _read(res, "transfer_folds_motor8.csv")
    sh = _read(res, "shots_folds_motor8.csv")
    if tr is None and sh is None:
        return False
    F._style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})

    if tr is not None:
        ax = axes[0]
        ref = (tr[tr.frame == "reference"].groupby("pipeline")["accuracy"].mean())
        sen = (tr[tr.frame == "sensor"].groupby("pipeline")["accuracy"].mean())
        order = ref.sort_values().index
        ys = np.arange(len(order))
        for y, p in zip(ys, order):
            ax.plot([sen[p], ref[p]], [y, y], color=GRID, lw=2.4, zorder=1,
                    solid_capstyle="round")
        grp = ["quantum" if p.startswith("quantum") else "classical"
               for p in order]
        ax.scatter(sen[order], ys, s=26, color=SENSOR_C, zorder=3, label="Sensor")
        ax.scatter(ref[order], ys, s=26, color=REF_C, zorder=3, label="Reference")
        for y, p, g in zip(ys, order, grp):
            ax.text(0.462, y, "Q" if g == "quantum" else "C", fontsize=7,
                    color=ORANGE if g == "quantum" else BLUE, va="center",
                    fontweight="bold")
        ax.set_yticks(ys)
        ax.set_yticklabels([p.split("/")[-1].replace("-SVM", "")
                            for p in order], fontsize=7.6)
        ax.axvline(0.5, color=INK_MUTED, lw=0.8, ls=":")
        ax.set_xlabel("leave-one-subject-out accuracy")
        ax.set_xlim(0.45, 0.70)
        ax.legend(fontsize=7.5, loc="lower right")
        F._despine(ax)
        ax.text(-0.42, 1.05, "a", transform=ax.transAxes, fontsize=11,
                fontweight="bold", color=INK)

    if sh is not None:
        ax = axes[1]
        for kern, ls in (("HS-overlap", "-"), ("HS-RBF", "--")):
            sub = sh[sh.kernel == kern]
            if not len(sub):
                continue
            for frame, col in (("sensor", SENSOR_C), ("reference", REF_C)):
                s = sub[(sub.frame == frame) & (sub.shots > 0)]
                g = s.groupby("shots")["accuracy"].mean().sort_index()
                ax.plot(g.index, g.values, ls, color=col, lw=1.5, marker="o",
                        ms=3.4)
            inf = sub[(sub.frame == "sensor") & (sub.shots == -1)].accuracy.mean()
            ax.axhline(inf, color=SENSOR_C, lw=0.9, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("shots per Gram entry")
        ax.set_ylabel("accuracy")
        ax.axhline(0.5, color=INK_MUTED, lw=0.8, ls=":")
        ax.text(9.0e5, 0.503, "chance", fontsize=7, color=INK_MUTED,
                ha="right", va="bottom")
        ax.text(9.0e5, 0.5585, "sensor frame, infinite shots", fontsize=7,
                color=SENSOR_C, ha="right", va="bottom")
        ax.plot([], [], "-", color=REF_C, label="Reference")
        ax.plot([], [], "-", color=SENSOR_C, label="Sensor")
        ax.plot([], [], "-", color=INK_MUTED, label="HS overlap")
        ax.plot([], [], "--", color=INK_MUTED, label="HS-RBF")
        ax.legend(fontsize=7, loc="lower right", ncol=2)
        F._despine(ax)
        ax.text(-0.24, 1.05, "b", transform=ax.transAxes, fontsize=11,
                fontweight="bold", color=INK)

    F._caption(fig, (
        "(a) Cross-subject transfer: recentring helps every method and helps "
        "the quantum kernels most, but in the\nreference frame all nine "
        "geometries are indistinguishable. (b) The reference frame needs more "
        "shots to reach\nits own higher ceiling, yet from $10^4$ shots it "
        "exceeds what the sensor frame reaches with unlimited shots."))
    F._save(fig, out, "fig9_transfer_shots")
    return True


# ==========================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None)
    ap.add_argument("--paper", action="store_true",
                    help="drop in-figure titles/notes and use a white canvas")
    args = ap.parse_args(argv)

    F.PAPER = args.paper
    res = Path(args.results)
    out = Path(args.out) if args.out else (
        res / ("figures_paper" if args.paper else "figures"))
    out.mkdir(parents=True, exist_ok=True)

    built = {
        "fig5_circuits": fig_circuits(out),
        "fig6_reference": fig_reference(res, out),
        "fig7_frame": fig_frame(res, out),
        "fig8_twin": fig_twin(res, out),
        "fig9_transfer_shots": fig_transfer_shots(res, out),
    }
    for k, v in built.items():
        print(f"  {'ok  ' if v else 'skip'} {k}")
    print(f"-> {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
