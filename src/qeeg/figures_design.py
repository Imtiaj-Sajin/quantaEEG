"""Figure: the study design, and where the control sits.

Why this figure exists
----------------------
The contribution of this work is not an accuracy number, it is a comparison
structure: a quantum kernel is only interesting if something classical, given
every advantage it has, still cannot match it. That structure is hard to carry
in prose because it is a conjunction of five things held fixed and one thing
varied. A reader who sees the metric-matched twin sitting next to the quantum
kernel, fed by the same covariances, in the same frame, tuned on the same grid,
inside the same support vector machine, understands the argument before
reading a word of it.

The figure is drawn, not measured, with one exception: subject counts come
from the result files when those exist, so the figure cannot claim a cohort
the repository does not contain.

    PYTHONPATH=src python -m qeeg.figures_design --paper
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .figures import (AQUA, BLUE, GRID, INK, INK_2, INK_MUTED, MAGENTA,
                      ORANGE, _style, _surface)
from . import figures as F

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Fallbacks, used only when the result file is absent. They are the cohort
# sizes the protocol specifies, not results.
PLANNED = {"physionet": 104, "bci2a": 9, "cho2017": 52}


def _n_subjects(res: Path, name: str, patterns: list[str]) -> int:
    """Subjects actually present in the results, else the planned cohort."""
    for pat in patterns:
        hits = sorted(res.glob(pat))
        if not hits:
            continue
        try:
            n = pd.concat([pd.read_csv(h, usecols=["subject"]) for h in hits],
                          ignore_index=True).subject.nunique()
        except (ValueError, KeyError, pd.errors.EmptyDataError):
            continue
        if n:
            return int(n)
    return PLANNED[name]


def _box(ax, x, y, w, h, text, face, edge, *, fontsize=8.2, weight="normal",
         color=INK, radius=0.012, lw=1.0, align="center"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=color, zorder=3, linespacing=1.45,
            fontweight=weight, multialignment=align)


def _arrow(ax, xy_from, xy_to, color=INK_MUTED, lw=1.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        xy_from, xy_to, arrowstyle=style, mutation_scale=9,
        linewidth=lw, color=color, zorder=1,
        shrinkA=1.5, shrinkB=1.5))


def _tint(hexcolor: str, alpha: float) -> tuple:
    """Flat blend towards white: printers handle a solid fill better than
    transparency, and the PDF stays one page-sized object."""
    h = hexcolor.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return tuple(1 - alpha * (1 - c) for c in (r, g, b))


def fig_design(out: Path, res: Path) -> bool:
    _style()
    n_phys = _n_subjects(res, "physionet", ["summary_motor8.csv",
                                            "raw_folds_motor8.csv"])
    n_bci = _n_subjects(res, "bci2a", ["summary_bci2a_motor8.csv",
                                       "raw_folds_bci2a_motor8.csv"])
    n_cho = _n_subjects(res, "cho2017", ["raw_folds_cho2017*.csv",
                                         "calib_folds_cho2017.csv",
                                         "calib_folds_calib_cho_b0?.csv"])

    fig, ax = plt.subplots(figsize=(7.1, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(_surface())

    def band(y, label):
        """Section label on its own line, with a rule under it, so nothing
        centred can ever collide with it."""
        ax.text(0.012, y, label, fontsize=8.6, color=INK_MUTED,
                ha="left", va="center", fontweight="bold")
        ax.plot([0.012, 0.988], [y - 0.017, y - 0.017], color=GRID,
                linewidth=0.8, zorder=0)

    # ---------------------------------------------------------------- data
    band(0.982, "1  DATA")
    data = [f"PhysioNet EEGMMIDB\n{n_phys} subjects, 45 trials",
            f"BCI IV-2a\n{n_bci} subjects, 288 trials\ntwo sessions",
            f"Cho2017\n{n_cho} subjects, 200 trials"]
    for i, t in enumerate(data):
        _box(ax, 0.040 + i * 0.320, 0.868, 0.280, 0.078, t,
             _tint(BLUE, 0.10), _tint(BLUE, 0.45), fontsize=7.8)
        _arrow(ax, (0.180 + i * 0.320, 0.868), (0.5, 0.836))

    _box(ax, 0.130, 0.776, 0.74, 0.056,
         "the same 8 sensorimotor channels, 8 to 30 Hz, one-second window\n"
         r"spatial covariance $C$ (8$\times$8, symmetric positive definite)",
         _tint(INK_MUTED, 0.08), _tint(INK_MUTED, 0.35), fontsize=7.9)

    # ---------------------------------------------------------------- frame
    band(0.745, "2  FRAME")
    ax.text(0.5, 0.708,
            "EEG nuisances (electrode gain, head geometry, source mixing) act "
            r"on $C$ by congruence $C \rightarrow ACA^{\top}$",
            ha="center", va="center", fontsize=7.9, color=INK_2)
    _arrow(ax, (0.5, 0.776), (0.5, 0.722))

    _box(ax, 0.040, 0.578, 0.42, 0.098,
         "sensor frame\n"
         r"$\rho = C\,/\,\mathrm{tr}\,C$" "\n"
         r"invariant under $O C O^{\top}$ only",
         _tint(ORANGE, 0.10), _tint(ORANGE, 0.55), fontsize=8.0)
    _box(ax, 0.540, 0.578, 0.42, 0.098,
         "reference frame\n"
         r"$\tilde{\rho} = WCW / \mathrm{tr}(WCW)$, $W = M^{-1/2}$" "\n"
         r"exactly invariant under $A C A^{\top}$",
         _tint(AQUA, 0.10), _tint(AQUA, 0.55), fontsize=8.0)
    _arrow(ax, (0.44, 0.694), (0.250, 0.680))
    _arrow(ax, (0.56, 0.694), (0.750, 0.680))
    ax.text(0.5, 0.552,
            "every suite is run in both frames, changing nothing else",
            ha="center", va="center", fontsize=7.6, color=INK_MUTED,
            style="italic")

    # ------------------------------------------------------------ pipelines
    band(0.523, "3  PIPELINES")
    # Four equal columns: 0.040 to 0.960 with three 0.018 gutters.
    cw, gap, x0 = 0.2165, 0.018, 0.040
    ty, th = 0.438, 0.052        # title box
    by, bh = 0.292, 0.138        # body box
    cols = [
        ("classical\nbaselines", BLUE, "CSP + LDA\nTS + LR\nMDM\nFBCSP"),
        ("metric-matched\ntwin", AQUA,
         "Riemannian\nkernel SVM\n(log-Euclidean\nas a second)"),
        ("quantum\nkernels", ORANGE,
         "HS overlap\nFidelity\nHS-RBF\nBures-RBF\nQRE-RBF"),
        ("circuit kernels\n+ ablation", MAGENTA,
         "IQP entangled\nIQP entanglers\ndeleted\nPCA-matched"),
    ]
    for i, (title, colour, body) in enumerate(cols):
        x = x0 + i * (cw + gap)
        _box(ax, x, ty, cw, th, title, _tint(colour, 0.22),
             _tint(colour, 0.6), fontsize=8.0, weight="bold")
        _box(ax, x, by, cw, bh, body, _tint(colour, 0.06),
             _tint(colour, 0.40), fontsize=7.6)

    # The control, drawn as what it is: one frame around the only two columns
    # that differ in a single respect.
    x_twin = x0 + 1 * (cw + gap)
    x_quant_end = x0 + 2 * (cw + gap) + cw
    pad = 0.008
    ax.add_patch(FancyBboxPatch(
        (x_twin - pad, by - pad), (x_quant_end - x_twin) + 2 * pad,
        (ty + th) - by + 2 * pad,
        boxstyle="round,pad=0,rounding_size=0.012",
        linewidth=1.3, edgecolor=INK, facecolor="none",
        linestyle=(0, (4, 2.5)), zorder=4))
    ax.text((x_twin + x_quant_end) / 2, 0.250,
            "same covariances, same SVM, same tuning budget, same frame:\n"
            "only the metric differs",
            ha="center", va="center", fontsize=7.8, color=INK,
            fontweight="bold", linespacing=1.4)

    # ----------------------------------------------------------- evaluation
    band(0.222, "4  EVALUATION")
    _box(ax, 0.040, 0.132, 0.92, 0.056,
         "nested cross-validation: 5 folds $\\times$ 3 repeats outer, "
         "4-fold inner grid search,\n"
         "the identical search budget for every pipeline, quantum and classical",
         _tint(INK_MUTED, 0.08), _tint(INK_MUTED, 0.35), fontsize=7.9)
    _box(ax, 0.040, 0.062, 0.92, 0.056,
         "paired per-subject Wilcoxon signed-rank, Holm corrected within each "
         "pre-specified family,\n"
         "and two one-sided tests for equivalence against the twin",
         _tint(INK_MUTED, 0.08), _tint(INK_MUTED, 0.35), fontsize=7.9)
    ax.text(0.5, 0.030,
            "settings: within-subject  |  cross-subject (leave one out)  |  "
            "cross-session  |  few-trial calibration  |  registers 3 to 6 qubits",
            ha="center", va="center", fontsize=7.6, color=INK_2)

    out.mkdir(parents=True, exist_ok=True)
    dest = out / "fig_design.pdf"
    fig.savefig(dest, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(dest.with_suffix(".png"), dpi=200, bbox_inches="tight",
                pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {dest}  (PhysioNet n={n_phys}, IV-2a n={n_bci}, "
          f"Cho2017 n={n_cho})")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--results", default="results")
    ap.add_argument("--paper", action="store_true",
                    help="white canvas and no titles, for the manuscript")
    args = ap.parse_args(argv)
    F.PAPER = args.paper
    fig_design(Path(args.out), Path(args.results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
