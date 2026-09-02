"""Publication figures from the benchmark and concentration results.

Outputs 300-dpi PNG plus vector PDF for each figure, into ``results/figures/``.

Design notes (so the choices are reviewable, not taste):
- Palette is a validated categorical set; adjacent-pair CVD Delta E >= 8 and
  normal-vision Delta E >= 15 in both the 6-slot (lines) and 3-slot (bars) cuts.
- Three slots sit below 3:1 contrast on the light surface, so every figure
  ships visible direct labels as relief -- identity is never colour-alone.
- Line style carries a second, colour-independent channel (kernel family), so
  the figures survive greyscale printing and colour-vision deficiency.
- One y-axis per panel. No dual axes anywhere.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical slots (light mode, surface #fcfcfb).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
)
INK, INK_2, INK_MUTED = "#0b0b0b", "#52514e", "#8a8880"
SURFACE, GRID = "#fcfcfb", "#e4e3de"

# Paper mode: the LaTeX \caption supplies the title and the explanatory note,
# so the figure must not repeat them, and the canvas must be plain white to sit
# on the page without a visible tint block. Toggled by --paper.
PAPER = False

GROUP_COLOR = {"classical": BLUE, "quantum": ORANGE, "control": AQUA}

# Kernel -> (colour, linestyle). Linestyle encodes the family: dotted = raw
# overlap, solid = bandwidth-corrected, dashed = circuit embedding.
KERNEL_STYLE = {
    "HS-overlap":    (MAGENTA, ":"),
    "Fidelity":      (YELLOW,  ":"),
    "HS-RBF":        (BLUE,    "-"),
    "Bures-RBF":     (AQUA,    "-"),
    "IQP-entangled": (ORANGE,  "--"),
    "IQP-product":   (GREEN,   "--"),
}


def _surface() -> str:
    """Actual canvas colour: white for the manuscript, tinted for standalone."""
    return "#ffffff" if PAPER else SURFACE


def _style() -> None:
    surface = _surface()
    plt.rcParams.update({
        "figure.facecolor": surface,
        "axes.facecolor": surface,
        "savefig.facecolor": surface,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "figure.dpi": 120,
    })


def _despine(ax, keep=("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def _caption(fig, text: str) -> None:
    """Place a caption below the figure, clear of the x-axis label.

    Figure coordinates with a negative y; ``bbox_inches="tight"`` at save time
    expands the canvas to include it, so it can never collide with the axes.
    Suppressed in paper mode, where LaTeX's \\caption does this job.
    """
    if PAPER:
        return
    fig.text(0.005, -0.015, text, fontsize=7.6, color=INK_MUTED,
             ha="left", va="top", linespacing=1.45)


def _title(ax, text: str) -> None:
    """Figure-embedded title. Suppressed in paper mode (LaTeX supplies it)."""
    if PAPER:
        return
    ax.set_title(text, color=INK, loc="left", pad=12)


def _place_end_labels(ax, entries) -> None:
    """Direct line-end labels, pushed apart so they never overlap.

    ``entries`` is a list of (text, x, y, colour) in data coordinates. Labels
    are laid out in axes space with a minimum vertical separation; any label
    that had to move gets a thin leader line back to its data point.
    """
    if not entries:
        return
    inv = ax.transAxes.inverted()
    placed = []
    for text, x, y, color in entries:
        fx, fy = inv.transform(ax.transData.transform((x, y)))
        placed.append([text, float(fx), float(fy), float(fy), color])
    placed.sort(key=lambda e: e[2])

    min_gap = 0.055
    for i in range(1, len(placed)):
        if placed[i][2] - placed[i - 1][2] < min_gap:
            placed[i][2] = placed[i - 1][2] + min_gap
    overflow = placed[-1][2] - 0.98
    if overflow > 0:
        for e in placed:
            e[2] -= overflow

    for text, fx, fy, y_true, color in placed:
        ax.annotate(
            text, xy=(fx, fy), xycoords="axes fraction",
            xytext=(9, 0), textcoords="offset points",
            va="center", ha="left", fontsize=8.5, color=INK,
            fontweight="bold", annotation_clip=False, zorder=6,
        )
        if abs(y_true - fy) > 0.005:
            ax.annotate(
                "", xy=(fx + 0.012, fy), xycoords="axes fraction",
                xytext=(fx, y_true), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-", color=color,
                                linewidth=0.9, alpha=0.75),
                annotation_clip=False, zorder=4,
            )


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


# --------------------------------------------------------------------------
# Figure 1 - kernel concentration vs register size
# --------------------------------------------------------------------------

def fig_concentration(results: Path, out: Path) -> bool:
    src = results / "concentration_summary.csv"
    if not src.exists():
        print("  [skip] fig1: concentration_summary.csv not found")
        return False
    df = pd.read_csv(src)

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.set_yscale("log")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)

    end_labels: list[tuple[str, float, float, str]] = []
    for kern, (color, ls) in KERNEL_STYLE.items():
        g = df[df["kernel"] == kern].sort_values("n_qubits")
        if g.empty:
            continue
        x, y = g["n_qubits"].to_numpy(), g["variance"].to_numpy()
        ax.plot(x, y, color=color, linestyle=ls, linewidth=2.0,
                marker="o", markersize=5, markerfacecolor=color,
                markeredgecolor=_surface(), markeredgewidth=1.4, zorder=3)
        # Direct labels are the relief required by the contrast warning:
        # identity is never carried by colour alone.
        end_labels.append((kern, float(x[-1]), float(y[-1]), color))

    ax.set_xlabel(
        "Register size (qubits)      —      4 / 8 / 16 / 32 / 64 EEG channels")
    ax.set_ylabel("Kernel variance  (off-diagonal Gram)")
    _title(ax, "Quantum kernels concentrate on EEG — entanglement accelerates it")
    ax.set_xticks([2, 3, 4, 5, 6])
    ax.set_xlim(1.9, 6.35)
    _despine(ax)
    _place_end_labels(ax, end_labels)

    n = int(df["n_subjects"].max()) if "n_subjects" in df else 0
    _caption(fig, (
        f"PhysioNet EEGMMIDB, n = {n} subjects. Lower is worse: a kernel whose "
        "variance collapses cannot separate trials.\n"
        "Line style marks the family — dotted = raw overlap, "
        "solid = bandwidth-corrected, dashed = circuit embedding."
    ))
    _save(fig, out, "fig1_concentration")
    return True


# --------------------------------------------------------------------------
# Figure 2 - benchmark accuracy by pipeline
# --------------------------------------------------------------------------

def fig_benchmark(results: Path, out: Path, tag: str = "motor8_q4") -> bool:
    summ = results / f"summary_{tag}.csv"
    raw = results / f"raw_folds_{tag}.csv"
    if not summ.exists():
        print(f"  [skip] fig2: summary_{tag}.csv not found")
        return False
    s = pd.read_csv(summ).sort_values("acc_mean")

    per_subj = None
    if raw.exists():
        r = pd.read_csv(raw)
        per_subj = r.groupby(["pipeline", "subject"])["accuracy"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(7.4, 0.38 * len(s) + 1.9))
    ypos = np.arange(len(s))
    colors = [GROUP_COLOR.get(g, INK_MUTED) for g in s["group"]]

    ax.axvline(0.5, color=INK_MUTED, linewidth=1.0, linestyle="--", zorder=1)
    ax.barh(ypos, s["acc_mean"], height=0.62, color=colors, zorder=2,
            edgecolor=_surface(), linewidth=2.0)

    # Per-subject spread: shows the heterogeneity that mean rankings hide.
    if per_subj is not None:
        rng = np.random.default_rng(0)
        for i, name in enumerate(s["pipeline"]):
            v = per_subj.loc[per_subj["pipeline"] == name, "accuracy"].to_numpy()
            if len(v):
                ax.scatter(v, i + rng.uniform(-0.16, 0.16, size=len(v)),
                           s=9, color=INK, alpha=0.28, linewidths=0, zorder=4)

    # Value labels live in a gutter to the right of every mark, so they never
    # sit inside the per-subject dot cloud.
    data_max = float(s["acc_mean"].max())
    if per_subj is not None and len(per_subj):
        data_max = max(data_max, float(per_subj["accuracy"].max()))
    label_x = data_max + 0.015
    xmax = label_x + 0.055

    for i, v in enumerate(s["acc_mean"]):
        ax.annotate(f"{v:.3f}", xy=(label_x, i), va="center", ha="left",
                    fontsize=8.2, color=INK, fontweight="bold", zorder=5)

    ax.set_yticks(ypos)
    ax.set_yticklabels(s["pipeline"], fontsize=8.4, color=INK)
    ax.set_xlabel("Within-subject accuracy  (nested CV, mean over subjects)")
    ax.set_xlim(0.33, xmax)
    ax.set_ylim(-0.8, len(s) - 0.2)
    # Title states what the data shows, rather than asserting a conclusion the
    # run might not support.
    best_q = s[s["group"] == "quantum"]["acc_mean"].max()
    best_c = s[s["group"] == "classical"]["acc_mean"].max()
    if np.isnan(best_q) or np.isnan(best_c):
        title = "Within-subject decoding accuracy by pipeline"
    elif best_q > best_c:
        title = (f"Best quantum kernel leads the classical baselines "
                 f"({best_q:.3f} vs {best_c:.3f})")
    else:
        title = (f"Tuned classical baselines lead the quantum kernels "
                 f"({best_c:.3f} vs {best_q:.3f})")
    _title(ax, title)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    _despine(ax)

    handles = [
        plt.Line2D([], [], marker="s", linestyle="", markersize=8,
                   markerfacecolor=GROUP_COLOR[g], markeredgecolor=_surface(),
                   label=g)
        for g in ("classical", "quantum", "control")
    ]
    handles.append(plt.Line2D([], [], marker="o", linestyle="", markersize=4,
                              color=INK, alpha=0.35, label="per-subject mean"))
    ax.legend(handles=handles, loc="lower right", fontsize=8, ncol=2)

    n = int(s["n_subjects"].max())
    _caption(fig, (
        f"PhysioNet EEGMMIDB left/right motor imagery, n = {n} subjects. "
        "Dashed line = chance (0.50).\n"
        "Every pipeline, classical ones included, was tuned by the same "
        "inner 4-fold GridSearchCV."
    ))
    _save(fig, out, "fig2_benchmark")
    return True


# --------------------------------------------------------------------------
# Figure 3 - paired per-subject differences vs the reference pipeline
# --------------------------------------------------------------------------

def fig_paired(results: Path, out: Path, tag: str = "motor8_q4",
               reference: str = "classical/TS+LR") -> bool:
    raw = results / f"raw_folds_{tag}.csv"
    if not raw.exists():
        print(f"  [skip] fig3: raw_folds_{tag}.csv not found")
        return False
    r = pd.read_csv(raw)
    per = r.groupby(["pipeline", "subject"])["accuracy"].mean().unstack("pipeline")
    if reference not in per.columns:
        print(f"  [skip] fig3: reference {reference} absent")
        return False

    groups = r.drop_duplicates("pipeline").set_index("pipeline")["group"].to_dict()
    others = [c for c in per.columns if c != reference]
    deltas = {c: (per[c] - per[reference]).dropna().to_numpy() for c in others}
    order = sorted(others, key=lambda c: deltas[c].mean())

    fig, ax = plt.subplots(figsize=(7.4, 0.38 * len(order) + 1.9))
    ax.axvline(0.0, color=INK_2, linewidth=1.2, zorder=2)
    rng = np.random.default_rng(0)

    for i, name in enumerate(order):
        d = deltas[name]
        col = GROUP_COLOR.get(groups.get(name, ""), INK_MUTED)
        ax.scatter(d, i + rng.uniform(-0.17, 0.17, size=len(d)), s=14,
                   color=col, alpha=0.45, linewidths=0, zorder=3)
        m = float(d.mean())
        ax.scatter([m], [i], s=64, marker="D", color=col,
                   edgecolor=_surface(), linewidth=1.6, zorder=5)
        ax.annotate(f"{m:+.3f}", xy=(m, i), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=7.8,
                    color=INK, fontweight="bold", zorder=6)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8.4, color=INK)
    ax.set_xlabel(f"Accuracy difference vs {reference}   (per subject)")
    ax.set_ylim(-0.8, len(order) - 0.2)
    q_deltas = [deltas[c].mean() for c in order if groups.get(c) == "quantum"]
    if q_deltas and max(q_deltas) > 0:
        sub = f"best quantum kernel {max(q_deltas):+.3f} vs baseline"
    elif q_deltas:
        sub = f"every quantum kernel below baseline (best {max(q_deltas):+.3f})"
    else:
        sub = "paired per-subject differences"
    _title(ax, f"Paired differences vs {reference} — {sub}")
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    _despine(ax)

    handles = [
        plt.Line2D([], [], marker="D", linestyle="", markersize=7,
                   markerfacecolor=GROUP_COLOR[g], markeredgecolor=_surface(),
                   label=g)
        for g in ("classical", "quantum", "control")
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    _caption(fig, (
        "Small dots = individual subjects; diamond = mean difference. "
        "Points left of zero are worse than the baseline.\n"
        "See tests_vs_*.csv for Wilcoxon signed-rank p-values with "
        "Holm correction."
    ))
    _save(fig, out, "fig3_paired_differences")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate publication figures")
    ap.add_argument("--results", type=str, default="results")
    ap.add_argument("--tag", type=str, default="motor8_q4")
    ap.add_argument("--reference", type=str, default="classical/TS+LR")
    ap.add_argument("--paper", action="store_true",
                    help="omit in-figure titles/captions and use a white "
                         "canvas, for embedding in the LaTeX manuscript where "
                         "the LaTeX caption supplies that text")
    args = ap.parse_args(argv)

    global PAPER
    PAPER = args.paper
    _style()
    results = Path(args.results)
    out = results / ("figures_paper" if PAPER else "figures")
    print(f"Writing figures to {out.resolve()}")
    made = [
        fig_concentration(results, out),
        fig_benchmark(results, out, args.tag),
        fig_paired(results, out, args.tag, args.reference),
    ]
    print(f"Done: {sum(made)}/3 figures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
