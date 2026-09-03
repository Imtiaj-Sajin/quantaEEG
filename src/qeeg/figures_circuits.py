"""Circuit figure, rendered by Qiskit from the verified circuit construction.

This replaces a hand-drawn matplotlib schematic. The circuits drawn here are
built by `circuits_qiskit.build_feature_map`, which is checked gate for gate
against the PennyLane circuit the experiments actually executed: the check
compares full Gram matrices and is run before drawing. If the two ever diverge,
this script raises instead of producing a figure that misrepresents the method.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import figures as F  # noqa: E402
from .circuits_qiskit import build_feature_map, verify_against_pennylane  # noqa: E402

PANELS = [
    (dict(entangle=True, entangler="iqp"), "IQP feature map",
     "data-dependent $ZZ$ coupling"),
    (dict(entangle=True, entangler="ring"), "Ring CNOT feature map",
     "fixed entanglers"),
    (dict(entangle=False, entangler="iqp"), "Entanglement ablation",
     "identical circuit, entanglers deleted"),
]


def fig_circuits(out: Path, n_qubits: int = 3, n_layers: int = 1) -> bool:
    """Three-panel circuit figure at `n_qubits` qubits, one layer shown."""
    report = verify_against_pennylane(n_qubits=n_qubits, n_layers=2)
    worst = max(report.values())
    print(f"  circuit check: max |Gram difference| = {worst:.2e} "
          f"across {len(report)} configurations")

    F._style()

    # Qiskit's mpl drawer stretches a circuit to fill whatever axes it is given,
    # so passing three axes of different widths renders the same gate at three
    # different sizes. Instead each circuit is drawn to its own figure at its
    # natural scale, rasterised, and then placed with the aspect preserved. The
    # gates are then identical across panels and only the circuits differ.
    import io

    import numpy as np

    panels = []
    for kw, title, sub in PANELS:
        qc = build_feature_map(None, n_qubits=n_qubits, n_layers=n_layers,
                               parameterised=True, **kw)
        sub_fig = qc.draw("mpl", style={"backgroundcolor": F._surface()},
                          fold=-1, initial_state=False, scale=1.0)
        buf = io.BytesIO()
        sub_fig.savefig(buf, format="png", dpi=400, bbox_inches="tight",
                        facecolor=F._surface())
        plt.close(sub_fig)
        buf.seek(0)
        panels.append((plt.imread(buf), title, sub))

    widths = [im.shape[1] for im, _, _ in panels]
    heights = [im.shape[0] for im, _, _ in panels]
    scale = max(heights)

    fig, axes = plt.subplots(
        1, 3, figsize=(13.0, 3.1),
        gridspec_kw={"width_ratios": widths, "wspace": 0.08},
    )
    for ax, (im, title, sub) in zip(axes, panels):
        # Pad every panel to the tallest so all three share one vertical scale.
        pad = scale - im.shape[0]
        if pad:
            filler = np.ones((pad, im.shape[1], im.shape[2]), dtype=im.dtype)
            im = np.vstack([im, filler])
        ax.imshow(im, interpolation="antialiased")
        ax.axis("off")
        ax.set_title(title, fontsize=10, fontweight="bold", color=F.INK,
                     loc="left", pad=8)
        ax.text(0.0, -0.04, sub, transform=ax.transAxes, fontsize=8.2,
                color=F.INK_MUTED, ha="left", va="top")

    F._caption(fig, (
        "Circuit feature maps at three qubits, one layer shown, rendered with "
        "Qiskit from the same gate sequence the\n"
        "experiments execute. Rotation angles carry the PCA-reduced "
        "tangent-space features; the kernel is the state overlap."
    ))
    F._save(fig, out, "fig5_circuits")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Qiskit circuit figure")
    ap.add_argument("--results", default="results")
    ap.add_argument("--paper", action="store_true")
    args = ap.parse_args(argv)

    F.PAPER = args.paper
    out = Path(args.results) / ("figures_paper" if args.paper else "figures")
    print(f"Writing circuit figure to {out.resolve()}")
    fig_circuits(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
