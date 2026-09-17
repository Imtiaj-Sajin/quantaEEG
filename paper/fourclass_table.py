"""Four-class IV-2a: the table and macros for section 3.x.

The referee asked whether the invariance argument is specific to a single
binary contrast. It is not, and four classes make the case more sharply than
two: chance is 0.25, the sensor-frame kernels sit barely above it, and
recentring moves them by a quarter of the accuracy range.

Every number here is computed from the merged four-class result file. No figure in
the manuscript is typed by hand.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

PAIRS = [
    ("quantum/Fidelity-SVM", "quantum/Fidelity-ref-SVM", "Fidelity"),
    ("quantum/HS-overlap-SVM", "quantum/HS-overlap-ref-SVM", "HS overlap"),
    ("quantum/HS-RBF-SVM", "quantum/HS-RBF-ref-SVM", "HS-RBF"),
    ("quantum/Bures-RBF-SVM", "quantum/Bures-RBF-ref-SVM", "Bures-RBF"),
    ("quantum/QRE-RBF-SVM", "quantum/QRE-RBF-ref-SVM", "QRE-RBF"),
]
TWIN = "control/riemann-kernel-SVM"
SECOND = "control/logeuclid-kernel-SVM"
CLASSICAL = ["classical/TS+LR", "classical/CSP+LDA", "classical/MDM"]
CHANCE = 0.25


def load(res: Path):
    """Read the merged four-class run.

    This used to glob the per-subject batch files. Those are build
    intermediates, losslessly absorbed into the merged file, and deleting them
    silently dropped this whole table and its section's macros from the
    manuscript. Read the canonical file, and fall back to the batches only if
    the merge has not been done yet.
    """
    merged = res / "raw_folds_refstate_bci2a4_motor8_q4.csv"
    if merged.exists():
        d = pd.read_csv(merged)
    else:
        files = sorted(res.glob("raw_folds_bci4_b0?.csv"))
        if not files:
            return None
        d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return d.groupby(["pipeline", "subject"]).accuracy.mean().unstack("pipeline")


def _w(diff):
    return float(diff.mean()), float(wilcoxon(diff).pvalue), int((diff > 0).sum())


def table_fourclass(per, fmt_p, out: list[str]) -> bool:
    if per is None:
        return False
    n = len(per)
    out.append(r"""
%% --------------------------------------------------- Table: four-class IV-2a
\begin{table}[htbp]
\caption{\label{tab:fourclass}Four-class motor imagery on IV-2a (left hand,
right hand, feet, tongue; chance $=0.25$), $n=""" + f"{n}" + r"""$ subjects,
extended suite, protocol identical to the binary runs. $\Delta$ is the gain
from the reference frame, paired by subject and tested by Wilcoxon
signed-rank. Recentring is worth far more here than in the binary task, and
the classical twin remains at least level with every quantum kernel
afterwards, so neither conclusion of this paper depends on the task having
been a single binary contrast.}
\begin{tabular}{@{}lcccc@{}}
\hline
Kernel & Sensor & Reference & $\Delta$ & $p$ \\
\hline""")
    for a, b, lab in PAIRS:
        if a not in per or b not in per:
            continue
        d, p, better = _w((per[b] - per[a]).dropna())
        out.append(f"{lab} & {per[a].mean():.3f} & {per[b].mean():.3f} & "
                   f"$+{d:.4f}$ & {fmt_p(p)} \\\\")
    out.append(r"\hline")
    for name, lab in ((TWIN, "Riemannian kernel (twin)"),
                      (SECOND, "Log-Euclidean kernel")):
        if name in per:
            out.append(f"{lab} & & {per[name].mean():.3f} & & \\\\")
    bc = max((c for c in CLASSICAL if c in per), key=lambda c: per[c].mean())
    out.append(f"{bc.split('/')[1].replace('+', chr(92) + ',+' + chr(92) + ',')} "
               f"(best classical) & & {per[bc].mean():.3f} & & \\\\")
    out.append(r"""\hline
\end{tabular}
\end{table}
""")
    return True


def macros(per, fmt_p_eq, out: list[str]) -> None:
    if per is None:
        return
    frame = [_w((per[b] - per[a]).dropna()) for a, b, _ in PAIRS
             if a in per and b in per]
    twin = [_w((per[b] - per[TWIN]).dropna()) for _, b, _ in PAIRS
            if b in per and TWIN in per]
    bc = max((c for c in CLASSICAL if c in per), key=lambda c: per[c].mean())
    best_q = max((b for _, b, _ in PAIRS if b in per), key=lambda c: per[c].mean())
    defs = {
        "FourClassN": f"{len(per)}",
        "FourClassChance": f"{CHANCE:g}",
        "FourClassSensorMin": f"{min(per[a].mean() for a, _, _ in PAIRS):.3f}",
        "FourClassSensorMax": f"{max(per[a].mean() for a, _, _ in PAIRS):.3f}",
        "FourClassFrameMin": f"{min(d for d, _, _ in frame):+.4f}",
        "FourClassFrameMax": f"{max(d for d, _, _ in frame):+.4f}",
        "FourClassFrameMaxP": fmt_p_eq(max(p for _, p, _ in frame)),
        "FourClassFrameAllBetter": "yes" if all(
            b == len(per) for _, _, b in frame) else "no",
        # Signs matter here: every one of these is negative, i.e. the twin is
        # ahead, and the prose must not be able to say otherwise.
        "FourClassTwinDeltaMin": f"{min(d for d, _, _ in twin):+.4f}",
        "FourClassTwinDeltaMax": f"{max(d for d, _, _ in twin):+.4f}",
        "FourClassTwinMinP": fmt_p_eq(min(p for _, p, _ in twin)),
        "FourClassTwinAhead": "yes" if all(d < 0 for d, _, _ in twin) else "no",
        "FourClassTwinAcc": f"{per[TWIN].mean():.3f}" if TWIN in per else "n/a",
        "FourClassBestQuantum": best_q.split("/")[1].replace("-ref-SVM", ""),
        "FourClassBestQuantumAcc": f"{per[best_q].mean():.3f}",
        "FourClassBestClassical": bc.split("/")[1].replace("+", r"\,+\,"),
        "FourClassBestClassicalAcc": f"{per[bc].mean():.3f}",
    }
    out.append("\n%% ------------------------------ four-class macros\n")
    for k, v in defs.items():
        out.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    out.append("")
