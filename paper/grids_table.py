"""The hyperparameter grids and fixed settings, read from the code that ran.

A reviewer asked for every grid so that "the same inner search budget" can be
checked. Typing them into the manuscript would let the table drift from the
code, so this module imports the grids the benchmark actually passes to
GridSearchCV and reads the fixed settings off the pipeline objects. If a grid
changes in src/qeeg, this table changes with it.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _fmt(values) -> str:
    out = []
    for v in values:
        if isinstance(v, str):
            out.append(r"\texttt{" + v + "}")
        elif isinstance(v, float) and v.is_integer() and v >= 1:
            out.append(f"{v:g}")
        else:
            out.append(f"{v:g}")
    return r"$\{$" + ", ".join(out) + r"$\}$"


LABEL = {
    "clf__C": "$C$",
    "C": "$C$",
    "clf__gamma": r"RBF $\gamma$",
    "clf__gamma_mult": r"bandwidth multiplier $\times\gamma_0$",
    "clf__scale": "angle scale",
    "csp__nfilter": "CSP filters",
    "perband__base__csp__nfilter": "CSP filters per band",
}


def _row_groups():
    from qeeg.benchmark import make_grids
    from qeeg.filterbank import make_filterbank_grids
    from qeeg.transfer import C_GRID, GAMMA_MULT_GRID

    within = make_grids()
    fb = make_filterbank_grids()

    def describe(grid: dict) -> str:
        if not grid:
            return "none (no tuned hyperparameter)"
        return "; ".join(f"{LABEL.get(k, k)} {_fmt(v)}" for k, v in grid.items())

    groups: dict[str, list[str]] = {}
    for name, grid in within.items():
        groups.setdefault(describe(grid), []).append(name)
    for name, grid in fb.items():
        groups.setdefault(describe(grid), []).append(name)
    transfer = (f"$C$ {_fmt(C_GRID)}; bandwidth multiplier $\\times\\gamma_0$ "
                f"{_fmt(GAMMA_MULT_GRID)} for distance kernels")
    return groups, transfer


def _fixed_settings() -> dict:
    """Read the fixed settings off the objects the benchmark builds."""
    import inspect

    from qeeg import crosssession, transfer
    from qeeg.benchmark import evaluate_subject
    from qeeg.pipelines import make_pipelines

    pipes = make_pipelines(suite="core")
    circ = pipes["quantum/IQP-kernel-SVM"]
    return {
        "pca": circ.named_steps["pca"].n_components,
        "layers": circ.named_steps["clf"].n_layers,
        "angle_hi": circ.named_steps["angle"].feature_range[1],
        "lr_iter": pipes["classical/TS+LR"].named_steps["clf"].max_iter,
        "cov": pipes["classical/MDM"].named_steps["cov"].estimator,
        "inner_within": inspect.signature(evaluate_subject).parameters["inner_splits"].default,
        "inner_transfer": inspect.signature(transfer._fit_eval_quantum).parameters["inner_splits"].default,
        "inner_session": crosssession.INNER_SPLITS,
    }


def table_grids(out: list[str]) -> bool:
    try:
        groups, transfer = _row_groups()
        fx = _fixed_settings()
    except Exception as exc:  # noqa: BLE001 - never break the whole table build
        print(f"  ! grids table skipped: {type(exc).__name__}: {exc}")
        return False
    import numpy as np
    angle = r"\pi" if np.isclose(fx["angle_hi"], np.pi) else f"{fx['angle_hi']:g}"

    def short(names):
        names = sorted(n.split("/")[-1] for n in names)
        return ", ".join(n.replace("_", r"\_") for n in names)

    out.append(r"""
%% --------------------------------------------------- Table: grids
\begin{table}[htbp]
\caption{\label{tab:grids}Hyperparameter grids and fixed settings. Every grid is
searched by the same inner cross-validation (""" + f"{fx['inner_within']}" + r"""-fold
within subject; """ + f"{fx['inner_transfer']}" + r"""-fold subject-grouped for
cross-subject transfer; """ + f"{fx['inner_session']}" + r"""-fold stratified
within the training session for cross-session transfer). The grids and
settings are read from the code that produced the results. $\gamma_0$ is the
median-heuristic bandwidth, computed on the training split only.}""")
    out.append(r"""
\begin{tabular}{@{}p{0.40\textwidth}p{0.55\textwidth}@{}}
\hline
Pipelines & Searched grid \\
\hline""")
    for grid, names in sorted(groups.items(), key=lambda kv: kv[0]):
        out.append(f"{short(names)} & {grid} \\\\")
    out.append(r"\hline")
    out.append(r"Cross-subject and cross-session quantum kernels & " + transfer + r" \\")
    out.append(r"""\hline
\multicolumn{2}{@{}p{0.95\textwidth}@{}}{\emph{Fixed in every pipeline:} """
               + fx["cov"].upper() + r""" shrinkage covariance; LDA with
Ledoit--Wolf shrinkage; tangent space at the Riemannian mean with feature
standardisation; circuit kernels and dimension-matched controls on a PCA
reduction to """ + f"{fx['pca']}" + r""" components scaled to $[0,""" + angle
               + r"""]$, """ + f"{fx['layers']}" + r""" circuit layers; SVMs on
precomputed kernels projected to the positive semi-definite cone; logistic
regression with up to """ + f"{fx['lr_iter']}" + r""" iterations.} \\
\hline
\end{tabular}
\end{table}
""")
    return True
