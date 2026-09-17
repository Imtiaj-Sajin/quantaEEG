"""Tables and macros for the few-trial calibration experiment.

Reads results/calib_folds_{cho2017,bci2a}.csv (written by
`python -m qeeg.calibration --merge ...`). Each builder returns False when its
inputs are absent, like every other table module.

The question the experiment answers was fixed before it ran: if a quantum
divergence copes better than the affine-invariant metric with a reference
state estimated from few trials, the quantum-minus-twin difference should be
largest, and positive, at the smallest training size.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATASETS = [("cho2017", "Cho2017"), ("bci2a", "IV-2a")]
TWIN = "control/riemann-kernel-SVM"
REF_KERNELS = ["quantum/Fidelity-ref-SVM", "quantum/HS-overlap-ref-SVM",
               "quantum/HS-RBF-ref-SVM", "quantum/Bures-RBF-ref-SVM",
               "quantum/QRE-RBF-ref-SVM"]
SENSOR_KERNELS = [k.replace("-ref-SVM", "-SVM") for k in REF_KERNELS]
MIN_SUBJECTS = 5


def load(res: Path) -> dict:
    out = {}
    for key, label in DATASETS:
        f = res / f"calib_folds_{key}.csv"
        if f.exists():
            df = pd.read_csv(f)
            per = (df.groupby(["n_train", "pipeline", "subject"])["accuracy"]
                   .mean().unstack("pipeline"))
            out[key] = (label, per)
    return out


def _paired(per_k: pd.DataFrame, a: str, b: str) -> dict:
    d = (per_k[a] - per_k[b]).dropna().to_numpy()
    n = len(d)
    res = {"n": n, "delta": float(d.mean()) if n else np.nan, "p": np.nan,
           "bound": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    if n >= MIN_SUBJECTS and not np.allclose(d, 0):
        res["p"] = float(stats.wilcoxon(d).pvalue)
        se = d.std(ddof=1) / np.sqrt(n)
        half = float(stats.t.ppf(0.95, n - 1) * se)
        res["ci_low"], res["ci_high"] = d.mean() - half, d.mean() + half
        res["bound"] = max(abs(res["ci_low"]), abs(res["ci_high"]))
    return res


def table_calibration(d: dict, fmt_p, out: list[str]) -> bool:
    if not d:
        return False
    out.append(r"""
%% ------------------------------------------------ Table: calibration
\begin{table}[htbp]
\caption{\label{tab:calibration}Few-trial calibration. For each subject a fixed,
balanced test set is held out; each pipeline is tuned and trained on $k$
balanced trials drawn from the rest (ten random draws, averaged per subject),
so the reference state is estimated from those $k$ trials alone. Columns give
the Riemannian twin, tangent-space logistic regression, the best of the five
reference-frame quantum kernels and the best sensor-frame one, the range of the
five quantum-minus-twin differences, the smallest $p$ among them, and the
largest 90\,\% equivalence bound.}
\begin{tabular}{@{}llccccccc@{}}
\hline
Dataset & $k$ & Twin & TS+LR & Quantum & Sensor & Quantum $-$ twin & $\min p$ & Bound \\
\hline""")
    first_block = True
    for key, (label, per) in d.items():
        if not first_block:
            out.append(r"\hline")
        first_block = False
        for i, k in enumerate(sorted(per.index.get_level_values(0).unique())):
            pk = per.loc[k]
            ks = [q for q in REF_KERNELS if q in pk.columns]
            ss = [q for q in SENSOR_KERNELS if q in pk.columns]
            if TWIN not in pk.columns or not ks:
                continue
            st = [_paired(pk, q, TWIN) for q in ks]
            deltas = [s["delta"] for s in st]
            ps = [s["p"] for s in st if not np.isnan(s["p"])]
            bounds = [s["bound"] for s in st if not np.isnan(s["bound"])]
            lead = f"{label} ($n={len(pk)}$)" if i == 0 else ""
            out.append(
                f"{lead} & {int(k)} & {pk[TWIN].mean():.3f} & "
                f"{pk['classical/TS+LR'].mean():.3f} & "
                f"{pk[ks].mean().max():.3f} & "
                f"{(pk[ss].mean().max() if ss else float('nan')):.3f} & "
                f"$[{min(deltas):+.4f}, {max(deltas):+.4f}]$ & "
                f"{fmt_p(min(ps)) if ps else 'n/a'} & "
                f"{(max(bounds) if bounds else float('nan')):.4f} \\\\"
            )
    out.append(r"""\hline
\end{tabular}
\end{table}
""")
    return True


def macros(d: dict, fmt_p_eq, esc, out: list[str]) -> None:
    if not d:
        return
    defs: dict[str, str] = {}
    words = {1: "one", 2: "two", 3: "three"}
    defs["CalibNDatasets"] = words.get(len(d), str(len(d)))
    all_sizes = sorted({int(k) for _, per in d.values()
                        for k in per.index.get_level_values(0).unique()})
    defs["CalibKMin"], defs["CalibKMax"] = f"{all_sizes[0]}", f"{all_sizes[-1]}"
    any_sig_pos, any_sig_neg, worst_bound = [], [], 0.0
    for key, (label, per) in d.items():
        prefix = "CalibCho" if key == "cho2017" else "CalibBci"
        defs[prefix + "N"] = f"{per.index.get_level_values(1).nunique()}"
        for tag, k in (("Small", min(per.index.get_level_values(0))),
                       ("Large", max(per.index.get_level_values(0)))):
            pk = per.loc[k]
            ks = [q for q in REF_KERNELS if q in pk.columns]
            if TWIN not in pk.columns or not ks:
                continue
            st = {q: _paired(pk, q, TWIN) for q in ks}
            deltas = [s["delta"] for s in st.values()]
            ps = [s["p"] for s in st.values() if not np.isnan(s["p"])]
            defs[f"{prefix}{tag}DeltaMin"] = f"{min(deltas):+.4f}"
            defs[f"{prefix}{tag}DeltaMax"] = f"{max(deltas):+.4f}"
            if ps:
                defs[f"{prefix}{tag}MinP"] = fmt_p_eq(min(ps))
            defs[f"{prefix}{tag}TwinAcc"] = f"{pk[TWIN].mean():.3f}"
            defs[f"{prefix}{tag}QuantumAcc"] = f"{pk[ks].mean().max():.3f}"
        for k in per.index.get_level_values(0).unique():
            pk = per.loc[k]
            for q in REF_KERNELS:
                if q in pk.columns and TWIN in pk.columns:
                    s = _paired(pk, q, TWIN)
                    if not np.isnan(s["p"]) and s["p"] < 0.05:
                        (any_sig_pos if s["delta"] > 0 else any_sig_neg).append(
                            f"{q.split('/')[1].replace('-ref-SVM', '')} at $k={int(k)}$ on {label}")
                    if not np.isnan(s["bound"]):
                        worst_bound = max(worst_bound, s["bound"])
    defs["CalibNSigQuantum"] = f"{len(any_sig_pos)}"
    defs["CalibNSigTwin"] = f"{len(any_sig_neg)}"
    defs["CalibWorstBound"] = f"{worst_bound:.3f}"

    # The same count under correction. "11 in favour, 0 against" was an
    # uncorrected tally over every kernel, size and dataset, and a referee
    # asked which correction applied. Two families are reported: the five
    # kernels at one dataset and size (the family every other twin comparison
    # in the paper uses) and all comparisons pooled into one.
    from reference_tables import holm
    rows = []                          # (delta, raw p, per-family Holm p, bound)
    for key, (label, per) in d.items():
        for k in per.index.get_level_values(0).unique():
            pk = per.loc[k]
            ks = [q for q in REF_KERNELS if q in pk.columns]
            if TWIN not in pk.columns or not ks:
                continue
            st = [_paired(pk, q, TWIN) for q in ks]
            st = [x for x in st if not np.isnan(x["p"])]
            for x, a in zip(st, holm([x["p"] for x in st])):
                rows.append((x["delta"], x["p"], a, x["bound"]))
    pooled = holm([r[1] for r in rows])
    defs["CalibNComparisons"] = f"{len(rows)}"
    defs["CalibNSigQuantumHolm"] = f"{sum(r[2] < 0.05 and r[0] > 0 for r in rows)}"
    defs["CalibNSigTwinHolm"] = f"{sum(r[2] < 0.05 and r[0] < 0 for r in rows)}"
    defs["CalibNSigQuantumPooled"] = f"{sum(q < 0.05 and r[0] > 0 for r, q in zip(rows, pooled))}"
    defs["CalibNSigTwinPooled"] = f"{sum(q < 0.05 and r[0] < 0 for r, q in zip(rows, pooled))}"
    # The largest mean difference is what sits inside the margin; the largest
    # bound (CalibWorstBound) does not, and the prose had conflated the two.
    big = max(rows, key=lambda r: abs(r[0]))
    defs["CalibMaxDelta"] = f"{big[0]:+.4f}"
    defs["CalibNEquiv"] = f"{sum(r[3] < 0.02 for r in rows)}"

    # Where the best reference-frame kernel beats the best classical pipeline.
    # "Only at the smallest training set" was typed by hand; on Cho2017 the
    # uncorrected test also passes at the second-smallest.
    beats, head_p = [], []
    for key, (label, per) in d.items():
        for k in sorted(per.index.get_level_values(0).unique()):
            pk = per.loc[k]
            ks = [q for q in REF_KERNELS if q in pk.columns]
            cl = [c for c in pk.columns if c.startswith("classical/")]
            if not ks or not cl:
                continue
            bq = max(ks, key=lambda q: pk[q].mean())
            bc = pk[cl].mean().idxmax()
            x = _paired(pk, bq, bc)
            if np.isnan(x["p"]):
                continue
            head_p.append((label, int(k), x["delta"], x["p"]))
    head_adj = holm([h[3] for h in head_p])
    by_set: dict[str, list[str]] = {}
    for (label, k, dl, pv), a in zip(head_p, head_adj):
        if pv < 0.05 and dl > 0:
            by_set.setdefault(label, []).append(k)

    def _sizes(v):
        v = [f"${x}$" for x in v]
        v[0] = "$k=" + v[0][1:]
        return v[0] if len(v) == 1 else ", ".join(v[:-1]) + " and " + v[-1]

    parts = [f"on {lab} at {_sizes(v)}" for lab, v in by_set.items()]
    defs["CalibBeatClassicalList"] = (
        "nowhere" if not parts else parts[0] if len(parts) == 1
        else ", ".join(parts[:-1]) + ", and " + parts[-1])
    n_h = sum(a < 0.05 and dl > 0 for (_, _, dl, _), a in zip(head_p, head_adj))
    small = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    defs["CalibBeatClassicalHolm"] = small.get(n_h, str(n_h)) if n_h else "none"
    defs["CalibNHeadComparisons"] = f"{len(head_p)}"
    out.append("\n%% ------------------------------ calibration macros\n")
    for key, value in defs.items():
        out.append(f"\\newcommand{{\\{key}}}{{{value}}}")
    out.append("")
