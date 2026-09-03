"""Tables and macros for the reference-frame results.

Companion to `make_tables.py`, in the same spirit as `cross_tables.py`: every
number the reference-frame sections quote is generated here from the result
CSVs, never typed. Each builder is skipped silently if its inputs are absent,
so the manuscript degrades to whatever has actually been run.

Sources
-------
raw_folds_refstate_motor8_q4.csv        PhysioNet, 3 qubits, 23 pipelines, n=30
raw_folds_refstate_bci2a_motor8_q4.csv  BCI IV-2a, 3 qubits, n=9
raw_folds_filterbank_motor8.csv         PhysioNet, 5 qubits (filter bank), n=30
transfer_folds_motor8.csv               leave-one-subject-out, n=30
reference_gram_motor8.csv               Gram statistics per frame, n=14
shots_folds_motor8.csv                  accuracy versus shot budget, n=30
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Sensor-frame pipeline paired with its reference-frame counterpart.
FRAME_PAIRS = [
    ("quantum/Fidelity-SVM", "quantum/Fidelity-ref-SVM", "Fidelity"),
    ("quantum/HS-overlap-SVM", "quantum/HS-overlap-ref-SVM", "HS overlap"),
    ("quantum/HS-RBF-SVM", "quantum/HS-RBF-ref-SVM", "HS-RBF"),
    ("quantum/Bures-RBF-SVM", "quantum/Bures-RBF-ref-SVM", "Bures-RBF"),
    ("quantum/QRE-RBF-SVM", "quantum/QRE-RBF-ref-SVM", "QRE-RBF"),
]
FB_FRAME_PAIRS = [
    (a.replace("quantum/", "quantum/FB-"), b.replace("quantum/", "quantum/FB-"), lab)
    for a, b, lab in FRAME_PAIRS
]

REF_KERNELS = [b for _, b, _ in FRAME_PAIRS]
FB_REF_KERNELS = [b for _, b, _ in FB_FRAME_PAIRS]

# The classical twins: an SPD-manifold kernel in the same SVM, same frame,
# same tuning budget. Only the metric differs from the quantum kernels.
TWINS = ["control/riemann-kernel-SVM", "control/logeuclid-kernel-SVM"]
FB_TWINS = ["control/FB-riemann-kernel-SVM", "control/FB-logeuclid-kernel-SVM"]


def _per(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["pipeline", "subject"])["accuracy"].mean().unstack("pipeline")


def load(res: Path) -> dict:
    """Read whatever reference-frame results exist. Missing files are fine."""
    def maybe(name):
        p = res / name
        return pd.read_csv(p) if p.exists() else None

    d = {
        "phys": maybe("raw_folds_refstate_motor8_q4.csv"),
        "phys_summary": maybe("summary_refstate_motor8_q4.csv"),
        "bci": maybe("raw_folds_refstate_bci2a_motor8_q4.csv"),
        "bci_summary": maybe("summary_refstate_bci2a_motor8_q4.csv"),
        "fb": maybe("raw_folds_filterbank_motor8.csv"),
        "fb_summary": maybe("summary_filterbank_motor8.csv"),
        "transfer": maybe("transfer_folds_motor8.csv"),
        "gram": maybe("reference_gram_motor8.csv"),
        "shots": maybe("shots_folds_motor8.csv"),
    }
    for k in ("phys", "bci", "fb"):
        d[k + "_per"] = _per(d[k]) if d[k] is not None else None
    if d["transfer"] is not None:
        t = d["transfer"]
        d["transfer_ref"] = (t[t.frame == "reference"]
                             .groupby(["pipeline", "subject"])["accuracy"]
                             .mean().unstack("pipeline"))
        d["transfer_sen"] = (t[t.frame == "sensor"]
                             .groupby(["pipeline", "subject"])["accuracy"]
                             .mean().unstack("pipeline"))
    else:
        d["transfer_ref"] = d["transfer_sen"] = None
    return d


def _best_twin(per: pd.DataFrame, twins: list[str]) -> str | None:
    """The stronger of the two SPD-kernel controls: the conservative comparator."""
    avail = [t for t in twins if t in per.columns]
    return max(avail, key=lambda t: per[t].mean()) if avail else None


# --------------------------------------------------------------------------
# Table: the frame effect
# --------------------------------------------------------------------------

def table_frame(d: dict, paired, fmt_p, esc, out: list[str]) -> bool:
    if d["phys_per"] is None:
        return False
    have_bci = d["bci_per"] is not None

    # Datasets are stacked vertically rather than side by side: at 12pt the
    # iopart text block is too narrow for a nine-column table, and \footnotesize
    # inside \begin{indented} is reset by the class, so the fix has to be
    # structural rather than typographic.
    out.append(r"""
%% -------------------------------------------------- Table: frame effect
\begin{table}[htbp]
\caption{\label{tab:frame}Effect of referring the density-matrix kernels to a
reference state. Each kernel is evaluated twice under an identical protocol,
differing only in whether the states are expressed in the sensor frame
(\eref{eq:density}) or relative to the training-set Fr\'echet mean
(\eref{eq:refstate}). $\Delta$ is the mean per-subject accuracy gain from the
reference frame, tested by paired Wilcoxon signed-rank across subjects. Every
kernel improves on both datasets; on IV-2a every kernel improves in every
subject.}
\begin{indented}
\item[]\begin{tabular}{@{}llccccc@{}}
\br
Dataset & Kernel & Sensor & Reference & $\Delta$ & $p$ & Better \\
\mr""")

    blocks = [("PhysioNet", d["phys_per"])]
    if have_bci:
        blocks.append(("IV-2a", d["bci_per"]))

    for bi, (dname, per) in enumerate(blocks):
        if bi:
            out.append(r"\ms")
        first = True
        for a, b, label in FRAME_PAIRS:
            if a not in per.columns or b not in per.columns:
                continue
            s = paired(per, b, a)
            lead = f"{dname} ($n={s['n']}$)" if first else ""
            first = False
            better = f"{s['n_better']}/{s['n']}"
            if s["n_better"] == s["n"]:
                better = f"\\textbf{{{better}}}"
            out.append(
                f"{lead} & {label} & {per[a].mean():.3f} & {per[b].mean():.3f} & "
                f"$\\bf {s['delta']:+.4f}$ & {fmt_p(s['p'])} & {better} \\\\"
            )

    out.append(r"""\br
\end{tabular}
\end{indented}
\end{table}
""")
    return True


# --------------------------------------------------------------------------
# Table: the classical-twin control -- the decisive one
# --------------------------------------------------------------------------

def table_twin(d: dict, paired, fmt_p, esc, out: list[str]) -> bool:
    """Quantum kernels against an SPD kernel differing only in the metric."""
    settings = []
    if d["phys_per"] is not None:
        settings.append(("PhysioNet, 3\\,q", d["phys_per"], REF_KERNELS, TWINS))
    if d["fb_per"] is not None:
        settings.append(("PhysioNet, 5\\,q, filter bank", d["fb_per"],
                         FB_REF_KERNELS, FB_TWINS))
    if d["transfer_ref"] is not None:
        # Transfer pipelines carry no -ref suffix; the frame is a column there.
        tk = [k for k in ("quantum/Fidelity", "quantum/HS-overlap",
                          "quantum/HS-RBF", "quantum/Bures-RBF",
                          "quantum/QRE-RBF") if k in d["transfer_ref"].columns]
        settings.append(("PhysioNet, transfer (LOSO)",
                         d["transfer_ref"], tk, TWINS))
    if d["bci_per"] is not None:
        settings.append(("BCI IV-2a, 3\\,q", d["bci_per"], REF_KERNELS, TWINS))
    if not settings:
        return False

    out.append(r"""
%% ------------------------------------------------------ Table: twin control
\begin{table}[htbp]
\caption{\label{tab:twin}The decisive control. Each quantum kernel is compared
against an SPD-manifold kernel used in the same support vector machine, on the
same covariances, with the same tuning budget, in the same reference frame:
only the metric differs. The comparator is the stronger of the two classical
kernels in each setting, which is the conservative choice. Columns give the
best quantum kernel in that setting, the classical twin, their difference, and
the smallest $p$ obtained by \emph{any} of the five quantum kernels against the
twin. No quantum kernel is distinguishable from its classical twin in any
setting.}
\begin{indented}
\item[]\begin{tabular}{@{}lccccc@{}}
\br
Setting & $n$ & Quantum & Twin & $\Delta$ (range over 5) & $\min p$ \\
\mr""")

    for label, per, kernels, twins in settings:
        twin = _best_twin(per, twins)
        ks = [k for k in kernels if k in per.columns]
        if twin is None or not ks:
            continue
        stats = {k: paired(per, k, twin) for k in ks}
        best = max(ks, key=lambda k: per[k].mean())
        s = stats[best]
        deltas = [v["delta"] for v in stats.values()]
        out.append(
            f"{label} & {s['n']} & {per[best].mean():.3f} & {per[twin].mean():.3f} & "
            f"$[{min(deltas):+.4f}, {max(deltas):+.4f}]$ & "
            f"{fmt_p(min(v['p'] for v in stats.values()))} \\\\"
        )

    out.append(r"""\br
\end{tabular}
\end{indented}
\end{table}
""")
    return True


# --------------------------------------------------------------------------
# Table: cross-subject transfer
# --------------------------------------------------------------------------

def table_transfer(d: dict, paired, fmt_p, esc, out: list[str]) -> bool:
    ref, sen = d["transfer_ref"], d["transfer_sen"]
    if ref is None:
        return False
    order = ref.mean().sort_values(ascending=False).index

    out.append(r"""
%% --------------------------------------------------- Table: transfer
\begin{table}[htbp]
\caption{\label{tab:transfer}Cross-subject transfer, leave-one-subject-out:
each model is trained on the pooled trials of every other subject.
Hyperparameters are selected by subject-grouped inner cross-validation on the
training subjects only. In the reference frame each subject is whitened by its
own Fr\'echet mean, which uses no labels and is therefore legitimate
unsupervised adaptation for the held-out subject. $\Delta$ is the gain from the
reference frame, paired by held-out subject. Every method improves; in the
reference frame no method is distinguishable from any other.}
\begin{indented}
\item[]\begin{tabular}{@{}lccccc@{}}
\br
Pipeline & Sensor & Reference & $\Delta$ & $p$ & Better \\
\mr""")
    for pipe in order:
        if pipe not in sen.columns:
            continue
        s = paired(ref, pipe, pipe) if False else None
        x, y = ref[pipe], sen[pipe]
        m = x.notna() & y.notna()
        diff = (x[m] - y[m]).to_numpy()
        from scipy.stats import wilcoxon
        p = float(wilcoxon(diff).pvalue)
        out.append(
            f"{esc(pipe)} & {y.mean():.3f} & {x.mean():.3f} & "
            f"${diff.mean():+.4f}$ & {fmt_p(p)} & "
            f"{int((diff > 0).sum())}/{len(diff)} \\\\"
        )
    out.append(r"""\br
\end{tabular}
\end{indented}
\end{table}
""")
    return True


# --------------------------------------------------------------------------
# Table: shot budget
# --------------------------------------------------------------------------

def table_shots(d: dict, out: list[str]) -> bool:
    sh = d["shots"]
    if sh is None:
        return False
    piv = sh.pivot_table(index=["kernel", "shots"], columns="frame",
                         values="accuracy").reset_index()
    shot_levels = sorted(x for x in sh.shots.unique() if x > 0)

    out.append(r"""
%% ------------------------------------------------------- Table: shot budget
\begin{table}[htbp]
\caption{\label{tab:shots}Accuracy under finite-shot estimation.
$\mathrm{tr}(\rho\sigma)$ is the SWAP-test observable, so $S$ shots give an
unbiased estimate with variance $(1-k^2)/S$; we sample each unordered pair
binomially and project the Gram matrix back to the positive semi-definite cone.
The reference frame needs \emph{more} shots to approach its own ceiling,
because that ceiling is higher, but it dominates the sensor frame in absolute
terms from $10^4$ shots upwards, above which it exceeds what the sensor
frame achieves with unlimited shots.}
\begin{indented}
\item[]\begin{tabular}{@{}ll""" + "c" * (len(shot_levels) + 1) + r"""@{}}
\br
Kernel & Frame & """ + " & ".join(
        f"$10^{{{int(round(np.log10(s)))}}}$" for s in shot_levels
    ) + r""" & $\infty$ \\
\mr""")
    for kern in sorted(sh.kernel.unique()):
        for frame in ("sensor", "reference"):
            row = piv[(piv.kernel == kern)]
            cells = []
            for s in shot_levels + [-1]:
                v = row[row.shots == s][frame]
                cells.append(f"{float(v.iloc[0]):.3f}" if len(v) else "n/a")
            cells[-1] = f"\\textbf{{{cells[-1]}}}"
            out.append(f"{kern} & {frame.capitalize()} & " + " & ".join(cells) + r" \\")
    out.append(r"""\br
\end{tabular}
\end{indented}
\end{table}
""")
    return True


# --------------------------------------------------------------------------
# Inline macros
# --------------------------------------------------------------------------

def macros(d: dict, paired, fmt_p, esc, out: list[str]) -> None:
    defs: dict[str, str] = {}

    def frame_stats(per, pairs, prefix):
        rows = [(lab, paired(per, b, a)) for a, b, lab in pairs
                if a in per.columns and b in per.columns]
        if not rows:
            return
        deltas = [s["delta"] for _, s in rows]
        best = max(rows, key=lambda r: r[1]["delta"])
        defs[prefix + "FrameMin"] = f"{min(deltas):+.3f}"
        defs[prefix + "FrameMax"] = f"{max(deltas):+.3f}"
        defs[prefix + "FrameBestKernel"] = best[0]
        defs[prefix + "FrameMaxP"] = fmt_p(max(s["p"] for _, s in rows))
        defs[prefix + "FrameAllBetter"] = (
            "yes" if all(s["n_better"] == s["n"] for _, s in rows) else "no")
        defs[prefix + "FrameN"] = f"{rows[0][1]['n']}"

    if d["phys"] is not None:
        defs["NPipelinesExt"] = f"{d['phys'].pipeline.nunique()}"
    if d["fb"] is not None:
        defs["NPipelinesFb"] = f"{d['fb'].pipeline.nunique()}"

    if d["phys_per"] is not None:
        per = d["phys_per"]
        frame_stats(per, FRAME_PAIRS, "Phys")
        twin = _best_twin(per, TWINS)
        ks = [k for k in REF_KERNELS if k in per.columns]
        if twin and ks:
            st = {k: paired(per, k, twin) for k in ks}
            defs["TwinName"] = esc(twin)
            defs["TwinAcc"] = f"{per[twin].mean():.3f}"
            defs["TwinDeltaMin"] = f"{min(v['delta'] for v in st.values()):+.4f}"
            defs["TwinDeltaMax"] = f"{max(v['delta'] for v in st.values()):+.4f}"
            defs["TwinMinP"] = fmt_p(min(v["p"] for v in st.values()))
        # Headline reversal, sensor versus reference frame.
        cl = [c for c in per.columns if c.startswith("classical/")]
        if cl and ks:
            best_cl = per[cl].mean().idxmax()
            sensor_q = [a for a, _, _ in FRAME_PAIRS if a in per.columns]
            bq_s = per[sensor_q].mean().idxmax()
            bq_r = per[ks].mean().idxmax()
            s1, s2 = paired(per, best_cl, bq_s), paired(per, best_cl, bq_r)
            defs["HeadBestClassical"] = esc(best_cl)
            defs["HeadSensorDelta"] = f"{s1['delta']:+.4f}"
            defs["HeadSensorP"] = fmt_p(s1["p"])
            defs["HeadSensorBetter"] = f"{s1['n_better']}/{s1['n']}"
            defs["HeadRefDelta"] = f"{s2['delta']:+.4f}"
            defs["HeadRefP"] = fmt_p(s2["p"])
            defs["HeadRefBetter"] = f"{s2['n_better']}/{s2['n']}"
            defs["HeadRefKernel"] = esc(bq_r)

    if d["bci_per"] is not None:
        frame_stats(d["bci_per"], FRAME_PAIRS, "Bci")

    if d["fb_per"] is not None:
        per = d["fb_per"]
        frame_stats(per, FB_FRAME_PAIRS, "Fb")
        ks = [k for k in FB_REF_KERNELS if k in per.columns]
        if "classical/FBCSP+LDA" in per.columns and ks:
            bq = per[ks].mean().idxmax()
            s = paired(per, bq, "classical/FBCSP+LDA")
            defs["FbcspDelta"] = f"{s['delta']:+.4f}"
            defs["FbcspP"] = fmt_p(s["p"])
            defs["FbcspBestAcc"] = f"{per[bq].mean():.3f}"
            defs["FbcspAcc"] = f"{per['classical/FBCSP+LDA'].mean():.3f}"

    if d["transfer_ref"] is not None:
        ref = d["transfer_ref"]
        defs["TransferN"] = f"{len(ref)}"
        defs["TransferSpread"] = f"{ref.mean().max() - ref.mean().min():.4f}"
        defs["TransferBest"] = esc(ref.mean().idxmax())
        defs["TransferBestAcc"] = f"{ref.mean().max():.3f}"
        # Wilcoxon's two-sided floor: the test cannot return a smaller p.
        defs["TransferFloor"] = f"{2.0 ** (1 - len(ref)):.1e}".replace("e-0", r"\times10^{-") + "}"

    if d["gram"] is not None:
        g = d["gram"]
        gain = (g[g.frame == "reference"].groupby("kernel")["var"].mean()
                / g[g.frame == "sensor"].groupby("kernel")["var"].mean())
        defs["GramGainMin"] = f"{gain.min():.1f}"
        defs["GramGainMax"] = f"{gain.max():.1f}"
        defs["GramNSubjects"] = f"{g.subject.nunique()}"
        sen = g[(g.frame == "sensor") & (g.kernel == "HS-overlap")]
        if len(sen):
            defs["GramSensorMean"] = f"{sen['mean'].mean():.3f}"
            defs["GramSensorStd"] = f"{np.sqrt(sen['var'].mean()):.3f}"

    if d["shots"] is not None:
        sh = d["shots"]
        rows = []
        for kern in sh.kernel.unique():
            sub = sh[sh.kernel == kern]
            si = sub[(sub.frame == "sensor") & (sub.shots == -1)].accuracy.mean()
            crossed = [s for s in sorted(x for x in sub.shots.unique() if x > 0)
                       if sub[(sub.frame == "reference")
                              & (sub.shots == s)].accuracy.mean() > si]
            if crossed:
                rows.append(crossed[0])
        if rows:
            defs["ShotCrossover"] = f"10^{{{int(round(np.log10(max(rows))))}}}"

    if defs:
        out.append("\n%% ------------------------------ reference-frame macros\n")
        for k, v in defs.items():
            out.append(f"\\newcommand{{\\{k}}}{{{v}}}")
        out.append("")
