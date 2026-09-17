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


def fmt_p_eq(p):
    """Relation-carrying p-value; see make_tables.fmt_p_eq."""
    rel = "<" if p < 0.001 else "="
    val = "0.001" if p < 0.001 else f"{p:.3f}"
    return r"\ensuremath{{}" + rel + r"{}}" + val

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

# The transfer and cross-session suites carry the frame as a column rather
# than a "-ref" suffix, so their quantum pipelines are named without it.
TRANSFER_KERNELS = ["quantum/Fidelity", "quantum/HS-overlap", "quantum/HS-RBF",
                    "quantum/Bures-RBF", "quantum/QRE-RBF"]


def _per(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["pipeline", "subject"])["accuracy"].mean().unstack("pipeline")


def load(res: Path) -> dict:
    """Read whatever reference-frame results exist. Missing files are fine."""
    def maybe(name):
        p = res / name
        return pd.read_csv(p) if p.exists() else None

    d = {
        "_res": res,
        "phys": maybe("raw_folds_refstate_motor8_q4.csv"),
        "phys_summary": maybe("summary_refstate_motor8_q4.csv"),
        "bci": maybe("raw_folds_refstate_bci2a_motor8_q4.csv"),
        "bci_summary": maybe("summary_refstate_bci2a_motor8_q4.csv"),
        "fb": maybe("raw_folds_filterbank_motor8.csv"),
        "fb_summary": maybe("summary_filterbank_motor8.csv"),
        "transfer": maybe("transfer_folds_motor8.csv"),
        "gram": maybe("reference_gram_motor8.csv"),
        "sweep": maybe("reference_gram_sweep.csv"),
        "shots": maybe("shots_folds_motor8.csv"),
        "cs": maybe("crosssession_folds_bci2a_motor8.csv"),
        # Robustness runs: the same extended suite at 16 channels (4 qubits),
        # on Cho2017 (52 subjects), and under two further outer-CV seeds.
        "m16": maybe("raw_folds_refstate_motor16_q4.csv"),
        "cho": maybe("raw_folds_refstate_cho2017_motor8_q4.csv"),
        "cho_summary": maybe("summary_refstate_cho2017_motor8_q4.csv"),
        "seeds": {s: maybe(f"raw_folds_refstate_motor8_q4_seed{s}.csv")
                  for s in (1, 2)},
    }
    for k in ("phys", "bci", "fb", "m16", "cho"):
        d[k + "_per"] = _per(d[k]) if d[k] is not None else None
    d["seeds_per"] = {s: _per(v) for s, v in d["seeds"].items() if v is not None}
    if d["phys_per"] is not None:
        d["seeds_per"] = {0: d["phys_per"], **d["seeds_per"]}
    if d["cs"] is not None:
        # Both directions (A->B, B->A) are averaged per subject first, so each
        # subject is one paired observation, exactly as in the transfer suite.
        c = d["cs"]
        d["cs_ref"] = (c[c.frame == "reference"]
                       .groupby(["pipeline", "subject"])["accuracy"]
                       .mean().unstack("pipeline"))
        d["cs_sen"] = (c[c.frame == "sensor"]
                       .groupby(["pipeline", "subject"])["accuracy"]
                       .mean().unstack("pipeline"))
    else:
        d["cs_ref"] = d["cs_sen"] = None
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


def _sci(x: float) -> str:
    """LaTeX scientific notation, for any exponent.

    The previous version did this by string-replacing "e-0", which silently
    assumed a single-digit exponent. That held while the cohort was 30 subjects
    (Wilcoxon's floor is 1.9e-09) and broke at 104 (9.9e-32): the replacement
    did not fire, and the closing brace that was appended unconditionally was
    then unmatched, leaving a macro file that would not compile.
    """
    mantissa, exponent = f"{x:.1e}".split("e")
    return f"{mantissa}\\times10^{{{int(exponent)}}}"


def _best_twin(per: pd.DataFrame, twins: list[str]) -> str | None:
    """The comparator: the Riemannian SPD kernel, always.

    It is the only classical kernel here with the reference-frame quantum
    kernels' invariance group: pyRiemann whitens by the Frechet mean before
    taking logarithms, so a common congruence cancels exactly, to machine
    precision. The log-Euclidean kernel centres in the log domain instead and
    is not congruence-invariant, so it is reported as a second classical
    kernel, never used as the twin. (Until 2026-09-16 the stronger of the
    two was used, which made the log-Euclidean kernel the comparator in the
    five-qubit setting.)
    """
    riemann = [t for t in twins if "riemann" in t and t in per.columns]
    return riemann[0] if riemann else None


def holm(pvals) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def _second_kernel(per: pd.DataFrame, twins: list[str]) -> str | None:
    """The log-Euclidean SPD kernel, reported alongside the twin."""
    le = [t for t in twins if "logeuclid" in t and t in per.columns]
    return le[0] if le else None


# --------------------------------------------------------------------------
# Table: the frame effect
# --------------------------------------------------------------------------

def table_frame(d: dict, paired, fmt_p, esc, out: list[str]) -> bool:
    if d["phys_per"] is None:
        return False
    have_bci = d["bci_per"] is not None

    # Datasets are stacked vertically rather than side by side: at 12pt the
    # iopart text block is too narrow for a nine-column table, and \footnotesize
    # inside  is reset by the class, so the fix has to be
    # structural rather than typographic.
    out.append(r"""
%% -------------------------------------------------- Table: frame effect
\begin{table}[htbp]
\caption{\label{tab:frame}Effect of referring the density-matrix kernels to a
reference state. Each kernel is evaluated twice under an identical protocol,
differing only in whether the states are expressed in the sensor frame
(\ref{eq:density}) or relative to the training-set Fr\'echet mean
(\ref{eq:refstate}). $\Delta$ is the mean per-subject accuracy gain from the
reference frame, tested by paired Wilcoxon signed-rank across subjects;
$p_{\mathrm{Holm}}$ corrects for every row of this table as one family.
Every kernel improves on every dataset.}
\begin{tabular}{@{}llcccccc@{}}
\hline
Dataset & Kernel & Sensor & Reference & $\Delta$ & $p$ & $p_{\mathrm{Holm}}$ & Better \\
\hline""")

    blocks = [("PhysioNet, 3\\,q", d["phys_per"])]
    if d["m16_per"] is not None:
        blocks.append(("PhysioNet, 4\\,q", d["m16_per"]))
    if d["cho_per"] is not None:
        blocks.append(("Cho2017, 3\\,q", d["cho_per"]))
    if have_bci:
        blocks.append(("IV-2a, 3\\,q", d["bci_per"]))

    # One Holm family over every row, computed before any row is written.
    rows = []
    for dname, per in blocks:
        for a, b, label in FRAME_PAIRS:
            if a in per.columns and b in per.columns:
                rows.append((dname, per, a, b, label, paired(per, b, a)))
    p_holm = holm([r[5]["p"] for r in rows])

    prev = None
    for (dname, per, a, b, label, s), ph in zip(rows, p_holm):
        if prev is not None and dname != prev:
            out.append(r"\hline")
        lead = f"{dname} ($n={s['n']}$)" if dname != prev else ""
        prev = dname
        better = f"{s['n_better']}/{s['n']}"
        if s["n_better"] == s["n"]:
            better = f"\\textbf{{{better}}}"
        out.append(
            f"{lead} & {label} & {per[a].mean():.3f} & {per[b].mean():.3f} & "
            f"$\\bf {s['delta']:+.4f}$ & {fmt_p(s['p'])} & {fmt_p(ph)} & "
            f"{better} \\\\"
        )

    out.append(r"""\hline
\end{tabular}
\end{table}
""")
    return True


# --------------------------------------------------------------------------
# Table: the classical-twin control: the decisive one
# --------------------------------------------------------------------------

def table_twin(d: dict, paired, fmt_p, esc, out: list[str]) -> bool:
    """Quantum kernels against an SPD kernel differing only in the metric."""
    settings = []
    if d["phys_per"] is not None:
        settings.append(("PhysioNet, 3\\,q", d["phys_per"], REF_KERNELS, TWINS))
    if d["m16_per"] is not None:
        settings.append(("PhysioNet, 4\\,q", d["m16_per"], REF_KERNELS, TWINS))
    if d["fb_per"] is not None:
        settings.append(("PhysioNet, 5\\,q, filter bank", d["fb_per"],
                         FB_REF_KERNELS, FB_TWINS))
    if d["transfer_ref"] is not None:
        # Transfer pipelines carry no -ref suffix; the frame is a column there.
        tk = [k for k in TRANSFER_KERNELS if k in d["transfer_ref"].columns]
        settings.append(("PhysioNet, transfer (LOSO)",
                         d["transfer_ref"], tk, TWINS))
    if d["bci_per"] is not None:
        settings.append(("BCI IV-2a, 3\\,q", d["bci_per"], REF_KERNELS, TWINS))
    if d["cs_ref"] is not None:
        ck = [k for k in TRANSFER_KERNELS if k in d["cs_ref"].columns]
        settings.append(("BCI IV-2a, cross-session", d["cs_ref"], ck, TWINS))
    if d["cho_per"] is not None:
        settings.append(("Cho2017, 3\\,q", d["cho_per"], REF_KERNELS, TWINS))
    if not settings:
        return False

    out.append(r"""
%% ------------------------------------------------------ Table: twin control
\begin{table}[htbp]
\caption{\label{tab:twin}The decisive control. Each quantum kernel is compared
against an SPD-manifold kernel used in the same support vector machine, on the
same covariances, with the same tuning budget, in the same reference frame:
only the metric differs. In every setting the twin is the affine-invariant
Riemannian kernel, the one classical kernel here that shares the
reference-frame quantum kernels' invariance under congruence. The
log-Euclidean kernel lacks that invariance and is shown for comparison only.
Columns give the best quantum kernel, the twin, the log-Euclidean kernel, the
range of the five quantum-minus-twin differences, and the smallest $p$
obtained by \emph{any} of the five quantum kernels against the twin.
Table~\ref{tab:seeds} repeats the PhysioNet three-qubit row under two further
partitions.}
\begin{tabular}{@{}lcccccc@{}}
\hline
Setting & $n$ & Quantum & Twin & Log-Eucl. & $\Delta$ (range over 5) & $\min p$ \\
\hline""")

    for label, per, kernels, twins in settings:
        twin = _best_twin(per, twins)
        second = _second_kernel(per, twins)
        ks = [k for k in kernels if k in per.columns]
        if twin is None or not ks:
            continue
        stats = {k: paired(per, k, twin) for k in ks}
        best = max(ks, key=lambda k: per[k].mean())
        s = stats[best]
        deltas = [v["delta"] for v in stats.values()]
        second_acc = f"{per[second].mean():.3f}" if second else "n/a"
        out.append(
            f"{label} & {s['n']} & {per[best].mean():.3f} & "
            f"{per[twin].mean():.3f} & {second_acc} & "
            f"$[{min(deltas):+.4f}, {max(deltas):+.4f}]$ & "
            f"{fmt_p(min(v['p'] for v in stats.values()))} \\\\"
        )

    out.append(r"""\hline
\end{tabular}
\end{table}
""")
    return True


# --------------------------------------------------------------------------
# Table: equivalence (TOST) against the classical twin
# --------------------------------------------------------------------------

def table_equivalence(res: Path, out: list[str], margin: float = 0.02) -> bool:
    """Two one-sided tests, so the null being rejected is a *difference*.

    A non-significant difference does not establish sameness. This table is
    what turns "we could not tell them apart" into "they agree to within a
    stated margin", which is the claim the paper actually needs.
    """
    p = res / "equivalence_twin.csv"
    if not p.exists():
        return False
    d = pd.read_csv(p)

    out.append(r"""
%% --------------------------------------------------- Table: equivalence
\begin{table}[htbp]
\caption{\label{tab:equiv}Equivalence of each quantum kernel to its classical
twin, by two one-sided tests on the paired per-subject differences. A
non-significant difference would only be an absence of evidence; TOST instead
takes a \emph{difference} as its null, so rejecting it supports equivalence.
The margin $m=""" + f"{margin:g}" + r"""$ accuracy is not pre-registered. It
was chosen after the first within-subject analyses of PhysioNet and IV-2a, the
filter-bank analysis and the cross-subject analysis had been run, from the size
of the reference-frame effects they showed, and before every other analysis
reported here. It is left unchanged although some later effects are smaller
(the correction now spans \FrameEffectMin{} to \FrameEffectMax). CI is the
90\,\% interval, which
corresponds to TOST at $\alpha=0.05$; ``bound'' is the smallest margin at which
equivalence would hold, so a reader preferring a stricter margin can read the
answer off directly.}
\begin{tabular}{@{}llcccccc@{}}
\hline
Setting & Kernel & $n$ & $\Delta$ & 90\,\% CI & $p_{\mathrm{TOST}}$ &
Equiv. & Bound \\
\hline""")
    prev = None
    for _, r in d.iterrows():
        lead = r["setting"] if r["setting"] != prev else ""
        prev = r["setting"]
        # A cross, not a dash: the project uses no em-dashes anywhere.
        mark = r"\checkmark" if r["equivalent"] else r"$\times$"
        out.append(
            f"{lead} & {r['kernel']} & {int(r['n'])} & ${r['mean']:+.4f}$ & "
            f"$[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]$ & "
            f"{r['p_tost']:.4f} & {mark} & {r['bound']:.4f} \\\\"
        )
    out.append(r"""\hline
\end{tabular}
\end{table}
""")
    return True


def equivalence_macros(res: Path, out: list[str], margin: float = 0.02) -> None:
    p = res / "equivalence_twin.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    zero_excl = d[(d.ci_low > 0) | (d.ci_high < 0)]
    defs = {
        "EquivMargin": f"{margin:g}",
        "EquivN": f"{len(d)}",
        "EquivPass": f"{int(d.equivalent.sum())}",
        "EquivWorstBound": f"{d['bound'].max():.3f}",
        "EquivFailQuantum": f"{int((d[~d.equivalent]['mean'] > 0).sum())}",
        "EquivFailClassical": f"{int((d[~d.equivalent]['mean'] < 0).sum())}",
        "EquivNZeroExcluded": f"{len(zero_excl)}",
    }
    # The number of settings, as a word, so that "five settings" in the prose
    # cannot go stale when a setting is added.
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
             7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    ns = d.setting.nunique()
    defs["EquivNSettings"] = words.get(ns, str(ns))
    defs["EquivNDatasets"] = words[len({
        "PhysioNet" if "PhysioNet" in s or "Transfer" in s
        else "Cho2017" if "Cho2017" in s else "IV-2a" for s in d.setting})]
    # Register sizes: "N qubits" in the setting name; the transfer and
    # cross-session suites run on the eight-channel, three-qubit register.
    import re
    regs = {int(m.group(1)) if (m := re.search(r"(\d) qubits", s)) else 3
            for s in d.setting}
    defs["EquivNRegisters"] = words[len(regs)]
    # Per-setting figures. Exact names: "IV-2a" would otherwise also match the
    # cross-session setting, and the within-subject sentence would be wrong.
    for prefix, name in (("Bci", "BCI IV-2a, 3 qubits"),
                         ("Cs", "IV-2a cross-session"),
                         ("Tr", "Transfer (LOSO)"),
                         ("FourQ", "PhysioNet, 4 qubits"),
                         ("Cho", "Cho2017, 3 qubits")):
        sub = d[d.setting == name]
        if not len(sub):
            continue
        defs[f"Equiv{prefix}Pass"] = f"{int(sub.equivalent.sum())}"
        defs[f"Equiv{prefix}N"] = f"{len(sub)}"
        defs[f"Equiv{prefix}Worst"] = f"{sub['bound'].max():.4f}"
    # Grammar that follows the count, so a caption written once stays correct
    # whether the new cohort leaves zero, one or several intervals off zero.
    n_ze = len(zero_excl)
    defs["EquivZeroExclWord"] = "interval" if n_ze == 1 else "intervals"
    defs["EquivZeroExclVerb"] = "excludes" if n_ze == 1 else "exclude"
    defs["EquivZeroExclMarker"] = ("an orange diamond" if n_ze == 1
                                   else "orange diamonds")
    if len(zero_excl):
        r = zero_excl.iloc[0]
        defs["EquivZeroExclName"] = f"{r['kernel']} ({r['setting']})"
        defs["EquivZeroExclDelta"] = f"{r['mean']:+.4f}"
        # All of them, as a sentence fragment, so the prose can name each one
        # however many there turn out to be.
        items = [f"{x['kernel']} on {x['setting'].replace(', 3 qubits', '')} "
                 f"at ${x['mean']:+.4f}$" for _, x in zero_excl.iterrows()]
        defs["EquivZeroExclList"] = (items[0] if len(items) == 1
                                     else ", ".join(items[:-1]) + " and " + items[-1])
        defs["EquivZeroExclAllQuantum"] = (
            "yes" if (zero_excl["mean"] > 0).all() else "no")
        # The sign pattern, as a clause: the manuscript's claim that every
        # off-zero interval favours the quantum kernel must not survive a
        # cohort in which one of them does not.
        defs["EquivZeroExclSide"] = (
            "all of them on the quantum side" if (zero_excl["mean"] > 0).all()
            else "all of them on the twin's side" if (zero_excl["mean"] < 0).all()
            else "mixed in sign")
    else:
        defs["EquivZeroExclList"] = "none"
        defs["EquivZeroExclName"] = "none"
        defs["EquivZeroExclDelta"] = "n/a"
        defs["EquivZeroExclAllQuantum"] = "n/a"
        defs["EquivZeroExclSide"] = "none"
    out.append("\n%% ------------------------------ equivalence macros\n")
    for k, v in defs.items():
        out.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    out.append("")


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
\begin{tabular}{@{}lccccc@{}}
\hline
Pipeline & Sensor & Reference & $\Delta$ & $p$ & Better \\
\hline""")
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
            f"{esc(pipe.split(chr(47), 1)[-1])} & {y.mean():.3f} & {x.mean():.3f} & "
            f"${diff.mean():+.4f}$ & {fmt_p(p)} & "
            f"{int((diff > 0).sum())}/{len(diff)} \\\\"
        )
    out.append(r"""\hline
\end{tabular}
\end{table}
""")
    return True


# --------------------------------------------------------------------------
# Table: cross-session transfer
# --------------------------------------------------------------------------

def table_crosssession(d: dict, paired, fmt_p, esc, out: list[str]) -> bool:
    ref, sen = d["cs_ref"], d["cs_sen"]
    if ref is None:
        return False
    from scipy.stats import wilcoxon
    order = ref.mean().sort_values(ascending=False).index

    out.append(r"""
%% ----------------------------------------------- Table: cross-session
\begin{table}[htbp]
\caption{\label{tab:crosssession}Cross-session transfer within subject on
IV-2a. For each subject and in both directions, each model is trained on one
recording session and tested on the other; the two directions are averaged
per subject before any statistic is computed, so $n$ is the number of
subjects. Hyperparameters are chosen by stratified inner cross-validation
inside the training session alone. In the reference frame each session is
whitened by its own Fr\'echet mean, and for the test session that mean is
computed from all of its trials. This uses test data, so it is stated here as
well as in the text: it uses no labels, and it is exactly the unsupervised
adaptation a deployed decoder can perform on the unlabelled data it receives
before decoding anything. $\Delta$ is the gain from the
reference frame, paired by subject and tested by Wilcoxon signed-rank.}
\begin{tabular}{@{}lccccc@{}}
\hline
Pipeline & Sensor & Reference & $\Delta$ & $p$ & Better \\
\hline""")
    for pipe in order:
        if pipe not in sen.columns:
            continue
        x, y = ref[pipe], sen[pipe]
        m = x.notna() & y.notna()
        diff = (x[m] - y[m]).to_numpy()
        p = float(wilcoxon(diff).pvalue)
        better = f"{int((diff > 0).sum())}/{len(diff)}"
        if (diff > 0).all():
            better = f"\\textbf{{{better}}}"
        out.append(
            f"{esc(pipe.split(chr(47), 1)[-1])} & {y.mean():.3f} & {x.mean():.3f} & "
            f"${diff.mean():+.4f}$ & {fmt_p(p)} & {better} \\\\"
        )
    out.append(r"""\hline
\end{tabular}
\end{table}
""")
    return True


# --------------------------------------------------------------------------
# Table: the decisive comparison under three outer-CV seeds
# --------------------------------------------------------------------------

def _tost_bound(dvec, alpha=0.05):
    """Smallest equivalence margin at which paired TOST would pass.

    Duplicates `qeeg.equivalence.equivalence_bound` so that make_tables does
    not need the package on its path.
    """
    from scipy.stats import t as tdist
    dvec = np.asarray(dvec, float)
    n = len(dvec)
    se = dvec.std(ddof=1) / np.sqrt(n)
    half = float(tdist.ppf(1 - alpha, n - 1) * se)
    return max(abs(dvec.mean() - half), abs(dvec.mean() + half))


def table_seeds(d: dict, paired, fmt_p, esc, out: list[str],
                margin: float = 0.02) -> bool:
    seeds = d.get("seeds_per", {})
    if len(seeds) < 2:
        return False

    out.append(r"""
%% ------------------------------------------------------ Table: seeds
\begin{table}[htbp]
\caption{\label{tab:seeds}The PhysioNet three-qubit comparison under three
different outer cross-validation partitions (seeds). Each row repeats the
full extended suite with new fold assignments; the inner search, the reference
state and every other element are unchanged. Columns give the frame effect
(range over the five kernels), the best reference-frame quantum kernel and
the classical twin, the quantum-minus-twin difference (range over the five
kernels), the smallest $p$ against the twin, and how many of the five
comparisons pass two one-sided tests at the pre-specified $\pm""" + f"{margin:g}" + r"""$
margin, with the largest bound. The frame effect and the uncontrolled
reversal reproduce in every partition; the twin comparison moves against the
bandwidth-parameterised quantum kernels under the other two partitions and
never in their favour (see text).}
\begin{tabular}{@{}lcccccc@{}}
\hline
Seed & Frame $\Delta$ & Quantum & Twin & Quantum $-$ twin & $\min p$ &
TOST \\
\hline""")
    for seed in sorted(seeds):
        per = seeds[seed]
        twin = _best_twin(per, TWINS)
        ks = [k for k in REF_KERNELS if k in per.columns]
        if twin is None or not ks:
            continue
        fr = [paired(per, b, a)["delta"] for a, b, _ in FRAME_PAIRS
              if a in per.columns and b in per.columns]
        st = {k: paired(per, k, twin) for k in ks}
        best = max(ks, key=lambda k: per[k].mean())
        deltas = [v["delta"] for v in st.values()]
        bounds = [_tost_bound((per[k] - per[twin]).dropna()) for k in ks]
        npass = sum(b < margin for b in bounds)
        out.append(
            f"{seed} & $[{min(fr):+.3f}, {max(fr):+.3f}]$ & "
            f"{per[best].mean():.3f} & {per[twin].mean():.3f} & "
            f"$[{min(deltas):+.4f}, {max(deltas):+.4f}]$ & "
            f"{fmt_p(min(v['p'] for v in st.values()))} & "
            f"{npass}/{len(ks)} ($\\le{max(bounds):.3f}$) \\\\"
        )
    out.append(r"""\hline
\end{tabular}
\end{table}
""")
    return True


# --------------------------------------------------------------------------
# Table: the register sweep in both frames
# --------------------------------------------------------------------------

SWEEP_LABEL = {"HS-overlap": "HS overlap", "Fidelity": "Fidelity",
               "Bures-d2": "Bures", "QRE": "QRE"}


def table_sweep(d: dict, out: list[str]) -> bool:
    sw = d["sweep"]
    if sw is None:
        return False
    sw = sw.sort_values(["qubits", "kernel"])
    base = sw[sw.qubits == sw.qubits.min()].set_index("kernel")

    out.append(r"""
%% ------------------------------------------------------ Table: sweep
\begin{table}[htbp]
\caption{\label{tab:sweep}Off-diagonal Gram variance versus register size in
both frames, $n=""" + f"{int(sw.n.iloc[0])}" + r"""$ subjects, PhysioNet.
Larger is better: a kernel whose Gram variance collapses cannot separate
trials. ``Gain'' is reference over sensor at the same register; ``vs.\ 3\,q''
is the reference-frame variance relative to its own three-qubit value. In the
reference frame no kernel's variance falls below its three-qubit value at any
larger register, so the concentration that the sensor frame exhibits is not a
qubit-count effect once the frame is corrected.}
\begin{tabular}{@{}llcccc@{}}
\hline
Register & Kernel & Sensor ($\times10^{-3}$) & Reference ($\times10^{-3}$) &
Gain & vs.\ 3\,q \\
\hline""")
    prev = None
    for _, r in sw.iterrows():
        q = int(r.qubits)
        lead = f"{q}\\,q ({int(2 ** q)} ch)" if q != prev else ""
        if q != prev and prev is not None:
            out.append(r"\hline")
        prev = q
        ratio = r.ref_var / base.loc[r.kernel, "ref_var"]
        out.append(
            f"{lead} & {SWEEP_LABEL.get(r.kernel, r.kernel)} & "
            f"{1e3 * r.sensor_var:.2f} & {1e3 * r.ref_var:.2f} & "
            f"{r.gain:.1f}$\\times$ & {ratio:.2f} \\\\"
        )
    out.append(r"""\hline
\end{tabular}
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
\caption{\label{tab:shots}Accuracy under finite-shot estimation. Unlike every
other table, the reference state here is estimated from all of a subject's
trials, not the training split alone; it uses no labels, but these accuracies
are comparable only with one another, not with the nested cross-validation
results. $\mathrm{tr}(\rho\sigma)$ is the SWAP-test observable, so $S$ shots give an
unbiased estimate with variance $(1-k^2)/S$; we sample each unordered pair
binomially and project the Gram matrix back to the positive semi-definite cone.
The reference frame needs \emph{more} shots to approach its own ceiling,
because that ceiling is higher, but it dominates the sensor frame in absolute
terms from $10^4$ shots upwards, above which it exceeds what the sensor
frame achieves with unlimited shots.}
\begin{tabular}{@{}ll""" + "c" * (len(shot_levels) + 1) + r"""@{}}
\hline
Kernel & Frame & """ + " & ".join(
        f"$10^{{{int(round(np.log10(s)))}}}$" for s in shot_levels
    ) + r""" & $\infty$ \\
\hline""")
    for kern in sorted(sh.kernel.unique()):
        for frame in ("sensor", "reference"):
            row = piv[(piv.kernel == kern)]
            cells = []
            for s in shot_levels + [-1]:
                v = row[row.shots == s][frame]
                cells.append(f"{float(v.iloc[0]):.3f}" if len(v) else "n/a")
            cells[-1] = f"\\textbf{{{cells[-1]}}}"
            out.append(f"{kern} & {frame.capitalize()} & " + " & ".join(cells) + r" \\")
    out.append(r"""\hline
\end{tabular}
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
            defs["TwinMinP"] = fmt_p_eq(min(v["p"] for v in st.values()))
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
            defs["HeadSensorP"] = fmt_p_eq(s1["p"])
            defs["HeadSensorBetter"] = f"{s1['n_better']}/{s1['n']}"
            defs["HeadRefDelta"] = f"{s2['delta']:+.4f}"
            defs["HeadRefLead"] = f"{abs(s2['delta']):.4f}"
            defs["HeadRefP"] = fmt_p_eq(s2["p"])
            defs["HeadRefBetter"] = f"{s2['n_better']}/{s2['n']}"
            defs["HeadRefKernel"] = esc(bq_r)

    if d["bci_per"] is not None:
        frame_stats(d["bci_per"], FRAME_PAIRS, "Bci")

    def twin_stats(per, kernels, twins, prefix):
        twin = _best_twin(per, twins)
        ks = [k for k in kernels if k in per.columns]
        if twin is None or not ks:
            return
        st = {k: paired(per, k, twin) for k in ks}
        best = max(ks, key=lambda k: per[k].mean())
        defs[prefix + "TwinName"] = esc(twin)
        defs[prefix + "TwinAcc"] = f"{per[twin].mean():.3f}"
        defs[prefix + "BestKernel"] = esc(best)
        defs[prefix + "BestAcc"] = f"{per[best].mean():.3f}"
        defs[prefix + "TwinDeltaMin"] = f"{min(v['delta'] for v in st.values()):+.4f}"
        defs[prefix + "TwinDeltaMax"] = f"{max(v['delta'] for v in st.values()):+.4f}"
        defs[prefix + "TwinMinP"] = fmt_p_eq(min(v["p"] for v in st.values()))
        defs[prefix + "NSubjects"] = f"{len(per)}"
        # The kernel that leads the twin by most, with its own test, so the
        # prose can name it without hand-typing anything.
        top = max(ks, key=lambda k: st[k]["delta"])
        defs[prefix + "TwinTopKernel"] = esc(top)
        defs[prefix + "TwinTopDelta"] = f"{st[top]['delta']:+.4f}"
        defs[prefix + "TwinTopP"] = fmt_p_eq(st[top]["p"])
        defs[prefix + "TwinTopBetter"] = f"{st[top]['n_better']}/{st[top]['n']}"
        others = [k for k in ks if k != top]
        if others:
            defs[prefix + "TwinRestDeltaMin"] = f"{min(st[k]['delta'] for k in others):+.4f}"
            defs[prefix + "TwinRestDeltaMax"] = f"{max(st[k]['delta'] for k in others):+.4f}"
            defs[prefix + "TwinRestMinP"] = fmt_p_eq(min(st[k]["p"] for k in others))
        cl = [c for c in per.columns if c.startswith("classical/")]
        if cl:
            bc = per[cl].mean().idxmax()
            s = paired(per, best, bc)
            defs[prefix + "BestClassical"] = esc(bc)
            defs[prefix + "BestClassicalAcc"] = f"{per[bc].mean():.3f}"
            defs[prefix + "HeadRefDelta"] = f"{s['delta']:+.4f}"
            defs[prefix + "HeadRefP"] = fmt_p_eq(s["p"])
            defs[prefix + "HeadRefBetter"] = f"{s['n_better']}/{s['n']}"
            # The same comparison for the classical twin: if the twin leads the
            # best classical baseline by as much as the quantum kernel does,
            # the lead is the frame and the kernel formulation, not the metric.
            t = paired(per, twin, bc)
            defs[prefix + "TwinHeadDelta"] = f"{t['delta']:+.4f}"
            defs[prefix + "TwinHeadP"] = fmt_p_eq(t["p"])
            defs[prefix + "TwinHeadBetter"] = f"{t['n_better']}/{t['n']}"

    # 16 channels = 4 qubits, same suite, same pipeline names.
    if d["m16_per"] is not None:
        frame_stats(d["m16_per"], FRAME_PAIRS, "FourQ")
        twin_stats(d["m16_per"], REF_KERNELS, TWINS, "FourQ")

    # Cho2017: the better-powered replication of the frame/twin comparison.
    if d["cho_per"] is not None:
        frame_stats(d["cho_per"], FRAME_PAIRS, "Cho")
        twin_stats(d["cho_per"], REF_KERNELS, TWINS, "Cho")
        # Trials per subject come from the batch metadata the merge was built
        # from; the merged meta records only the file list.
        import glob
        import json
        counts = {}
        for f in glob.glob(str(Path(d["_res"]) / "meta_cho_batch*.json")):
            counts.update(json.load(open(f, encoding="utf-8"))
                          .get("n_trials_per_subject", {}))
        if counts:
            vals = sorted(set(int(v) for v in counts.values()))
            defs["ChoNTrialsMin"] = f"{vals[0]}"
            defs["ChoNTrialsMax"] = f"{vals[-1]}"
            defs["ChoNTrialsMaxCount"] = f"{sum(int(v) == vals[-1] for v in counts.values())}"
        # Sensor-frame position of the best quantum kernel against the best
        # classical baseline on this dataset: the "classical wins in the sensor
        # frame" statement has to hold here too before the abstract says it.
        per = d["cho_per"]
        sq = [a for a, _, _ in FRAME_PAIRS if a in per.columns]
        cl = [c for c in per.columns if c.startswith("classical/")]
        if sq and cl:
            bq, bc = per[sq].mean().idxmax(), per[cl].mean().idxmax()
            s = paired(per, bc, bq)
            defs["ChoHeadSensorDelta"] = f"{s['delta']:+.4f}"
            defs["ChoHeadSensorP"] = fmt_p_eq(s["p"])
            defs["ChoSensorBestQuantum"] = esc(bq)

    # Seeds: does the decisive comparison depend on the fold partition?
    seeds = d.get("seeds_per", {})
    if len(seeds) >= 2:
        fr_all, dl_all, p_all, bounds, heads = [], [], [], [], []
        # Per-family bookkeeping: the bandwidth-parameterised kernels (an extra
        # hyperparameter tuned on 45 trials) behave differently from the
        # parameter-free overlap kernels under resampling, and the prose says so.
        rbf, ovl, sig = [], [], []
        for seed, per in sorted(seeds.items()):
            twin = _best_twin(per, TWINS)
            ks = [k for k in REF_KERNELS if k in per.columns]
            if twin is None or not ks:
                continue
            fr_all += [paired(per, b, a)["delta"] for a, b, _ in FRAME_PAIRS
                       if a in per.columns and b in per.columns]
            st = {k: paired(per, k, twin) for k in ks}
            dl_all += [v["delta"] for v in st.values()]
            p_all += [v["p"] for v in st.values()]
            bounds += [_tost_bound((per[k] - per[twin]).dropna()) for k in ks]
            for k, v in st.items():
                (rbf if "RBF" in k else ovl).append((v["delta"], v["p"]))
                if v["p"] < 0.05:
                    sig.append(v["delta"])
            cl = [c for c in per.columns if c.startswith("classical/")]
            if cl:
                bc = per[cl].mean().idxmax()
                bq = max(ks, key=lambda k: per[k].mean())
                heads.append(paired(per, bq, bc)["delta"])
        defs["SeedRbfWorst"] = f"{min(x for x, _ in rbf):+.4f}"
        defs["SeedRbfMinP"] = fmt_p_eq(min(p for _, p in rbf))
        defs["SeedOverlapDeltaMin"] = f"{min(x for x, _ in ovl):+.4f}"
        defs["SeedOverlapDeltaMax"] = f"{max(x for x, _ in ovl):+.4f}"
        defs["SeedOverlapMinP"] = fmt_p_eq(min(p for _, p in ovl))
        defs["SeedSigCount"] = f"{len(sig)}"
        defs["SeedSigN"] = f"{len(rbf) + len(ovl)}"
        defs["SeedSigFavourQuantum"] = f"{sum(x > 0 for x in sig)}"
        defs["SeedSigFavourTwin"] = f"{sum(x < 0 for x in sig)}"
        defs["SeedN"] = f"{len(seeds)}"
        defs["SeedFrameMin"] = f"{min(fr_all):+.3f}"
        defs["SeedFrameMax"] = f"{max(fr_all):+.3f}"
        defs["SeedTwinDeltaMin"] = f"{min(dl_all):+.4f}"
        defs["SeedTwinDeltaMax"] = f"{max(dl_all):+.4f}"
        defs["SeedTwinMinP"] = fmt_p_eq(min(p_all))
        defs["SeedEquivPass"] = f"{sum(b < 0.02 for b in bounds)}"
        defs["SeedEquivN"] = f"{len(bounds)}"
        defs["SeedWorstBound"] = f"{max(bounds):.3f}"
        if heads:
            defs["SeedHeadRefMin"] = f"{min(heads):+.4f}"
            defs["SeedHeadRefMax"] = f"{max(heads):+.4f}"
            defs["SeedHeadReversals"] = f"{sum(h > 0 for h in heads)}"

    # Wall-clock cost in the reference frame, where the accuracy comparison is
    # actually made. The core-suite cost macros describe the sensor frame.
    for prefix, summ in (("Phys", d["phys_summary"]), ("Bci", d["bci_summary"])):
        if summ is None:
            continue
        sec = summ.set_index("pipeline")["sec_per_subject"]
        twin = "control/riemann-kernel-SVM"
        slow = max(REF_KERNELS, key=lambda k: sec.get(k, 0.0))
        if twin in sec and slow in sec:
            defs[prefix + "CostTwinSec"] = f"{sec[twin]:.1f}"
            defs[prefix + "CostSlowName"] = esc(slow)
            defs[prefix + "CostSlowSec"] = f"{sec[slow]:.1f}"
            defs[prefix + "CostRatio"] = f"{sec[slow] / sec[twin]:.1f}"

    if d["fb_per"] is not None:
        per = d["fb_per"]
        frame_stats(per, FB_FRAME_PAIRS, "Fb")
        ks = [k for k in FB_REF_KERNELS if k in per.columns]
        if "classical/FBCSP+LDA" in per.columns and ks:
            bq = per[ks].mean().idxmax()
            s = paired(per, bq, "classical/FBCSP+LDA")
            defs["FbcspDelta"] = f"{s['delta']:+.4f}"
            defs["FbcspP"] = fmt_p_eq(s["p"])
            defs["FbcspBestAcc"] = f"{per[bq].mean():.3f}"
            defs["FbcspAcc"] = f"{per['classical/FBCSP+LDA'].mean():.3f}"

    if d["transfer_ref"] is not None:
        ref = d["transfer_ref"]
        defs["TransferN"] = f"{len(ref)}"
        defs["TransferSpread"] = f"{ref.mean().max() - ref.mean().min():.4f}"
        defs["TransferBest"] = esc(ref.mean().idxmax())
        defs["TransferBestAcc"] = f"{ref.mean().max():.3f}"
        # Wilcoxon's two-sided floor: the test cannot return a smaller p.
        defs["TransferFloor"] = _sci(2.0 ** (1 - len(ref)))

    if d["cs_ref"] is not None:
        ref, sen = d["cs_ref"], d["cs_sen"]
        from scipy.stats import wilcoxon
        qk = [k for k in TRANSFER_KERNELS if k in ref.columns]
        cl = [c for c in ref.columns if c.startswith("classical/")]
        defs["CsN"] = f"{len(ref)}"
        defs["CsFloor"] = f"{2.0 ** (1 - len(ref)):.1e}".replace("e-0", r"\times10^{-") + "}"
        if "n_train" in d["cs"].columns:
            # Sessions are equal-sized on IV-2a, so one number describes both.
            defs["CsTrialsPerSession"] = f"{int(d['cs'].n_train.mode().iloc[0])}"
        if qk:
            fr = {k: (ref[k] - sen[k]).dropna() for k in qk}
            defs["CsFrameQMin"] = f"{min(v.mean() for v in fr.values()):+.3f}"
            defs["CsFrameQMax"] = f"{max(v.mean() for v in fr.values()):+.3f}"
            defs["CsFrameQMaxP"] = fmt_p(max(float(wilcoxon(v).pvalue) for v in fr.values()))
            defs["CsFrameQAllBetter"] = (
                "yes" if all((v > 0).all() for v in fr.values()) else "no")
            defs["CsSensorQMin"] = f"{min(sen[k].mean() for k in qk):.3f}"
            defs["CsSensorQMax"] = f"{max(sen[k].mean() for k in qk):.3f}"
            defs["CsRefQMin"] = f"{min(ref[k].mean() for k in qk):.3f}"
            defs["CsRefQMax"] = f"{max(ref[k].mean() for k in qk):.3f}"
            bq = max(qk, key=lambda k: ref[k].mean())
            defs["CsBestKernel"] = esc(bq)
            defs["CsBestAcc"] = f"{ref[bq].mean():.3f}"
        if cl:
            fc = {c: (ref[c] - sen[c]).dropna() for c in cl}
            defs["CsFrameClMin"] = f"{min(v.mean() for v in fc.values()):+.3f}"
            defs["CsFrameClMax"] = f"{max(v.mean() for v in fc.values()):+.3f}"
            defs["CsSensorClMin"] = f"{min(sen[c].mean() for c in cl):.3f}"
            defs["CsSensorClMax"] = f"{max(sen[c].mean() for c in cl):.3f}"
            bc = max(cl, key=lambda c: ref[c].mean())
            defs["CsBestClassical"] = esc(bc)
            defs["CsBestClassicalAcc"] = f"{ref[bc].mean():.3f}"
        twin = _best_twin(ref, TWINS)
        if twin and qk:
            st = {k: paired(ref, k, twin) for k in qk}
            defs["CsTwinName"] = esc(twin)
            defs["CsTwinAcc"] = f"{ref[twin].mean():.3f}"
            defs["CsTwinDeltaMin"] = f"{min(v['delta'] for v in st.values()):+.4f}"
            defs["CsTwinDeltaMax"] = f"{max(v['delta'] for v in st.values()):+.4f}"
            defs["CsTwinMinP"] = fmt_p_eq(min(v["p"] for v in st.values()))
            defs["CsTwinAbsMax"] = f"{max(abs(v['delta']) for v in st.values()):.4f}"
        geo = qk + [t for t in TWINS if t in ref.columns]
        if geo:
            defs["CsSpread"] = f"{ref[geo].mean().max() - ref[geo].mean().min():.4f}"

    if d["sweep"] is not None:
        sw = d["sweep"]
        lo, hi = int(sw.qubits.min()), int(sw.qubits.max())
        a, b = sw[sw.qubits == lo].set_index("kernel"), sw[sw.qubits == hi].set_index("kernel")
        ratio = b.ref_var / a.ref_var
        sratio = b.sensor_var / a.sensor_var
        defs["SweepQMin"], defs["SweepQMax"] = f"{lo}", f"{hi}"
        defs["SweepN"] = f"{int(sw.n.iloc[0])}"
        defs["SweepRefRatioMin"] = f"{ratio.min():.1f}"
        defs["SweepRefRatioMax"] = f"{ratio.max():.1f}"
        defs["SweepSensorRatioMin"] = f"{sratio.min():.1f}"
        defs["SweepSensorRatioMax"] = f"{sratio.max():.1f}"
        defs["SweepGainLoMin"] = f"{a.gain.min():.1f}"
        defs["SweepGainLoMax"] = f"{a.gain.max():.1f}"
        defs["SweepGainHiMin"] = f"{b.gain.min():.1f}"
        defs["SweepGainHiMax"] = f"{b.gain.max():.1f}"
        # Does the reference-frame variance stay above its smallest-register
        # value at every larger register, for every kernel? (It is not strictly
        # monotone: three of four kernels dip slightly from 5 to 6 qubits.)
        above = all((sw[(sw.kernel == k) & (sw.qubits > lo)].ref_var
                     >= a.loc[k, "ref_var"]).all() for k in sw.kernel.unique())
        defs["SweepRefAboveBase"] = "every" if above else "not every"
        mono = all(sw[sw.kernel == k].sort_values("qubits").ref_var.is_monotonic_increasing
                   for k in sw.kernel.unique())
        defs["SweepRefMonotone"] = "every" if mono else "not every"
        # The largest 5q->6q dip in the reference frame, as a fraction.
        five = sw[sw.qubits == hi - 1].set_index("kernel")
        dip = (1 - b.ref_var / five.ref_var).clip(lower=0)
        defs["SweepRefDipMax"] = f"{100 * dip.max():.0f}"

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

    # The frame effect over every setting in which it was measured, for the
    # sentences that bound the equivalence margin against "the weakest effect
    # this paper claims as real".
    all_deltas = []
    for per, pairs in ((d["phys_per"], FRAME_PAIRS), (d["bci_per"], FRAME_PAIRS),
                       (d["fb_per"], FB_FRAME_PAIRS), (d["m16_per"], FRAME_PAIRS),
                       (d["cho_per"], FRAME_PAIRS)):
        if per is not None:
            all_deltas += [paired(per, b, a)["delta"] for a, b, _ in pairs
                           if a in per.columns and b in per.columns]
    for ref, sen in ((d["transfer_ref"], d["transfer_sen"]),
                     (d["cs_ref"], d["cs_sen"])):
        if ref is not None:
            all_deltas += [float((ref[k] - sen[k]).dropna().mean())
                           for k in TRANSFER_KERNELS if k in ref.columns]
    if all_deltas:
        defs["FrameEffectMin"] = f"{min(all_deltas):+.3f}"
        defs["FrameEffectMax"] = f"{max(all_deltas):+.3f}"

    if defs:
        out.append("\n%% ------------------------------ reference-frame macros\n")
        for k, v in defs.items():
            out.append(f"\\newcommand{{\\{k}}}{{{v}}}")
        out.append("")
