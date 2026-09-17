"""Second-dataset and cross-dataset LaTeX tables for the manuscript.

Imported by make_tables.py. Kept separate because the single-dataset tables
were written first and this keeps that file readable.

Two statistical points are baked in here so they cannot be forgotten when the
numbers are regenerated:

*Rank-test floor.* With n pairs the two-sided Wilcoxon signed-rank test cannot
return a p below 2^(1-n). At n=9 that is 0.0039, so a Holm correction over 14
comparisons cannot fall below 0.0547 whatever the effect size. Entries sitting
at the floor are daggered in the table and the caption says what that means.

*Combining evidence.* Fisher's method pools the two independent per-dataset
p-values, which is the right way to report an effect that is consistent across
datasets but underpowered on each.
"""

from __future__ import annotations


def fmt_p_eq(p):
    """Relation-carrying p-value; see make_tables.fmt_p_eq."""
    rel = "<" if p < 0.001 else "="
    val = "0.001" if p < 0.001 else f"{p:.3f}"
    return r"\ensuremath{{}" + rel + r"{}}" + val

import numpy as np
import pandas as pd
from scipy.stats import chi2

BCI_TAG = "bci2a_motor8_q4"

# (dataset label, per-subject accuracy table) pairs are supplied by the caller.
DENSITY_PIPE = "quantum/Fidelity-SVM"
BEST_CLASSICAL = "classical/CSP+LDA"


def wilcoxon_floor(n: int) -> float:
    """Smallest two-sided p the signed-rank test can return with n pairs."""
    return 2.0 ** (1 - n)


def fisher(pvals) -> tuple[float, float]:
    """Fisher's method for combining independent p-values."""
    pv = list(pvals)
    stat = -2.0 * float(np.sum(np.log(pv)))
    return stat, float(chi2.sf(stat, 2 * len(pv)))


NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
             "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
             "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]


def bottom_quantum(summary_p, summary_b) -> tuple[int, int]:
    """How many of the lowest positions the same quantum kernels hold on both.

    Returns (k, shift): the largest k for which the k lowest pipelines are
    quantum on both datasets and are the same set, and the largest rank move
    of any of them between the datasets. "The four lowest positions ... with
    zero rank shift" was typed by hand; with QRE-RBF in both core suites it is
    the five lowest, and two of them swap places.
    """
    op = list(summary_p.sort_values("acc_mean")["pipeline"])
    ob = list(summary_b.sort_values("acc_mean")["pipeline"])
    grp = dict(zip(summary_p.pipeline, summary_p.group))
    grp.update(dict(zip(summary_b.pipeline, summary_b.group)))
    k = 0
    for i in range(1, min(len(op), len(ob)) + 1):
        if (set(op[:i]) == set(ob[:i])
                and all(grp.get(p) == "quantum" for p in op[:i])):
            k = i
    shift = max((abs(op.index(p) - ob.index(p)) for p in op[:k]), default=0)
    return k, shift


def table_bci(summary_b, summary_p, out, esc, group_label) -> None:
    """Accuracy on IV-2a with the PhysioNet figure alongside."""
    pa = summary_p.set_index("pipeline")["acc_mean"]
    k, _ = bottom_quantum(summary_p, summary_b)
    n_b = int(summary_b["n_subjects"].max())
    lowest = (f"Quantum kernels occupy the {NUM_WORDS[k]}\n"
              "lowest positions on both datasets." if k else "")
    out.append(
        "\n%% ---------------------------------------------------------------- Table 5\n"
        r"\begin{table}[htbp]" "\n"
        r"\caption{\label{tab:bci}Replication on BCI Competition IV-2a "
        f"({n_b} subjects, 288 trials each), with the PhysioNet result repeated for\n"
        "comparison. Protocol, channels, tuning budget and pipelines are\n"
        f"identical; only the data differs. {lowest}}}\n"
        r"" "\n"
        # Group is its own column rather than a prefix on every name. It was
        # dropped once because the prefixed names overflowed the text block;
        # without the prefix they fit, and the table stops repeating the word
        # "quantum" sixteen times down its first column.
        r"\begin{tabular}{@{}llccc@{}}" "\n"
        r"\hline" "\n"
        r"Pipeline & Group & Acc (IV-2a) & AUC (IV-2a) & Acc (Phys.) \\" "\n"
        r"\hline"
    )
    top = summary_b["acc_mean"].max()
    for _, r in summary_b.sort_values("acc_mean", ascending=False).iterrows():
        name = r["pipeline"]
        acc = (r"\textbf{" + f"{r['acc_mean']:.3f}" + "}"
               if r["acc_mean"] == top else f"{r['acc_mean']:.3f}")
        ref = pa.get(name, float("nan"))
        out.append(
            f"{esc(name)} & {group_label.get(r['group'], r['group'])} & {acc} & "
            f"{r['auc_mean']:.3f} & {ref:.3f} " + r"\\"
        )
    out.append(r"\hline" "\n" r"\end{tabular}" "\n"        r"\end{table}" "\n")


def table_cross(per_p, per_b, comparisons, paired, fmt_p, out) -> None:
    """Pre-specified comparisons on both datasets, plus Fisher combination."""
    n_b = len(per_b)
    floor = wilcoxon_floor(n_b)
    primary = [paired(per_b, a, b) for a, b, _ in comparisons[:1]
               if a in per_b.columns and b in per_b.columns]
    dens = (paired(per_b, BEST_CLASSICAL, DENSITY_PIPE)
            if BEST_CLASSICAL in per_b.columns and DENSITY_PIPE in per_b.columns
            else None)
    held = [r for r in primary + ([dens] if dens else []) if r["n_better"] == r["n"]]
    every = (f" Both classical-versus-quantum contrasts hold in every one of the "
             f"{n_b} IV-2a subjects." if len(held) == 2 else "")
    out.append(
        "\n%% ---------------------------------------------------------------- Table 6\n"
        r"\begin{table}[htbp]" "\n"
        r"\caption{\label{tab:cross}Pre-specified comparisons on both datasets, "
        "combined by\n"
        r"Fisher's method. Positive $\Delta$ favours the first-named pipeline. "
        f"At $n={n_b}$ the\n"
        f"two-sided signed-rank test cannot return $p<{floor:.4f}$, so the IV-2a "
        "entries marked\n"
        r"$\dagger$ sit at that floor: the test is saturated rather than merely "
        "significant."
        + every + "}\n"
        r"\begin{tabular}{@{}llccc@{}}" "\n"
        r"\hline" "\n"
        r"Comparison & Dataset & $\Delta$ acc & $p$ & Better \\" "\n"
        r"\hline"
    )
    for a, b, label in comparisons:
        ps = []
        first = True
        for tag, per in (("PhysioNet", per_p), ("IV-2a", per_b)):
            if a not in per.columns or b not in per.columns:
                continue
            r = paired(per, a, b)
            ps.append(r["p"])
            dag = (r"$^\dagger$"
                   if np.isclose(r["p"], wilcoxon_floor(r["n"])) else "")
            shown = label if first else ""
            first = False
            out.append(
                f"{shown} & {tag} & ${r['delta']:+.4f}$ & {fmt_p(r['p'])}{dag} & "
                f"{r['n_better']}/{r['n']} " + r"\\"
            )
        if len(ps) == 2:
            _, pc = fisher(ps)
            cell = (r"\textbf{" + fmt_p(pc) + "}") if pc < 0.05 else fmt_p(pc)
            out.append(r" & \textit{Fisher combined} & & " + cell + r" & \\")
        out.append(r"\noalign{\smallskip}")
    out.append(r"\hline" "\n" r"\end{tabular}" "\n"        r"\end{table}" "\n")


def cross_macros(summary_p, summary_b, per_p, per_b, df_b,
                 comparisons, paired, fmt_p, out) -> None:
    """Inline numbers for the cross-dataset prose."""
    def gap(s):
        bc = s[s.group == "classical"]["acc_mean"].max()
        bq = s[s.group == "quantum"]["acc_mean"].max()
        wq = s[s.group == "quantum"]["acc_mean"].min()
        return bc, bq, bc - bq, bc - wq

    bc_p, bq_p, g_p, gw_p = gap(summary_p)
    bc_b, bq_b, g_b, gw_b = gap(summary_b)

    rp = summary_p.set_index("pipeline")["acc_mean"].rank(ascending=False)
    rb = summary_b.set_index("pipeline")["acc_mean"].rank(ascending=False)
    common = [i for i in rp.index if i in rb.index]
    rho = float(rp[common].corr(rb[common], method="spearman"))

    prim_p = paired(per_p, *comparisons[0][:2])
    prim_b = paired(per_b, *comparisons[0][:2])
    dens_p = paired(per_p, BEST_CLASSICAL, DENSITY_PIPE)
    dens_b = paired(per_b, BEST_CLASSICAL, DENSITY_PIPE)
    abl_p = paired(per_p, *comparisons[1][:2])
    abl_b = paired(per_b, *comparisons[1][:2])

    n_b = int(prim_b["n"])
    defs = {
        "BciNSubjects": str(int(df_b.subject.nunique())),
        "BciNTrials": "288",
        "BciNFoldScores": f"{len(df_b):,}".replace(",", r"\,"),
        "BciBestClassicalAcc": f"{bc_b:.3f}",
        "BciBestQuantumAcc": f"{bq_b:.3f}",
        "BciWorstQuantumAcc": f"{summary_b[summary_b.group == 'quantum']['acc_mean'].min():.3f}",
        "GapPhysio": f"{g_p:.3f}",
        "GapBci": f"{g_b:.3f}",
        "GapWorstPhysio": f"{gw_p:.3f}",
        "GapWorstBci": f"{gw_b:.3f}",
        "SpearmanRho": f"{rho:.3f}",
        "BciPrimaryDelta": f"{prim_b['delta']:+.3f}",
        "BciPrimaryP": fmt_p_eq(prim_b["p"]),
        "BciPrimaryDz": f"{prim_b['dz']:.2f}",
        "BciPrimaryBetter": f"{prim_b['n_better']}/{n_b}",
        "BciDensDelta": f"{dens_b['delta']:+.3f}",
        "BciDensDz": f"{dens_b['dz']:.2f}",
        "BciDensBetter": f"{dens_b['n_better']}/{n_b}",
        "FisherPrimaryP": fmt_p_eq(fisher([prim_p["p"], prim_b["p"]])[1]),
        "FisherDensP": fmt_p_eq(fisher([dens_p["p"], dens_b["p"]])[1]),
        "FisherAblationP": fmt_p_eq(fisher([abl_p["p"], abl_b["p"]])[1]),
        "BciWilcoxonFloor": f"{wilcoxon_floor(n_b):.4f}",
    }
    # The IV-2a correction family: every pipeline against the reference. Its
    # size was typed as 14, and the sentence around it quoted PhysioNet's
    # family size and "ten of the fourteen" by hand, all of which moved when
    # the IV-2a core suite gained its sixteenth pipeline.
    # What the six-fold increase in trials was worth to each family. The prose
    # said "roughly 0.15" and "about 0.05"; at n=104 the ranges are narrower
    # than either figure suggests, so they are quoted as ranges.
    acc_p = summary_p.set_index("pipeline")["acc_mean"]
    acc_b = summary_b.set_index("pipeline")["acc_mean"]
    gain = (acc_b - acc_p).dropna()
    cls = [p for p in gain.index if p.startswith("classical/")]
    dens = [p for p in gain.index if p.startswith("quantum/")
            and "kernel-SVM" not in p]          # not the IQP and CNOT circuits
    if cls and dens:
        defs["GainClassicalMin"] = f"{gain[cls].min():+.3f}"
        defs["GainClassicalMax"] = f"{gain[cls].max():+.3f}"
        defs["GainDensityMin"] = f"{gain[dens].min():+.3f}"
        defs["GainDensityMax"] = f"{gain[dens].max():+.3f}"

    bottom6 = summary_p.sort_values("acc_mean").head(6)
    defs["PhysBottomSixQuantum"] = NUM_WORDS[int((bottom6.group == "quantum").sum())]
    k_low, shift = bottom_quantum(summary_p, summary_b)
    defs["BottomQuantumN"] = NUM_WORDS[k_low]
    defs["BottomQuantumShift"] = (
        "in the same order" if shift == 0 else
        "no kernel moving more than one place" if shift == 1 else
        f"no kernel moving more than {NUM_WORDS[shift]} places")

    ref = "classical/TS+LR"
    fam = [c for c in per_b.columns if c != ref]
    fam_p = [paired(per_b, c, ref)["p"] for c in fam]
    floor = wilcoxon_floor(n_b)
    at_floor = sum(bool(np.isclose(x, floor)) for x in fam_p)
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven",
             "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
             "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
    defs["BciNFamilyTests"] = words[len(fam)] if len(fam) < len(words) else str(len(fam))
    defs["BciNAtFloor"] = (words[at_floor] if at_floor < len(words)
                           else str(at_floor)).capitalize()
    defs["BciHolmFloor"] = f"{min(1.0, floor * len(fam)):.4f}"
    out.append("\n%% -------------------------------- cross-dataset macros\n")
    for k, v in defs.items():
        out.append("\\newcommand{\\" + k + "}{" + v + "}")
    out.append("")
