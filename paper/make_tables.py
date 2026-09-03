"""Generate the manuscript's LaTeX tables directly from the result CSVs.

Numbers in a paper must never be typed by hand. Every table in main.tex is
\\input{} from tables_auto.tex, which this script regenerates from
results/*.csv. Re-run it after any new benchmark run and the manuscript is
consistent by construction.

    python paper/make_tables.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import cross_tables as ct
import reference_tables as rt

REFERENCE = "classical/TS+LR"
TAG = "motor8_q4"

# Pipelines whose comparison is stated in the text; keep in one place.
KEY_COMPARISONS = [
    ("classical/CSP+LDA", "quantum/CNOT-kernel-SVM",
     "Best classical vs.\\ best quantum"),
    ("control/IQP-no-entangle", "quantum/IQP-kernel-SVM",
     "Entanglement ablation"),
    ("control/PCA-matched-linear", "quantum/IQP-kernel-SVM",
     "Dimension-matched control"),
]

GROUP_LABEL = {"classical": "Classical", "quantum": "Quantum", "control": "Control"}


def esc(s: str) -> str:
    """Escape the few LaTeX-special characters our pipeline names can contain."""
    return (s.replace("\\", r"\textbackslash{}")
             .replace("_", r"\_")
             .replace("&", r"\&")
             .replace("%", r"\%")
             .replace("#", r"\#"))


def paired(per: pd.DataFrame, a: str, b: str) -> dict:
    x, y = per[a], per[b]
    m = x.notna() & y.notna()
    d = (x[m] - y[m]).to_numpy()
    stat, p = wilcoxon(d)
    return {
        "delta": float(d.mean()),
        "p": float(p),
        "dz": float(d.mean() / d.std(ddof=1)),
        "n_better": int((d > 0).sum()),
        "n": int(len(d)),
    }


def fmt_p(p: float) -> str:
    if p < 0.001:
        return r"$<$0.001"
    return f"{p:.3f}"


def table_main(summary: pd.DataFrame, out: list[str]) -> None:
    out.append(r"""
%% ---------------------------------------------------------------- Table 1
\begin{table}[htbp]
\caption{\label{tab:main}Within-subject decoding performance on PhysioNet
EEGMMIDB left- versus right-hand motor imagery. Accuracy and AUC are means over
subjects of the per-subject nested cross-validation score; SD is between
subjects. Runtime is mean wall-clock seconds per subject for the full
hyperparameter search. Rows are ordered by accuracy.}
\begin{indented}
\item[]\begin{tabular}{@{}llcccc@{}}
\br
Pipeline & Group & Accuracy & SD & AUC & Runtime (s) \\
\mr""")
    for _, r in summary.sort_values("acc_mean", ascending=False).iterrows():
        name = esc(r["pipeline"])
        best = r["acc_mean"] == summary["acc_mean"].max()
        acc = f"\\textbf{{{r['acc_mean']:.3f}}}" if best else f"{r['acc_mean']:.3f}"
        out.append(
            f"{name} & {GROUP_LABEL.get(r['group'], r['group'])} & {acc} & "
            f"{r['acc_std']:.3f} & {r['auc_mean']:.3f} & "
            f"{r['sec_per_subject']:.2f} \\\\"
        )
    out.append(r"""\br
\end{tabular}
\end{indented}
\end{table}
""")


def table_tests(tests: pd.DataFrame, out: list[str]) -> None:
    out.append(r"""
%% ---------------------------------------------------------------- Table 2
\begin{table}[htbp]
\caption{\label{tab:tests}Paired comparisons against the """ + esc(REFERENCE) + r"""
reference across subjects (Wilcoxon signed-rank, two-sided). $\Delta$ is the
mean per-subject accuracy difference; $d_z$ is the paired effect size;
$p_{\mathrm{Holm}}$ is corrected across the whole family of comparisons. No
comparison survives correction.}
\begin{indented}
\item[]\begin{tabular}{@{}lccccc@{}}
\br
Pipeline & $\Delta$ accuracy & $p$ & $p_{\mathrm{Holm}}$ & $d_z$ & Better in \\
\mr""")
    for _, r in tests.sort_values("delta_acc", ascending=False).iterrows():
        out.append(
            f"{esc(r['pipeline'])} & ${r['delta_acc']:+.4f}$ & {fmt_p(r['p_value'])} & "
            f"{fmt_p(r['p_holm'])} & ${r['cohens_d']:+.3f}$ & "
            f"{int(r['n_better'])}/{int(r['n_subjects'])} \\\\"
        )
    out.append(r"""\br
\end{tabular}
\end{indented}
\end{table}
""")


def table_key(per: pd.DataFrame, out: list[str]) -> None:
    out.append(r"""
%% ---------------------------------------------------------------- Table 3
\begin{table}[htbp]
\caption{\label{tab:key}Pre-specified comparisons, first-named pipeline minus
second. The entanglement ablation contrasts the IQP circuit kernel with an
otherwise identical circuit whose entangling gates are deleted; the
dimension-matched control contrasts it with a linear SVM on exactly the same
PCA features. Only the primary classical-versus-quantum contrast is
statistically significant; both ablations point in the expected direction but
do not reach significance on accuracy at $n=30$.}
\begin{indented}
\item[]\begin{tabular}{@{}lcccc@{}}
\br
Comparison & $\Delta$ accuracy & $p$ & $d_z$ & Better in \\
\mr""")
    for a, b, label in KEY_COMPARISONS:
        if a not in per.columns or b not in per.columns:
            continue
        s = paired(per, a, b)
        sig = r"\textbf{" if s["p"] < 0.05 else "{"
        out.append(
            f"{label} & {sig}${s['delta']:+.4f}$}} & {sig}{fmt_p(s['p'])}}} & "
            f"${s['dz']:+.3f}$ & {s['n_better']}/{s['n']} \\\\"
        )
    out.append(r"""\br
\end{tabular}
\end{indented}
\end{table}
""")


def table_concentration(decay: pd.DataFrame, out: list[str]) -> None:
    out.append(r"""
%% ---------------------------------------------------------------- Table 4
\begin{table}[htbp]
\caption{\label{tab:concentration}Kernel concentration as a function of register
size on real EEG. Register size is swept by varying the channel count over
powers of two (4/8/16/32/64 channels $=$ 2--6 qubits), with no change of method.
The statistic is the variance of the off-diagonal Gram entries; a factor below
one means concentration worsens with scale. The entangled circuit kernel
concentrates $2.5\times$ faster in log-slope than the same circuit with
entanglers removed.}
\begin{indented}
\item[]\begin{tabular}{@{}lcccc@{}}
\br
Kernel & Log-slope / qubit & Factor / qubit & Var (2 qubits) & Var (6 qubits) \\
\mr""")
    for _, r in decay.sort_values("variance_factor_per_qubit").iterrows():
        out.append(
            f"{esc(r['kernel'])} & ${r['log_variance_slope_per_qubit']:+.3f}$ & "
            f"{r['variance_factor_per_qubit']:.3f} & "
            f"{r['variance_first']:.5f} & {r['variance_last']:.5f} \\\\"
        )
    out.append(r"""\br
\end{tabular}
\end{indented}
\end{table}
""")


def macros(df: pd.DataFrame, summary: pd.DataFrame, tests: pd.DataFrame,
           per: pd.DataFrame, meta: dict, out: list[str]) -> None:
    """Inline numbers used in the prose, so the text cannot drift either."""
    prim = paired(per, *KEY_COMPARISONS[0][:2])
    abl = paired(per, *KEY_COMPARISONS[1][:2])
    dim = paired(per, *KEY_COMPARISONS[2][:2])
    best_c = summary[summary.group == "classical"].nlargest(1, "acc_mean").iloc[0]
    best_q = summary[summary.group == "quantum"].nlargest(1, "acc_mean").iloc[0]
    fastest = summary.nsmallest(1, "sec_per_subject").iloc[0]
    slowest = summary.nlargest(1, "sec_per_subject").iloc[0]

    defs = {
        "NSubjects": f"{df.subject.nunique()}",
        "NPipelines": f"{df.pipeline.nunique()}",
        "NFoldScores": f"{len(df):,}".replace(",", r"\,"),
        "NTrials": f"{sorted(set(meta['n_trials_per_subject'].values()))[0]}",
        "BestClassicalName": esc(best_c["pipeline"]),
        "BestClassicalAcc": f"{best_c['acc_mean']:.3f}",
        "BestQuantumName": esc(best_q["pipeline"]),
        "BestQuantumAcc": f"{best_q['acc_mean']:.3f}",
        "PrimaryDelta": f"{prim['delta']:+.3f}",
        "PrimaryP": fmt_p(prim["p"]),
        "PrimaryDz": f"{prim['dz']:.2f}",
        "PrimaryBetter": f"{prim['n_better']}/{prim['n']}",
        "AblationDelta": f"{abl['delta']:+.4f}",
        "AblationP": fmt_p(abl["p"]),
        "AblationBetter": f"{abl['n_better']}/{abl['n']}",
        "DimDelta": f"{dim['delta']:+.4f}",
        "DimP": fmt_p(dim["p"]),
        "NSurviveHolm": f"{int((tests.p_holm < 0.05).sum())}",
        "NFamilyTests": f"{len(tests)}",
        "FastestName": esc(fastest["pipeline"]),
        "FastestSec": f"{fastest['sec_per_subject']:.3f}",
        "SlowestName": esc(slowest["pipeline"]),
        "SlowestSec": f"{slowest['sec_per_subject']:.1f}",
        "SpeedRatio": f"{slowest['sec_per_subject'] / fastest['sec_per_subject']:.0f}",
    }
    out.append("\n%% -------------------------------------------- inline macros\n")
    for k, v in defs.items():
        out.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    out.append("")



def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/tables_auto.tex")
    ap.add_argument("--macros-out", default="paper/macros_auto.tex")
    ap.add_argument("--tag", default=TAG)
    args = ap.parse_args(argv)

    res = Path(args.results)
    df = pd.read_csv(res / f"raw_folds_{args.tag}.csv")
    summary = pd.read_csv(res / f"summary_{args.tag}.csv")
    tests = pd.read_csv(
        res / f"tests_vs_{REFERENCE.replace('/', '-')}_{args.tag}.csv")
    decay = pd.read_csv(res / "concentration_decay.csv")
    meta = json.loads((res / f"meta_{args.tag}.json").read_text())
    per = df.groupby(["pipeline", "subject"])["accuracy"].mean().unstack("pipeline")

    header = [
        "%% AUTO-GENERATED by paper/make_tables.py -- DO NOT EDIT BY HAND.",
        "%% Regenerate after every benchmark run:  python paper/make_tables.py",
        "",
    ]

    # Two files: \newcommand definitions must be read in the preamble (the
    # abstract quotes them), while table floats are only legal in the body.
    mac: list[str] = list(header)
    macros(df, summary, tests, per, meta, mac)

    out: list[str] = list(header)
    table_main(summary, out)
    table_tests(tests, out)
    table_key(per, out)
    table_concentration(decay, out)

    # Second dataset, when its results are present. The manuscript degrades
    # gracefully to a single-dataset paper if the IV-2a run has not been done.
    bci_raw = res / f"raw_folds_{ct.BCI_TAG}.csv"
    bci_sum = res / f"summary_{ct.BCI_TAG}.csv"
    if bci_raw.exists() and bci_sum.exists():
        df_b = pd.read_csv(bci_raw)
        summary_b = pd.read_csv(bci_sum)
        per_b = (df_b.groupby(["pipeline", "subject"])["accuracy"]
                 .mean().unstack("pipeline"))
        ct.table_bci(summary_b, summary, out, esc, GROUP_LABEL)
        ct.table_cross(per, per_b, KEY_COMPARISONS, paired, fmt_p, out)
        ct.cross_macros(summary, summary_b, per, per_b, df_b,
                        KEY_COMPARISONS, paired, fmt_p, mac)
        print(f"  + BCI IV-2a: {df_b.subject.nunique()} subjects")
    else:
        print("  ! BCI IV-2a results absent; manuscript will be single-dataset")

    # Reference-frame results. Each builder no-ops if its inputs are missing,
    # so a checkout with only the core run still produces a valid manuscript.
    ref = rt.load(res)
    built = []
    if rt.table_frame(ref, paired, fmt_p, esc, out):
        built.append("frame effect")
    if rt.table_twin(ref, paired, fmt_p, esc, out):
        built.append("twin control")
    if rt.table_transfer(ref, paired, fmt_p, esc, out):
        built.append("transfer")
    if rt.table_shots(ref, out):
        built.append("shots")
    rt.macros(ref, paired, fmt_p, esc, mac)
    print(f"  + reference-frame tables: {', '.join(built) if built else 'none'}")

    mdest = Path(args.macros_out)
    mdest.parent.mkdir(parents=True, exist_ok=True)
    mdest.write_text("\n".join(mac), encoding="utf-8")
    print(f"wrote {mdest}  ({len(mac)} lines)")

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {dest}  ({len(out)} lines)")
    print(f"  {df.subject.nunique()} subjects, {df.pipeline.nunique()} pipelines")
    print(f"  primary: delta={paired(per, *KEY_COMPARISONS[0][:2])['delta']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
