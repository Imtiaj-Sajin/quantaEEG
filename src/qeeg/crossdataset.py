"""Compare the pipeline suite across datasets and combine evidence.

PhysioNet EEGMMIDB (30 subjects, 45 trials each) and BCI Competition IV-2a
(9 subjects, 288 trials each) sit at opposite corners of the design space:
many subjects with little data each, versus few subjects with a lot. Running
the identical suite on both answers the question a single-dataset benchmark
cannot: is the ranking of methods a property of the methods, or of the trial
count?

Two statistical points this module handles explicitly.

*Rank-test floor.* With n subjects the two-sided Wilcoxon signed-rank test
cannot return a p-value below 2^{1-n}. At n=9 that floor is 0.0039, so after
Holm correction across 14 comparisons nothing can fall below 0.0547 no matter
how large the effect. Reporting "0 of 14 significant" without saying so would
badly misrepresent a result where every comparison sits at the floor with
perfect subject-wise consistency. `wilcoxon_floor` computes it.

*Combining evidence.* The entanglement ablation points the same way on both
datasets without reaching significance on either. Fisher's method combines the
two independent p-values into a single test rather than leaving two
underpowered results uninterpreted.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, wilcoxon

DATASETS = {
    "physionet": ("motor8_q4", "PhysioNet EEGMMIDB"),
    "bci2a": ("bci2a_motor8_q4", "BCI Competition IV-2a"),
}

COMPARISONS = [
    ("classical/CSP+LDA", "quantum/CNOT-kernel-SVM", "Best classical vs best quantum"),
    ("classical/CSP+LDA", "quantum/Fidelity-SVM", "Best classical vs density-matrix"),
    ("control/IQP-no-entangle", "quantum/IQP-kernel-SVM", "Entanglement ablation"),
    ("control/PCA-matched-linear", "quantum/IQP-kernel-SVM", "Dimension-matched control"),
]


def wilcoxon_floor(n: int) -> float:
    """Smallest two-sided p the signed-rank test can return with n pairs."""
    return 2.0 ** (1 - n)


def load(results: Path, tag: str) -> pd.DataFrame:
    return pd.read_csv(results / f"raw_folds_{tag}.csv")


def per_subject(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["pipeline", "subject"])["accuracy"].mean().unstack("pipeline")


def paired(per: pd.DataFrame, a: str, b: str) -> dict | None:
    if a not in per.columns or b not in per.columns:
        return None
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
        "at_floor": bool(np.isclose(p, wilcoxon_floor(len(d)))),
    }


def fisher(pvals: list[float]) -> tuple[float, float]:
    """Fisher's method for combining independent p-values."""
    stat = -2.0 * float(np.sum(np.log(pvals)))
    return stat, float(chi2.sf(stat, 2 * len(pvals)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Cross-dataset comparison")
    ap.add_argument("--results", default="results")
    args = ap.parse_args(argv)
    res = Path(args.results)

    frames, pers = {}, {}
    for key, (tag, label) in DATASETS.items():
        f = res / f"raw_folds_{tag}.csv"
        if not f.exists():
            print(f"[skip] {label}: {f.name} not found")
            continue
        frames[key] = load(res, tag)
        pers[key] = per_subject(frames[key])

    if len(pers) < 2:
        print("Need both datasets.")
        return 1

    pd.set_option("display.width", 220)

    # ---- side-by-side ranking -------------------------------------------
    cols = {}
    for key, (tag, label) in DATASETS.items():
        if key not in frames:
            continue
        s = pd.read_csv(res / f"summary_{tag}.csv").set_index("pipeline")
        cols[f"{key}_acc"] = s["acc_mean"]
        cols[f"{key}_rank"] = s["acc_mean"].rank(ascending=False).astype(int)
        cols["group"] = s["group"]
    table = pd.DataFrame(cols).sort_values("physionet_acc", ascending=False)
    table["rank_shift"] = table["bci2a_rank"] - table["physionet_rank"]

    print("=== PIPELINE RANKING ACROSS DATASETS ===")
    print(table[["group", "physionet_acc", "bci2a_acc",
                 "physionet_rank", "bci2a_rank", "rank_shift"]]
          .to_string(float_format=lambda x: f"{x:.4f}"))

    rho = table["physionet_rank"].corr(table["bci2a_rank"], method="spearman")
    print(f"\nSpearman rank correlation between datasets: {rho:.3f}")

    gaps = []
    for key in ("physionet", "bci2a"):
        best_c = table.loc[table.group == "classical", f"{key}_acc"].max()
        best_q = table.loc[table.group == "quantum", f"{key}_acc"].max()
        worst_q = table.loc[table.group == "quantum", f"{key}_acc"].min()
        gaps.append((key, best_c, best_q, best_c - best_q, best_c - worst_q))
    print("\n=== CLASSICAL-QUANTUM GAP ===")
    print(f"{'dataset':12s} {'best cls':>9s} {'best qnt':>9s} {'gap':>8s} {'gap to worst':>13s}")
    for k, bc, bq, g, gw in gaps:
        print(f"{k:12s} {bc:9.4f} {bq:9.4f} {g:8.4f} {gw:13.4f}")

    # ---- key comparisons, per dataset and combined ----------------------
    print("\n=== KEY COMPARISONS ===")
    for a, b, label in COMPARISONS:
        print(f"\n{label}")
        ps = []
        for key, (tag, dslabel) in DATASETS.items():
            if key not in pers:
                continue
            r = paired(pers[key], a, b)
            if r is None:
                continue
            floor = wilcoxon_floor(r["n"])
            note = "  (AT RANK-TEST FLOOR)" if r["at_floor"] else ""
            print(f"  {dslabel:24s} delta={r['delta']:+.4f}  p={r['p']:.4f}"
                  f"  dz={r['dz']:+.3f}  better in {r['n_better']}/{r['n']}"
                  f"  [floor p={floor:.4f}]{note}")
            ps.append(r["p"])
        if len(ps) == 2:
            stat, pc = fisher(ps)
            verdict = "significant" if pc < 0.05 else "not significant"
            print(f"  {'Fisher combined':24s} chi2={stat:.3f}  p={pc:.4f}"
                  f"  -> {verdict}")

    # ---- the floor caveat, stated numerically ---------------------------
    print("\n=== MULTIPLE-COMPARISON FLOOR ===")
    for key, (tag, label) in DATASETS.items():
        if key not in pers:
            continue
        n = pers[key].shape[0]
        floor = wilcoxon_floor(n)
        n_tests = 14
        print(f"{label}: n={n} subjects, min possible Wilcoxon p={floor:.4f}, "
              f"min possible Holm p over {n_tests} tests={min(1.0, floor*n_tests):.4f}")
    print("A Holm-corrected p above 0.05 at n=9 therefore says nothing about "
          "effect size; the test cannot resolve it. Use the pre-specified "
          "comparisons above instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
