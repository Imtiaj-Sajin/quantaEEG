"""Quantum minus twin at each training size, both datasets, paired by subject."""
import glob
import sys

import pandas as pd
from scipy.stats import wilcoxon

TWIN = "control/riemann-kernel-SVM"
REF = ["quantum/Fidelity-ref-SVM", "quantum/HS-overlap-ref-SVM",
       "quantum/HS-RBF-ref-SVM", "quantum/Bures-RBF-ref-SVM",
       "quantum/QRE-RBF-ref-SVM"]


def wide(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    ps = df.groupby(["n_train", "pipeline", "subject"]).accuracy.mean().reset_index()
    return ps.pivot_table(index=["n_train", "subject"], columns="pipeline",
                          values="accuracy")


for name, pat in (("IV-2a", "results/calib_folds_bci2a.csv"),
                  ("Cho2017", "results/calib_folds_cho2017.csv")):
    w = wide(pat)
    if w is None:
        continue
    n = w.index.get_level_values("subject").nunique()
    print(f"\n=== {name}, {n} subjects: mean of the five reference-frame "
          f"kernels minus the twin")
    print(f"  {'k':>4} {'twin':>8} {'quantum':>8} {'diff':>8} {'p':>7} "
          f"{'better':>8}   best kernel")
    for k, g in w.groupby(level=0):
        q = g[[c for c in REF if c in g]].mean(axis=1)
        d = (q - g[TWIN]).dropna()
        best = g[[c for c in REF if c in g]].mean().idxmax()
        db = (g[best] - g[TWIN]).dropna()
        print(f"  {k:4d} {g[TWIN].mean():8.4f} {q.mean():8.4f} "
              f"{d.mean():+8.4f} {wilcoxon(d).pvalue:7.3f} "
              f"{int((d > 0).sum()):4d}/{len(d):<3d}  "
              f"{best.split('/')[1]:20s} {db.mean():+.4f} "
              f"(p={wilcoxon(db).pvalue:.3f})")
