"""Full Cho2017 calibration picture, with the corrections stated up front.

Two analyses are reported, not one, because the pre-specified one (the slope of
quantum minus twin against log2 training size) turned out weaker than the
size-by-size comparison. Choosing between them after seeing both is exactly the
practice this paper exists to criticise, so both appear, corrected, always.
"""
import glob

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

TWIN = "control/riemann-kernel-SVM"
REF = ["quantum/Fidelity-ref-SVM", "quantum/HS-overlap-ref-SVM",
       "quantum/HS-RBF-ref-SVM", "quantum/Bures-RBF-ref-SVM",
       "quantum/QRE-RBF-ref-SVM"]
CLS = ["classical/TS+LR", "classical/CSP+LDA", "classical/MDM"]


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    out, run = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        out[i] = min(run, 1.0)
    return out


df = pd.concat([pd.read_csv(f) for f in
                sorted(glob.glob("results/calib_folds_calib_cho_b0?.csv"))],
               ignore_index=True)
ps = df.groupby(["n_train", "pipeline", "subject"]).accuracy.mean().reset_index()
w = ps.pivot_table(index=["n_train", "subject"], columns="pipeline",
                   values="accuracy")
sizes = sorted(set(w.index.get_level_values("n_train")))
n = w.index.get_level_values("subject").nunique()
print(f"Cho2017, {n} subjects, sizes {sizes}\n")

# 1. Size by size, averaging the five kernels so nothing is selected on outcome.
rows = []
for k in sizes:
    g = w.xs(k, level=0)
    q = g[REF].mean(axis=1)
    dt = (q - g[TWIN]).dropna()
    bc = g[CLS].mean().idxmax()
    dc = (q - g[bc]).dropna()
    rows.append((k, dt.mean(), wilcoxon(dt).pvalue, int((dt > 0).sum()),
                 bc, dc.mean(), wilcoxon(dc).pvalue, int((dc > 0).sum())))
ht = holm([r[2] for r in rows])
hc = holm([r[6] for r in rows])
print("mean of the five reference-frame kernels, per training size")
print(f"  {'k':>4} {'vs twin':>9} {'p':>7} {'Holm':>7} {'win':>6}   "
      f"{'vs best classical':>17} {'p':>7} {'Holm':>7} {'win':>6}")
for r, a, b in zip(rows, ht, hc):
    print(f"  {r[0]:4d} {r[1]:+9.4f} {r[2]:7.3f} {a:7.3f} {r[3]:3d}/{n:<3d}  "
          f"{r[4].split('/')[1]:>7s} {r[5]:+9.4f} {r[6]:7.3f} {b:7.3f} "
          f"{r[7]:3d}/{n:<3d}")

# 2. Per kernel at the sizes where the average separates, Holm over the five
# kernels, so no kernel is quoted without its correction.
for k in (10, 40):
    g = w.xs(k, level=0)
    ds = [(c, (g[c] - g[TWIN]).dropna()) for c in REF]
    pv = holm([wilcoxon(d).pvalue for _, d in ds])
    print(f"\nper kernel at k={k}, minus the twin (Holm over the five kernels)")
    for (c, d), h in zip(ds, pv):
        print(f"  {c.split('/')[1]:22s} {d.mean():+.4f}  "
              f"p={wilcoxon(d).pvalue:.4f}  Holm={h:.3f}  "
              f"{int((d > 0).sum())}/{n}")

# 3. Is the effect inside the equivalence margin the paper already uses?
print(f"\nlargest quantum-minus-twin difference at any size: "
      f"{max(abs(r[1]) for r in rows):.4f}  (the paper's margin is 0.02)")
