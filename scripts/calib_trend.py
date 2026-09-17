"""The pre-specified test of the few-trial prediction, as one test per kernel.

The prediction in the paper's discussion is about a *trend*: the quantum minus
twin difference should grow as the training set shrinks. Testing the five
kernels at five sizes separately is 25 tests and invites exactly the
selection effect the twin control exists to prevent. So fit, per subject and
per kernel, the slope of (quantum minus twin) against log2(training size), and
test the nine slopes against zero. The prediction is a negative slope. One test
per kernel, Holm corrected across the five.
"""
import glob

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

TWIN = "control/riemann-kernel-SVM"
REF = ["quantum/Fidelity-ref-SVM", "quantum/HS-overlap-ref-SVM",
       "quantum/HS-RBF-ref-SVM", "quantum/Bures-RBF-ref-SVM",
       "quantum/QRE-RBF-ref-SVM"]


def holm(pvals):
    order = np.argsort(pvals)
    out = np.empty(len(pvals))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(pvals) - rank) * pvals[i])
        out[i] = min(run, 1.0)
    return out


def load(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    ps = df.groupby(["n_train", "pipeline", "subject"]).accuracy.mean().reset_index()
    return ps.pivot_table(index=["n_train", "subject"], columns="pipeline",
                          values="accuracy")


def report(name, wide):
    print(f"\n=== {name}: {wide.index.get_level_values('subject').nunique()} "
          f"subjects, sizes {sorted(set(wide.index.get_level_values('n_train')))}")
    ps, slopes_all = [], {}
    for c in REF:
        if c not in wide:
            continue
        d = (wide[c] - wide[TWIN]).unstack(level=0)  # subject x n_train
        x = np.log2(np.array(d.columns, float))
        slopes = np.array([np.polyfit(x, row, 1)[0] for row in d.to_numpy()])
        p = wilcoxon(slopes).pvalue
        ps.append(p)
        slopes_all[c] = (slopes, p, d)
    corrected = holm(np.array(ps))
    print(f"  {'kernel':24s} {'slope/doubling':>15s} {'neg':>5s} {'p':>8s} {'Holm':>8s}")
    for (c, (slopes, p, d)), pc in zip(slopes_all.items(), corrected):
        print(f"  {c.split('/')[1]:24s} {slopes.mean():+15.5f} "
              f"{int((slopes < 0).sum()):3d}/{len(slopes)} {p:8.3f} {pc:8.3f}")
    print(f"  Wilcoxon floor at n={len(slopes)}: {2.0 ** (1 - len(slopes)):.4f}")
    # The same trend for the classical twin against the best classical pipeline,
    # as a negative control: if every method's advantage shrinks with k, the
    # trend is not about quantum structure.
    for base in ("classical/TS+LR", "classical/CSP+LDA"):
        if base not in wide:
            continue
        d = (wide[TWIN] - wide[base]).unstack(level=0)
        x = np.log2(np.array(d.columns, float))
        s = np.array([np.polyfit(x, r, 1)[0] for r in d.to_numpy()])
        print(f"  control: twin minus {base.split('/')[1]:9s} "
              f"slope {s.mean():+.5f}, p={wilcoxon(s).pvalue:.3f}")


bci = load("results/calib_folds_bci2a.csv")
if bci is not None:
    report("IV-2a", bci)
cho = load("results/calib_folds_cho2017.csv")
if cho is not None:
    report("Cho2017 (partial, batches still running)", cho)
else:
    print("\nCho2017: no finished batches yet")
