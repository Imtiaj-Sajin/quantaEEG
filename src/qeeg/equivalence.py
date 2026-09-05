"""Equivalence testing for the classical-twin control.

The paper's central claim is that a quantum kernel and a classical SPD kernel
differing only in the metric perform the *same*. Reporting a non-significant
difference does not establish that: absence of evidence is not evidence of
absence, and with enough noise any two methods look alike. The claim needs a
test whose null hypothesis is a *difference*, so that rejecting it supports
equivalence.

That is what two one-sided tests (TOST) do. For paired per-subject differences
d and an equivalence margin m, TOST tests

    H01: mu <= -m   against   mu > -m
    H02: mu >=  m   against   mu <  m

and declares equivalence when both are rejected, i.e. when
p_TOST = max(p1, p2) < alpha. Equivalently, the 90 % confidence interval for
the mean difference lies entirely inside (-m, +m) when alpha = 0.05.

Choosing the margin
-------------------
The margin is a scientific judgement and must be fixed in advance, not chosen
to make the result come out. We use **m = 0.02 accuracy**, on two grounds:

1. It is smaller than the smallest effect this paper treats as real. The
   reference-frame correction is worth +0.050 to +0.187 depending on dataset
   and register size, so a residual difference below 0.02 is at most 40 % of
   the *weakest* effect we claim, and under 11 % of the strongest.
2. It is below what is operationally meaningful in a motor-imagery BCI, where
   a two-point accuracy change does not alter whether a system is usable.

`equivalence_bound` additionally reports the smallest margin at which
equivalence *would* hold, so a reader who prefers a different margin can read
the answer off directly instead of taking ours on trust.

Caveat carried into the paper: TOST as implemented here is the paired
t-version and assumes the per-subject differences are approximately normal.
At n = 30 that is mild; at n = 9 (IV-2a) it is a real limitation, and the
achieved bound there should be read as indicative.

Run
---
    PYTHONPATH=src python -m qeeg.equivalence
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

import numpy as np
import pandas as pd

print = functools.partial(print, flush=True)  # noqa: A001

from scipy import stats

# Pre-specified equivalence margin, in accuracy units. See module docstring.
MARGIN = 0.02

FRAME_PAIRS = [
    ("quantum/Fidelity-ref-SVM", "Fidelity"),
    ("quantum/HS-overlap-ref-SVM", "HS overlap"),
    ("quantum/HS-RBF-ref-SVM", "HS-RBF"),
    ("quantum/Bures-RBF-ref-SVM", "Bures-RBF"),
    ("quantum/QRE-RBF-ref-SVM", "QRE-RBF"),
]
TWINS = ["control/riemann-kernel-SVM", "control/logeuclid-kernel-SVM"]
FB_TWINS = ["control/FB-riemann-kernel-SVM", "control/FB-logeuclid-kernel-SVM"]


def tost_paired(d: np.ndarray, margin: float = MARGIN,
                alpha: float = 0.05) -> dict:
    """Two one-sided tests on paired differences.

    Returns the TOST p-value, the (1-2*alpha) confidence interval that
    corresponds to it, and whether equivalence is established at `margin`.
    """
    d = np.asarray(d, dtype=float)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    df = n - 1
    if se == 0:
        p1 = p2 = 0.0 if abs(mean) < margin else 1.0
        half = 0.0
    else:
        # H01: mu <= -m, rejected when the mean sits well above -m.
        p1 = float(stats.t.sf((mean + margin) / se, df))
        # H02: mu >= +m, rejected when the mean sits well below +m.
        p2 = float(stats.t.cdf((mean - margin) / se, df))
        half = float(stats.t.ppf(1 - alpha, df) * se)
    p_tost = max(p1, p2)
    return {
        "n": n, "mean": mean, "sd": sd,
        "ci_low": mean - half, "ci_high": mean + half,
        "p_tost": p_tost, "equivalent": bool(p_tost < alpha),
        "dz": mean / sd if sd else np.nan,
    }


def equivalence_bound(d: np.ndarray, alpha: float = 0.05) -> float:
    """Smallest margin at which equivalence would be declared.

    This is just the larger absolute end of the (1-2*alpha) interval: the
    interval sits inside +-m exactly when m exceeds it.
    """
    d = np.asarray(d, dtype=float)
    n = len(d)
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    half = float(stats.t.ppf(1 - alpha, n - 1) * se)
    return max(abs(d.mean() - half), abs(d.mean() + half))


# --------------------------------------------------------------------------

def _per(df):
    return df.groupby(["pipeline", "subject"])["accuracy"].mean().unstack("pipeline")


def _best_twin(per, twins):
    avail = [t for t in twins if t in per.columns]
    return max(avail, key=lambda t: per[t].mean()) if avail else None


def collect(results: Path, margin: float = MARGIN) -> pd.DataFrame:
    """Run TOST for every quantum kernel against its twin, in every setting."""
    def read(name):
        p = results / name
        return pd.read_csv(p) if p.exists() else None

    settings = []
    phys = read("raw_folds_refstate_motor8_q4.csv")
    if phys is not None:
        settings.append(("PhysioNet, 3 qubits", _per(phys),
                         [(k, lab) for k, lab in FRAME_PAIRS], TWINS))
    fb = read("raw_folds_filterbank_motor8.csv")
    if fb is not None:
        settings.append(("PhysioNet, 5 qubits", _per(fb),
                         [(k.replace("quantum/", "quantum/FB-"), lab)
                          for k, lab in FRAME_PAIRS], FB_TWINS))
    tr = read("transfer_folds_motor8.csv")
    if tr is not None:
        per_t = (tr[tr.frame == "reference"]
                 .groupby(["pipeline", "subject"])["accuracy"]
                 .mean().unstack("pipeline"))
        settings.append(("Transfer (LOSO)", per_t,
                         [(k.replace("quantum/", "quantum/")
                            .replace("-ref-SVM", "").replace("-SVM", ""), lab)
                          for k, lab in FRAME_PAIRS], TWINS))
    bci = read("raw_folds_refstate_bci2a_motor8_q4.csv")
    if bci is not None:
        settings.append(("BCI IV-2a, 3 qubits", _per(bci),
                         [(k, lab) for k, lab in FRAME_PAIRS], TWINS))

    rows = []
    for name, per, kernels, twins in settings:
        twin = _best_twin(per, twins)
        if twin is None:
            continue
        for key, lab in kernels:
            if key not in per.columns:
                continue
            d = (per[key] - per[twin]).dropna().to_numpy()
            r = tost_paired(d, margin)
            r.update({"setting": name, "kernel": lab, "twin": twin,
                      "bound": equivalence_bound(d)})
            rows.append(r)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default="results")
    ap.add_argument("--margin", type=float, default=MARGIN)
    ap.add_argument("--out", default="results/equivalence_twin.csv")
    args = ap.parse_args(argv)

    df = collect(Path(args.results), args.margin)
    if df.empty:
        print("no inputs found")
        return 1
    df.to_csv(args.out, index=False)

    pd.set_option("display.width", 200)
    print(f"=== TOST vs the classical twin, margin +-{args.margin} accuracy ===")
    show = df[["setting", "kernel", "n", "mean", "ci_low", "ci_high",
               "p_tost", "equivalent", "bound"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nequivalent at +-{args.margin}: "
          f"{int(df.equivalent.sum())}/{len(df)} comparisons")
    for s, g in df.groupby("setting", sort=False):
        print(f"  {s:22s} {int(g.equivalent.sum())}/{len(g)}   "
              f"largest bound {g['bound'].max():.4f}")
    print(f"\nWorst-case bound over all comparisons: {df['bound'].max():.4f}")
    print("i.e. equivalence holds at any margin above that value.")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
