"""Cross-subject transfer: the setting where the geometry actually has to earn it.

Within-subject accuracy is a crowded problem on which classical methods are
already excellent. The unsolved problem in BCI is transfer -- a new user should
not have to sit through a calibration session. That is where a better geometry
could plausibly matter, and it is the setting this module evaluates:
leave-one-subject-out, train on the pooled trials of every other subject.

Why the frame matters here more than anywhere else
--------------------------------------------------
Inter-subject variability is dominated by congruence: different head geometry,
electrode placement, skull conductivity and source mixing act on the spatial
covariance as C -> A C A^T. So the naive version of the transfer hypothesis --
"quantum distances (Bures, quantum relative entropy) are more robust to
inter-subject covariance shift than the affine-invariant Riemannian distance"
-- is not merely unsupported, it is algebraically impossible. AIRM is invariant
under the whole congruence group. In the sensor frame the quantum distances are
invariant only under its orthogonal subgroup, so they must be strictly worse;
referred to a reference state they are exactly as invariant, so they can at
best match. Run in the sensor frame, this experiment would measure invariance
groups and report the answer as though it were about geometry.

`recenter=True` therefore whitens every subject by its own Frechet mean, which
uses no labels and is legitimate unsupervised target adaptation (the standard
Riemannian recentring of the transfer literature). It is simultaneously the
reference-state construction of `qeeg.reference`, so one operation puts the
classical and quantum families in the same frame and makes the remaining
comparison a comparison of geometry alone.

What is still open, and what this tests
---------------------------------------
Congruence is not all of the shift. Subjects differ in conditioning, effective
rank, and how peaked their spatial spectrum is, and none of that is removed by
recentring. Bures and the quantum relative entropy weight the eigenvalue
spectrum differently from AIRM -- QRE penalises support mismatch heavily, Bures
compresses large eigenvalue ratios -- so within the reference frame they may
still order transfer differently. That is the real hypothesis, and it is
falsifiable.

Protocol
--------
Kernels are precomputed once over all trials per (frame, kernel), then sliced
per fold, which is what makes tuning affordable at ~1300 pooled trials. The
bandwidth and every hyperparameter are selected on the training subjects only,
by subject-grouped inner CV, so no test subject influences any choice. Every
method, classical and quantum, gets the identical inner budget.

Run
---
    PYTHONPATH=src python -u -m qeeg.transfer --subjects 30
    PYTHONPATH=src python -u -m qeeg.transfer --dataset bci2a
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

print = functools.partial(print, flush=True)  # noqa: A001

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC

from .quantum import (
    bures_distance_sq,
    fidelity_kernel,
    hs_distance_sq,
    hs_overlap_kernel,
    qre_divergence,
    reference_whitener,
    to_density_matrices,
)

C_GRID = (0.1, 1.0, 10.0)
GAMMA_MULT_GRID = (0.25, 1.0, 4.0)

# name -> (callable, is_distance)
QUANTUM_KERNELS = {
    "HS-overlap": (hs_overlap_kernel, False),
    "Fidelity": (lambda A, B=None: fidelity_kernel(A, B, squared=True), False),
    "HS-RBF": (hs_distance_sq, True),
    "Bures-RBF": (bures_distance_sq, True),
    "QRE-RBF": (qre_divergence, True),
}


# --------------------------------------------------------------------------
# Data assembly
# --------------------------------------------------------------------------

def pooled_covariances(epochs_list, recenter: bool, estimator: str = "oas"):
    """Stack every subject's covariances, optionally each in its own frame.

    Recentring is per subject and label-free: it uses only that subject's own
    trials, which for the held-out subject is exactly the unlabelled data a
    real deployment would have.
    """
    from pyriemann.estimation import Covariances

    covs, ys, subs = [], [], []
    for ep in epochs_list:
        C = Covariances(estimator=estimator).fit_transform(ep.X)
        if recenter:
            W = reference_whitener(C)
            C = W @ C @ W
        covs.append(C)
        ys.append(ep.y)
        subs.append(np.full(len(ep.y), ep.subject))
    return np.concatenate(covs), np.concatenate(ys), np.concatenate(subs)


# --------------------------------------------------------------------------
# Quantum kernels, precomputed once
# --------------------------------------------------------------------------

def precompute(covs: np.ndarray) -> dict[str, tuple[np.ndarray, bool]]:
    """Full pairwise matrix per quantum kernel over all pooled trials."""
    rho = to_density_matrices(covs)
    out = {}
    for name, (fn, is_dist) in QUANTUM_KERNELS.items():
        t0 = time.perf_counter()
        M = fn(rho)
        if not is_dist:
            # Cosine-normalise the overlaps, as the within-subject pipelines do.
            d = np.sqrt(np.clip(np.diag(M), 1e-12, None))
            M = M / np.outer(d, d)
        out[name] = (M, is_dist)
        print(f"    {name:12s} {time.perf_counter() - t0:6.1f}s")
    return out


def _gram(M: np.ndarray, is_dist: bool, rows, cols, gamma: float | None):
    sub = M[np.ix_(rows, cols)]
    return np.exp(-gamma * sub) if is_dist else sub


def _median_bandwidth(D: np.ndarray) -> float:
    n = D.shape[0]
    off = D[~np.eye(n, dtype=bool)] if D.shape[0] == D.shape[1] else D.ravel()
    med = float(np.median(off))
    return 1.0 / med if med > 1e-12 else 1.0


def _fit_eval_quantum(M, is_dist, tr, te, y, groups, inner_splits=3):
    """Tune (C, gamma_mult) on training subjects only, then score the held-out one."""
    base_gamma = _median_bandwidth(M[np.ix_(tr, tr)]) if is_dist else None
    gm_grid = GAMMA_MULT_GRID if is_dist else (1.0,)

    n_groups = len(np.unique(groups[tr]))
    inner = GroupKFold(n_splits=min(inner_splits, n_groups))
    best, best_score = None, -np.inf
    for gm in gm_grid:
        gamma = None if base_gamma is None else gm * base_gamma
        for C in C_GRID:
            scores = []
            for itr, ite in inner.split(tr, y[tr], groups[tr]):
                a, b = tr[itr], tr[ite]
                K = _gram(M, is_dist, a, a, gamma)
                svc = SVC(kernel="precomputed", C=C).fit(K, y[a])
                scores.append(accuracy_score(
                    y[b], svc.predict(_gram(M, is_dist, b, a, gamma))))
            s = float(np.mean(scores))
            if s > best_score:
                best_score, best = s, (C, gm, gamma)

    C, gm, gamma = best
    svc = SVC(kernel="precomputed", C=C).fit(_gram(M, is_dist, tr, tr, gamma), y[tr])
    Kte = _gram(M, is_dist, te, tr, gamma)
    pred = svc.predict(Kte)
    try:
        auc = roc_auc_score(y[te], svc.decision_function(Kte))
    except Exception:  # noqa: BLE001
        auc = float("nan")
    return accuracy_score(y[te], pred), auc, {"C": C, "gamma_mult": gm}


# --------------------------------------------------------------------------
# Classical comparators
# --------------------------------------------------------------------------

def classical_models():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from pyriemann.classification import MDM
    from pyriemann.tangentspace import TangentSpace

    from .pipelines import SPDKernelSVC

    return {
        "classical/TS+LR": (
            Pipeline([("ts", TangentSpace(metric="riemann")),
                      ("sc", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=2000))]),
            {"clf__C": list(C_GRID)},
        ),
        "classical/MDM": (MDM(metric="riemann"), {}),
        "control/riemann-kernel-SVM": (
            SPDKernelSVC(metric="riemann"), {"C": list(C_GRID)}),
        "control/logeuclid-kernel-SVM": (
            SPDKernelSVC(metric="logeuclid"), {"C": list(C_GRID)}),
    }


def _fit_eval_classical(proto, grid, X, y, groups, tr, te, inner_splits=3):
    from sklearn.base import clone
    from sklearn.model_selection import GridSearchCV

    if grid:
        n_groups = len(np.unique(groups[tr]))
        cv = GroupKFold(n_splits=min(inner_splits, n_groups))
        search = GridSearchCV(clone(proto), grid, cv=cv, scoring="accuracy",
                              n_jobs=1, error_score=0.5)
        search.fit(X[tr], y[tr], groups=groups[tr])
        model, best = search.best_estimator_, search.best_params_
    else:
        model, best = clone(proto).fit(X[tr], y[tr]), {}
    pred = model.predict(X[te])
    try:
        auc = roc_auc_score(y[te], model.decision_function(X[te]))
    except Exception:  # noqa: BLE001
        try:
            auc = roc_auc_score(y[te], model.predict_proba(X[te])[:, 1])
        except Exception:  # noqa: BLE001
            auc = float("nan")
    return accuracy_score(y[te], pred), auc, best


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def run_loso(epochs_list, recenter: bool) -> list[dict]:
    frame = "reference" if recenter else "sensor"
    print(f"\n--- {frame} frame ---")
    X, y, groups = pooled_covariances(epochs_list, recenter=recenter)
    print(f"  pooled {len(y)} trials from {len(np.unique(groups))} subjects")
    print("  precomputing quantum kernels:")
    kernels = precompute(X)
    classical = classical_models()

    rows = []
    for held in np.unique(groups):
        te = np.flatnonzero(groups == held)
        tr = np.flatnonzero(groups != held)
        t0 = time.perf_counter()
        for name, (proto, grid) in classical.items():
            acc, auc, best = _fit_eval_classical(
                proto, grid, X, y, groups, tr, te)
            rows.append({"subject": int(held), "frame": frame,
                         "pipeline": name, "group": name.split("/")[0],
                         "accuracy": acc, "auc": auc, "best": json.dumps(best)})
        for name, (M, is_dist) in kernels.items():
            acc, auc, best = _fit_eval_quantum(M, is_dist, tr, te, y, groups)
            rows.append({"subject": int(held), "frame": frame,
                         "pipeline": f"quantum/{name}", "group": "quantum",
                         "accuracy": acc, "auc": auc, "best": json.dumps(best)})
        top = max((r for r in rows if r["subject"] == held),
                  key=lambda r: r["accuracy"])
        print(f"  S{held:03d} ({time.perf_counter() - t0:5.1f}s) "
              f"best: {top['pipeline']} {top['accuracy']:.4f}")
    return rows


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["frame", "pipeline", "group"]).agg(
        acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
        auc_mean=("auc", "mean"), n_subjects=("subject", "nunique"))
    return g.reset_index().sort_values(["frame", "acc_mean"], ascending=[True, False])


def paired_frame_test(df: pd.DataFrame) -> pd.DataFrame:
    """Per pipeline: reference frame minus sensor frame, paired by subject."""
    from scipy.stats import wilcoxon

    piv = df.pivot_table(index=["pipeline", "subject"], columns="frame",
                         values="accuracy").dropna()
    out = []
    for pipe, sub in piv.groupby(level=0):
        d = sub["reference"] - sub["sensor"]
        if len(d) < 3 or np.allclose(d, 0):
            p = float("nan")
        else:
            p = float(wilcoxon(sub["reference"], sub["sensor"]).pvalue)
        out.append({"pipeline": pipe, "n": len(d),
                    "sensor": float(sub["sensor"].mean()),
                    "reference": float(sub["reference"].mean()),
                    "delta": float(d.mean()),
                    "dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) else np.nan,
                    "p_wilcoxon": p,
                    "better": int((d > 0).sum())})
    return pd.DataFrame(out).sort_values("delta", ascending=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--subjects", type=int, default=30)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--channels", type=str, default="motor8")
    ap.add_argument("--dataset", type=str, default="physionet",
                    choices=("physionet", "bci2a", "bci2b"))
    ap.add_argument("--frames", type=str, default="both",
                    choices=("both", "sensor", "reference"))
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args(argv)

    from .data import CHANNEL_SETS, load_many, load_moabb

    chans = CHANNEL_SETS[args.channels]
    subs = list(range(args.start, args.start + args.subjects))
    print(f"Loading {args.dataset} ({args.channels}, {len(chans)} ch) ...")
    eps = (load_many(subs, channels=chans) if args.dataset == "physionet"
           else load_moabb(args.dataset, channels=chans))
    print(f"  usable: {len(eps)} subjects, {sum(len(e) for e in eps)} trials")
    if len(eps) < 3:
        print("Cross-subject transfer needs at least 3 subjects.")
        return 1

    frames = (["sensor", "reference"] if args.frames == "both"
              else [args.frames])
    t0 = time.perf_counter()
    rows = []
    for f in frames:
        rows += run_loso(eps, recenter=(f == "reference"))
    df = pd.DataFrame(rows)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prefix = "" if args.dataset == "physionet" else f"{args.dataset}_"
    tag = args.tag or f"{prefix}{args.channels}"
    df.to_csv(out / f"transfer_folds_{tag}.csv", index=False)
    summ = summarise(df)
    summ.to_csv(out / f"transfer_summary_{tag}.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== LEAVE-ONE-SUBJECT-OUT, mean over held-out subjects ===")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if len(frames) == 2:
        tests = paired_frame_test(df)
        tests.to_csv(out / f"transfer_frame_tests_{tag}.csv", index=False)
        print("\n=== REFERENCE FRAME MINUS SENSOR FRAME (paired by subject) ===")
        print(tests.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nWall clock: {time.perf_counter() - t0:.1f}s -> {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
