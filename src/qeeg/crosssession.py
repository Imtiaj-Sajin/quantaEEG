"""Cross-session transfer within subject: the shift the invariance argument does
not cover.

Cross-subject shift is dominated by congruence, and RESEARCH.md 4.6-4.7 show
that once the frame is corrected every geometry ties there, as the algebra
says it must. Cross-session shift is the interesting complement. Within one
subject, across two recordings of the same paradigm, the congruence component
is far milder: same head, same cap, similar impedances. What remains is the
residual the invariance argument says nothing about -- drift in conditioning,
effective rank and spectral shape -- and it is exactly the regime in which a
quantum divergence, which weights the eigenvalue spectrum differently from the
affine-invariant metric, could still separate from its classical twin.

The paper names this as the most promising remaining place for a real effect
(4.4), so it has to be tried rather than pointed at.

Protocol
--------
BCI Competition IV-2a has two sessions per subject (the original "training"
and "evaluation" sets, 144 trials each). For every subject and both
directions, train on one session and test on the other. Hyperparameters are
tuned by stratified inner CV *inside the training session only*; the test
session influences nothing.

Two frames:
  sensor     -- covariances as recorded;
  reference  -- each session whitened by its OWN Frechet mean, computed
                without labels. For the test session this is the unlabelled
                data a real deployment has before it decodes anything.

Every method, classical and quantum, gets the identical inner budget. Kernels
are precomputed once per subject over both sessions' trials and sliced per
direction, which makes tuning cheap at 288 trials.

Run
---
    PYTHONPATH=src python -u -m qeeg.crosssession
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

from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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
from .transfer import C_GRID, GAMMA_MULT_GRID, QUANTUM_KERNELS, classical_models

INNER_SPLITS = 4


# --------------------------------------------------------------------------

def _median_bandwidth(D):
    n = D.shape[0]
    off = D[~np.eye(n, dtype=bool)] if D.shape[0] == D.shape[1] else D.ravel()
    med = float(np.median(off))
    return 1.0 / med if med > 1e-12 else 1.0


def _gram(M, is_dist, rows, cols, gamma):
    sub = M[np.ix_(rows, cols)]
    return np.exp(-gamma * sub) if is_dist else sub


def _precompute(covs):
    rho = to_density_matrices(covs)
    out = {}
    for name, (fn, is_dist) in QUANTUM_KERNELS.items():
        M = fn(rho)
        if not is_dist:
            d = np.sqrt(np.clip(np.diag(M), 1e-12, None))
            M = M / np.outer(d, d)
        out[name] = (M, is_dist)
    return out


def _tune_quantum(M, is_dist, tr, y, seed):
    """C and bandwidth multiplier from stratified inner CV on the training
    session alone."""
    base_gamma = _median_bandwidth(M[np.ix_(tr, tr)]) if is_dist else None
    gm_grid = GAMMA_MULT_GRID if is_dist else (1.0,)
    inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
    best, best_s = None, -np.inf
    for gm in gm_grid:
        gamma = None if base_gamma is None else gm * base_gamma
        for C in C_GRID:
            scores = []
            for a, b in inner.split(tr, y[tr]):
                ia, ib = tr[a], tr[b]
                svc = SVC(kernel="precomputed", C=C).fit(
                    _gram(M, is_dist, ia, ia, gamma), y[ia])
                scores.append(accuracy_score(
                    y[ib], svc.predict(_gram(M, is_dist, ib, ia, gamma))))
            s = float(np.mean(scores))
            if s > best_s:
                best_s, best = s, (C, gm, gamma)
    return best


def _eval_quantum(M, is_dist, tr, te, y, seed):
    C, gm, gamma = _tune_quantum(M, is_dist, tr, y, seed)
    svc = SVC(kernel="precomputed", C=C).fit(_gram(M, is_dist, tr, tr, gamma), y[tr])
    Kte = _gram(M, is_dist, te, tr, gamma)
    pred = svc.predict(Kte)
    try:
        auc = roc_auc_score(y[te], svc.decision_function(Kte))
    except Exception:  # noqa: BLE001
        auc = float("nan")
    return accuracy_score(y[te], pred), auc, {"C": C, "gamma_mult": gm}


def _eval_classical(proto, grid, X, y, tr, te, seed):
    inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=seed)
    if grid:
        search = GridSearchCV(clone(proto), grid, cv=inner, scoring="accuracy",
                              n_jobs=1, error_score=0.5).fit(X[tr], y[tr])
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

def run_subject(ep, recenter: bool, seed: int = 0) -> list[dict]:
    from pyriemann.estimation import Covariances

    frame = "reference" if recenter else "sensor"
    sessions = np.unique(ep.session)
    if len(sessions) != 2:
        raise ValueError(f"subject {ep.subject}: expected 2 sessions, got {sessions}")

    C = Covariances(estimator="oas").fit_transform(ep.X)
    if recenter:
        # Per-session whitening. Label-free, so legitimate for the test
        # session: it is the unlabelled data a deployment has before decoding.
        C = C.copy()
        for s in sessions:
            idx = np.flatnonzero(ep.session == s)
            W = reference_whitener(C[idx])
            C[idx] = W @ C[idx] @ W
    y = ep.y

    kernels = _precompute(C)
    classical = classical_models()

    rows = []
    for train_s, test_s in ((sessions[0], sessions[1]), (sessions[1], sessions[0])):
        tr = np.flatnonzero(ep.session == train_s)
        te = np.flatnonzero(ep.session == test_s)
        direction = f"{train_s}->{test_s}"
        for name, (proto, grid) in classical.items():
            acc, auc, best = _eval_classical(proto, grid, C, y, tr, te, seed)
            rows.append({"subject": ep.subject, "direction": direction,
                         "frame": frame, "pipeline": name,
                         "group": name.split("/")[0], "accuracy": acc,
                         "auc": auc, "best": json.dumps(best),
                         "n_train": len(tr), "n_test": len(te)})
        for name, (M, is_dist) in kernels.items():
            acc, auc, best = _eval_quantum(M, is_dist, tr, te, y, seed)
            rows.append({"subject": ep.subject, "direction": direction,
                         "frame": frame, "pipeline": f"quantum/{name}",
                         "group": "quantum", "accuracy": acc, "auc": auc,
                         "best": json.dumps(best),
                         "n_train": len(tr), "n_test": len(te)})
    return rows


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    # Average the two directions per subject first, so each subject counts
    # once in the across-subject statistics.
    per = (df.groupby(["frame", "pipeline", "group", "subject"])["accuracy"]
           .mean().reset_index())
    g = per.groupby(["frame", "pipeline", "group"]).agg(
        acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"),
        n_subjects=("subject", "nunique")).reset_index()
    return g.sort_values(["frame", "acc_mean"], ascending=[True, False])


def paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import wilcoxon

    per = (df.groupby(["frame", "pipeline", "subject"])["accuracy"]
           .mean().unstack("pipeline"))
    out = []
    # Frame effect per pipeline.
    ref, sen = per.loc["reference"], per.loc["sensor"]
    for p in ref.columns:
        d = (ref[p] - sen[p]).dropna()
        out.append({"comparison": "reference - sensor", "pipeline": p,
                    "n": len(d), "delta": float(d.mean()),
                    "p": float(wilcoxon(d).pvalue) if len(d) >= 3 and not np.allclose(d, 0) else np.nan,
                    "better": int((d > 0).sum())})
    # Quantum versus the stronger classical twin, reference frame.
    twins = [t for t in ("control/riemann-kernel-SVM", "control/logeuclid-kernel-SVM")
             if t in ref.columns]
    if twins:
        twin = max(twins, key=lambda t: ref[t].mean())
        for p in ref.columns:
            if not p.startswith("quantum/"):
                continue
            d = (ref[p] - ref[twin]).dropna()
            out.append({"comparison": f"quantum - twin ({twin.split('/')[-1]}), reference",
                        "pipeline": p, "n": len(d), "delta": float(d.mean()),
                        "p": float(wilcoxon(d).pvalue) if len(d) >= 3 and not np.allclose(d, 0) else np.nan,
                        "better": int((d > 0).sum())})
    return pd.DataFrame(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="bci2a", choices=("bci2a", "bci2b"))
    ap.add_argument("--channels", default="motor8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)

    from .data import CHANNEL_SETS, load_moabb

    chans = CHANNEL_SETS[args.channels]
    print(f"Loading {args.dataset} ({args.channels}, {len(chans)} ch) ...")
    eps = load_moabb(args.dataset, channels=chans)
    eps = [e for e in eps if e.session is not None and len(np.unique(e.session)) == 2]
    print(f"  usable: {len(eps)} subjects with two sessions")
    if not eps:
        return 1

    t0 = time.perf_counter()
    rows = []
    for i, ep in enumerate(eps, 1):
        ts = time.perf_counter()
        for recenter in (False, True):
            rows += run_subject(ep, recenter, args.seed)
        print(f"  [{i}/{len(eps)}] S{ep.subject:03d} ({time.perf_counter() - ts:.0f}s)")
    df = pd.DataFrame(rows)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.dataset}_{args.channels}"
    df.to_csv(out / f"crosssession_folds_{tag}.csv", index=False)
    summ = summarise(df)
    summ.to_csv(out / f"crosssession_summary_{tag}.csv", index=False)
    tests = paired_tests(df)
    tests.to_csv(out / f"crosssession_tests_{tag}.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== CROSS-SESSION, mean over subjects (both directions averaged) ===")
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== PAIRED TESTS ===")
    print(tests.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nWall clock: {time.perf_counter() - t0:.0f}s -> {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
