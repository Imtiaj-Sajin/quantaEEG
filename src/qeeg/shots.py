"""Finite-shot estimation: what these kernels would actually cost on hardware.

Everything else in this study is infinite-shot and noiseless, which is the
setting most favourable to the quantum side. That is fine for a negative
result -- if quantum kernels lose under ideal simulation they lose on hardware
too -- but it is not fine once the reference-state kernels reach parity
(RESEARCH.md 4.6). A parity claim has to survive the estimation cost, so this
module puts a shot budget on it.

The shot model
--------------
``tr(rho sigma)`` is exactly the observable a SWAP test estimates. The ancilla
is measured in the computational basis with

    P(0) = (1 + tr(rho sigma)) / 2,

so with S shots the estimator k_hat = 2 * (n0/S) - 1 is unbiased with

    Var[k_hat] = (1 - k^2) / S.

We sample n0 ~ Binomial(S, (1+k)/2) per Gram entry, symmetrise (a real device
would estimate each unordered pair once), and project the result back to the
PSD cone, since shot noise readily pushes a Gram matrix out of it. The
Hilbert-Schmidt distance kernel follows from the same primitive:
tr((rho-sigma)^2) = tr(rho^2) + tr(sigma^2) - 2 tr(rho sigma), three SWAP-test
estimates, whose variances add.

Why the frame is the whole story here
-------------------------------------
Shot cost is set by how much kernel *variance* there is to resolve. Two trials
whose overlap differs by d need O(1/d^2) shots to be told apart. The
sensor-frame Gram has off-diagonal std ~0.022 (RESEARCH.md 4.6); the reference
frame raises the variance 4.7-9.5x, i.e. the standard deviation 2.2-3.1x, so
the same discrimination should need roughly 5-9x fewer shots. This module
measures that end to end, on accuracy, rather than inferring it.

On whether the reference frame is free: rho -> W rho W / tr(W rho W) is a
filtering operation, and run as a *post-selected* filter on a device it would
carry an acceptance penalty. It need not be run that way here. The covariances
are computed classically in this setting regardless, so preparing rho~ instead
of rho is the same state-preparation problem with different classical input,
and no rejection cost arises. Both readings are stated so the claim is not
mistaken for a free lunch.

Run
---
    PYTHONPATH=src python -u -m qeeg.shots --subjects 30
"""

from __future__ import annotations

import argparse
import functools
import time
from pathlib import Path

import numpy as np
import pandas as pd

print = functools.partial(print, flush=True)  # noqa: A001

from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC

from .quantum import (
    hs_overlap_kernel,
    median_bandwidth,
    psd_project,
    reference_whitener,
    to_density_matrices,
)

SHOT_GRID = (100, 1_000, 10_000, 100_000, 1_000_000, None)  # None = infinite


def swap_test_sample(K: np.ndarray, shots: int | None,
                     rng: np.random.Generator) -> np.ndarray:
    """Binomial SWAP-test estimate of an overlap Gram matrix.

    Only the upper triangle is sampled and then mirrored: a device estimates
    each unordered pair once, and pretending otherwise would halve the noise
    for free. Diagonal entries tr(rho^2) are sampled too -- purity is not
    known exactly on hardware either.
    """
    if shots is None:
        return K
    p = np.clip((1.0 + K) / 2.0, 0.0, 1.0)
    n = K.shape[0]
    iu = np.triu_indices(n)
    draws = rng.binomial(shots, p[iu]) / shots
    est = np.zeros_like(K)
    est[iu] = 2.0 * draws - 1.0
    est = est + est.T - np.diag(np.diag(est))
    return est


def evaluate(ep, shots_grid=SHOT_GRID, n_splits=5, n_repeats=3, seed=0,
             C_grid=(0.1, 1.0, 10.0), gamma_mults=(0.25, 1.0, 4.0)):
    """Accuracy versus shot budget, in both frames, for the SWAP-test kernels."""
    from pyriemann.estimation import Covariances

    rng = np.random.default_rng(seed)
    C = Covariances(estimator="oas").fit_transform(ep.X)
    y = ep.y
    outer = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                    random_state=seed)
    splits = list(outer.split(C, y))

    rows = []
    for frame in ("sensor", "reference"):
        cov = C
        if frame == "reference":
            # Note: fitted on all of this subject's trials. The reference state
            # is label-free, and the question here is the shot budget, not
            # generalisation, so this keeps the noise study isolated from the
            # train/test machinery. Accuracy comparisons across shot levels are
            # therefore internally valid; absolute values are not comparable to
            # the nested-CV benchmark.
            W = reference_whitener(cov)
            cov = W @ cov @ W
        rho = to_density_matrices(cov)
        K_exact = hs_overlap_kernel(rho)
        purity = np.diag(K_exact).copy()

        for shots in shots_grid:
            K = swap_test_sample(K_exact, shots, rng)
            # Cosine-normalised overlap kernel.
            d = np.sqrt(np.clip(np.diag(K), 1e-9, None))
            K_ov = psd_project(K / np.outer(d, d))
            # HS distance kernel from the same primitive.
            D = np.clip(purity[:, None] + purity[None, :] - 2.0 * K, 0.0, None)

            for kname in ("HS-overlap", "HS-RBF"):
                accs = []
                for tr, te in splits:
                    if kname == "HS-overlap":
                        G = K_ov
                        best = _tune(G, y, tr, C_grid, (1.0,), None)
                    else:
                        best = _tune(D, y, tr, C_grid, gamma_mults, "dist")
                    Cc, gm = best
                    if kname == "HS-RBF":
                        g = gm * median_bandwidth(D[np.ix_(tr, tr)])
                        G = psd_project(np.exp(-g * D))
                    svc = SVC(kernel="precomputed", C=Cc).fit(
                        G[np.ix_(tr, tr)], y[tr])
                    accs.append(accuracy_score(
                        y[te], svc.predict(G[np.ix_(te, tr)])))
                rows.append({
                    "subject": ep.subject, "frame": frame, "kernel": kname,
                    "shots": -1 if shots is None else shots,
                    "accuracy": float(np.mean(accs)),
                    "gram_var": float(np.var(
                        K_exact[~np.eye(len(K_exact), dtype=bool)])),
                })
    return rows


def _tune(M, y, tr, C_grid, gamma_mults, kind, inner=3):
    """Small inner CV on the training split; identical budget in every frame."""
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=inner, shuffle=True, random_state=0)
    best, best_s = (1.0, 1.0), -np.inf
    for gm in gamma_mults:
        if kind == "dist":
            g = gm * median_bandwidth(M[np.ix_(tr, tr)])
            G = psd_project(np.exp(-g * M))
        else:
            G = M
        for Cc in C_grid:
            s = []
            for a, b in skf.split(tr, y[tr]):
                ia, ib = tr[a], tr[b]
                svc = SVC(kernel="precomputed", C=Cc).fit(G[np.ix_(ia, ia)], y[ia])
                s.append(accuracy_score(y[ib], svc.predict(G[np.ix_(ib, ia)])))
            if np.mean(s) > best_s:
                best_s, best = float(np.mean(s)), (Cc, gm)
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--subjects", type=int, default=30)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--channels", type=str, default="motor8")
    ap.add_argument("--dataset", type=str, default="physionet",
                    choices=("physionet", "bci2a", "bci2b"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args(argv)

    from .data import CHANNEL_SETS, load_many, load_moabb

    chans = CHANNEL_SETS[args.channels]
    subs = list(range(args.start, args.start + args.subjects))
    eps = (load_many(subs, channels=chans) if args.dataset == "physionet"
           else load_moabb(args.dataset, channels=chans))
    print(f"loaded {len(eps)} subjects")

    t0 = time.perf_counter()
    rows = []
    for i, ep in enumerate(eps, 1):
        ts = time.perf_counter()
        rows += evaluate(ep, seed=args.seed)
        print(f"  [{i}/{len(eps)}] S{ep.subject:03d} "
              f"({time.perf_counter() - ts:.1f}s)")
    df = pd.DataFrame(rows)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prefix = "" if args.dataset == "physionet" else f"{args.dataset}_"
    tag = args.tag or f"{prefix}{args.channels}"
    df.to_csv(out / f"shots_folds_{tag}.csv", index=False)

    piv = df.pivot_table(index=["kernel", "shots"], columns="frame",
                         values="accuracy")
    piv.to_csv(out / f"shots_summary_{tag}.csv")
    pd.set_option("display.width", 200)
    print(f"\n=== ACCURACY VERSUS SHOT BUDGET (n = {df.subject.nunique()} "
          f"subjects; shots = -1 means infinite) ===")
    print(piv.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n=== SHOTS NEEDED TO REACH 99% OF THE INFINITE-SHOT ACCURACY ===")
    for kern in df.kernel.unique():
        for frame in ("sensor", "reference"):
            sub = df[(df.kernel == kern) & (df.frame == frame)]
            ceiling = sub[sub.shots == -1].accuracy.mean()
            reached = [s for s in sorted(sub.shots.unique()) if s > 0
                       and sub[sub.shots == s].accuracy.mean() >= 0.99 * ceiling]
            need = reached[0] if reached else None
            print(f"  {kern:12s} {frame:10s} ceiling={ceiling:.4f}  "
                  f"shots={'>1e6' if need is None else f'{need:.0e}'}")

    print(f"\nWall clock: {time.perf_counter() - t0:.1f}s -> {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
