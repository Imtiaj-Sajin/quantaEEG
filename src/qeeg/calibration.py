"""Few-trial calibration: does any quantum kernel degrade more gracefully than
its classical twin when the reference state is poorly estimated?

Why this experiment exists
--------------------------
The paper's reference-frame correction whitens every state by the Frechet mean
of the training covariances. Its quantum kernels and its Riemannian twin both
depend on that estimate. With 45 to 288 training trials the estimate is good,
and the two families tie. The paper names the one regime it had not tested:
very few calibration trials, where the mean is noisy. If a quantum divergence
weighted a noisy reference more robustly than the affine-invariant metric, the
quantum-minus-twin difference would grow as the training set shrinks. That is
the prediction this module tests, stated before running it.

Protocol
--------
For every subject:
  * a fixed, class-balanced test set of --test-size trials is held out once;
  * for each of --draws random draws, the remaining trials are shuffled within
    each class, and training sets of every size in --sizes are taken as the
    first k/2 trials of each class, so the sets are nested across sizes within
    a draw (the trend in k is then not confounded with which trials were drawn);
  * every pipeline is tuned on its training set alone by the same inner
    GridSearchCV the main benchmark uses (same grids, same 4 folds), refit, and
    scored on the fixed test set. The reference state is estimated inside each
    fit, from the k training trials only.
Accuracies are averaged over draws within subject, so each subject is one paired
observation per training size, as everywhere else in the paper.

    PYTHONPATH=src python -u -m qeeg.calibration --dataset cho2017 \
        --subject-list 1,2,3 --tag calib_cho_b01
    PYTHONPATH=src python -m qeeg.calibration --merge "calib_folds_calib_cho_b*.csv" \
        --tag cho2017
"""

from __future__ import annotations

import argparse
import functools
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

print = functools.partial(print, flush=True)  # noqa: A001

from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

PIPELINES = [
    "classical/TS+LR", "classical/MDM", "classical/CSP+LDA",
    "control/riemann-kernel-SVM", "control/logeuclid-kernel-SVM",
    "quantum/HS-overlap-SVM", "quantum/Fidelity-SVM", "quantum/HS-RBF-SVM",
    "quantum/Bures-RBF-SVM", "quantum/QRE-RBF-SVM",
    "quantum/HS-overlap-ref-SVM", "quantum/Fidelity-ref-SVM",
    "quantum/HS-RBF-ref-SVM", "quantum/Bures-RBF-ref-SVM",
    "quantum/QRE-RBF-ref-SVM",
]
INNER_SPLITS = 4


def _draw_indices(y, test_size, sizes, draws, seed):
    """Fixed balanced test set, then nested balanced training sets per draw."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class_test = test_size // len(classes)
    test, pool = [], {}
    for c in classes:
        idx = rng.permutation(np.flatnonzero(y == c))
        test.extend(idx[:per_class_test])
        pool[c] = idx[per_class_test:]
    max_k = max(sizes)
    avail = min(len(v) for v in pool.values()) * len(classes)
    if max_k > avail:
        raise ValueError(f"largest training size {max_k} exceeds the {avail} "
                         f"balanced trials left after the test set")
    plans = []
    for r in range(draws):
        order = {c: rng.permutation(pool[c]) for c in classes}
        for k in sizes:
            tr = np.concatenate([order[c][: k // len(classes)] for c in classes])
            plans.append((r, k, np.sort(tr)))
    return np.sort(np.array(test)), plans


def run_subject(ep, sizes, draws, test_size, seed=0):
    from .benchmark import make_grids
    from .pipelines import make_pipelines

    protos = make_pipelines(suite="extended", seed=seed)
    grids = make_grids()
    missing = [p for p in PIPELINES if p not in protos]
    if missing:
        raise KeyError(f"pipelines not in the extended suite: {missing}")

    test, plans = _draw_indices(ep.y, test_size, sizes, draws, seed + ep.subject)
    rows = []
    for r, k, tr in plans:
        for name in PIPELINES:
            model = clone(protos[name])
            grid = grids.get(name, {})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if grid:
                    search = GridSearchCV(model, grid, cv=INNER_SPLITS,
                                          scoring="accuracy", n_jobs=1,
                                          refit=True, error_score=0.5)
                    search.fit(ep.X[tr], ep.y[tr])
                    fitted, best = search.best_estimator_, search.best_params_
                else:
                    fitted, best = model.fit(ep.X[tr], ep.y[tr]), {}
                acc = accuracy_score(ep.y[test], fitted.predict(ep.X[test]))
            rows.append({
                "subject": ep.subject, "draw": r, "n_train": int(k),
                "n_test": int(len(test)), "pipeline": name,
                "group": name.split("/")[0], "accuracy": float(acc),
                "best": json.dumps({k2: (v if isinstance(v, (int, float, str)) else str(v))
                                    for k2, v in best.items()}),
            })
    return rows


def per_subject(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over draws: one accuracy per subject, pipeline and training size."""
    return (df.groupby(["n_train", "pipeline", "subject"])["accuracy"]
            .mean().reset_index())


def merge(out: Path, pattern: str, tag: str) -> int:
    files = sorted(out.glob(pattern))
    if not files:
        print(f"no files match {pattern!r}")
        return 1
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    dup = df.duplicated(["subject", "draw", "n_train", "pipeline"]).sum()
    if dup:
        print(f"refusing: {dup} duplicated rows")
        return 1
    df.to_csv(out / f"calib_folds_{tag}.csv", index=False)
    ps = per_subject(df)
    summ = (ps.groupby(["n_train", "pipeline"])["accuracy"]
            .agg(["mean", "std", "count"]).reset_index())
    summ.to_csv(out / f"calib_summary_{tag}.csv", index=False)
    print(f"merged {len(files)} files: {df.subject.nunique()} subjects, "
          f"{len(df)} rows -> calib_folds_{tag}.csv")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="cho2017", choices=("cho2017", "bci2a"))
    ap.add_argument("--channels", default="motor8")
    ap.add_argument("--subject-list", default=None)
    ap.add_argument("--sizes", default="10,20,40,80,160")
    ap.add_argument("--draws", type=int, default=10)
    ap.add_argument("--test-size", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--merge", default=None)
    args = ap.parse_args(argv)
    out = Path(args.out)

    if args.merge:
        return merge(out, args.merge, args.tag or args.dataset)

    from .data import CHANNEL_SETS, load_moabb

    sizes = sorted(int(x) for x in args.sizes.split(","))
    subs = ([int(x) for x in args.subject_list.split(",")]
            if args.subject_list else None)
    eps = load_moabb(args.dataset, subjects=subs, channels=CHANNEL_SETS[args.channels])
    print(f"loaded {len(eps)} subjects from {args.dataset}; sizes {sizes}, "
          f"{args.draws} draws, test set {args.test_size}")
    tag = args.tag or f"calib_{args.dataset}"
    partial = out / f"calib_folds_{tag}.partial.csv"
    rows, t0 = [], time.perf_counter()
    for i, ep in enumerate(eps, 1):
        ts = time.perf_counter()
        rows += run_subject(ep, sizes, args.draws, args.test_size, args.seed)
        pd.DataFrame(rows).to_csv(partial, index=False)
        print(f"  [{i}/{len(eps)}] S{ep.subject:03d} n={len(ep.y)} "
              f"({time.perf_counter() - ts:.1f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(out / f"calib_folds_{tag}.csv", index=False)
    partial.unlink(missing_ok=True)
    print(f"Wall clock: {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
