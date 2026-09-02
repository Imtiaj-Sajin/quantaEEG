"""Within-subject benchmark runner with nested cross-validation.

Protocol
--------
For each subject independently:
  outer loop  RepeatedStratifiedKFold(n_splits, n_repeats)  -> generalisation
  inner loop  GridSearchCV(4-fold)                          -> hyperparameters

Every pipeline gets its own hyperparameter grid and the *same* inner-CV
budget, so no method is handicapped by an unlucky fixed default. This matters:
the most common way a quantum-advantage claim evaporates is that the quantum
model was tuned and the classical baseline was not.

Aggregation across subjects uses the Wilcoxon signed-rank test (paired,
non-parametric, subject-wise), which is the standard for MI-BCI comparisons
and does not assume normally distributed per-subject accuracies.
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
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

from .data import load_many

# Redirected stdout is block-buffered; flush so progress is visible live.
print = functools.partial(print, flush=True)  # noqa: A001
from .pipelines import N_QUBITS, make_pipelines


# --------------------------------------------------------------------------
# Hyperparameter grids
# --------------------------------------------------------------------------

def make_grids() -> dict[str, dict]:
    """Per-pipeline grids. Kept small and comparable in size across methods."""
    C_GRID = [0.1, 1.0, 10.0]
    return {
        "classical/logvar+LDA": {},
        "classical/CSP+LDA": {"csp__nfilter": [2, 4, 6]},
        "classical/MDM": {},
        "classical/TS+LR": {"clf__C": C_GRID},
        "classical/TS+RBF-SVM": {"clf__C": C_GRID, "clf__gamma": ["scale", 0.01, 0.1]},
        "quantum/HS-overlap-SVM": {"clf__C": C_GRID},
        "quantum/Fidelity-SVM": {"clf__C": C_GRID},
        "quantum/HS-RBF-SVM": {"clf__C": C_GRID, "clf__gamma_mult": [0.25, 1.0, 4.0]},
        "quantum/Bures-RBF-SVM": {"clf__C": C_GRID, "clf__gamma_mult": [0.25, 1.0, 4.0]},
        "quantum/IQP-kernel-SVM": {"clf__C": C_GRID, "clf__scale": [0.5, 1.0, 2.0]},
        "quantum/CNOT-kernel-SVM": {"clf__C": C_GRID, "clf__scale": [0.5, 1.0, 2.0]},
        "control/IQP-no-entangle": {"clf__C": C_GRID, "clf__scale": [0.5, 1.0, 2.0]},
        "control/PCA-matched-RBF": {"clf__C": C_GRID, "clf__gamma": ["scale", 0.01, 0.1]},
        "control/PCA-matched-linear": {"clf__C": C_GRID},
        "control/logeuclid-TS+LR": {"clf__C": C_GRID},
    }


# --------------------------------------------------------------------------
# Core evaluation
# --------------------------------------------------------------------------

def _score_fold(pipe, Xtr, ytr, Xte, yte) -> tuple[float, float]:
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    acc = accuracy_score(yte, pred)
    try:
        if hasattr(pipe, "decision_function"):
            s = pipe.decision_function(Xte)
        else:
            s = pipe.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, s)
    except Exception:  # noqa: BLE001 - some estimators expose neither
        auc = float("nan")
    return float(acc), float(auc)


def evaluate_subject(
    ep,
    pipelines: dict,
    grids: dict,
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 0,
    inner_splits: int = 4,
) -> list[dict]:
    """Nested CV for one subject. Returns one row per (pipeline, outer fold)."""
    rows = []
    outer = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    splits = list(outer.split(ep.X, ep.y))

    for name, proto in pipelines.items():
        grid = grids.get(name, {})
        t0 = time.perf_counter()
        for fold, (tr, te) in enumerate(splits):
            from sklearn.base import clone

            pipe = clone(proto)
            if grid:
                search = GridSearchCV(
                    pipe, grid, cv=inner_splits, scoring="accuracy",
                    n_jobs=1, refit=True, error_score=0.5,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    search.fit(ep.X[tr], ep.y[tr])
                model = search.best_estimator_
                best = {k: v for k, v in search.best_params_.items()}
                acc = accuracy_score(ep.y[te], model.predict(ep.X[te]))
                try:
                    s = model.decision_function(ep.X[te])
                    auc = roc_auc_score(ep.y[te], s)
                except Exception:  # noqa: BLE001
                    try:
                        auc = roc_auc_score(
                            ep.y[te], model.predict_proba(ep.X[te])[:, 1]
                        )
                    except Exception:  # noqa: BLE001
                        auc = float("nan")
            else:
                best = {}
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    acc, auc = _score_fold(
                        pipe, ep.X[tr], ep.y[tr], ep.X[te], ep.y[te]
                    )
            rows.append({
                "subject": ep.subject,
                "pipeline": name,
                "group": name.split("/")[0],
                "fold": fold,
                "accuracy": acc,
                "auc": auc,
                "best_params": json.dumps(best),
            })
        elapsed = time.perf_counter() - t0
        for r in rows[-len(splits):]:
            r["fit_seconds_total"] = elapsed
    return rows


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per-pipeline summary: mean over folds within subject, then over subjects."""
    per_subj = (
        df.groupby(["pipeline", "group", "subject"])[["accuracy", "auc"]]
        .mean()
        .reset_index()
    )
    agg = (
        per_subj.groupby(["pipeline", "group"])
        .agg(
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            n_subjects=("subject", "nunique"),
        )
        .reset_index()
        .sort_values("acc_mean", ascending=False)
    )
    t = (
        df.groupby("pipeline")["fit_seconds_total"]
        .mean()
        .rename("sec_per_subject")
        .reset_index()
    )
    return agg.merge(t, on="pipeline", how="left")


def paired_tests(df: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Wilcoxon signed-rank of every pipeline against `reference`, subject-wise."""
    from scipy.stats import wilcoxon

    per_subj = (
        df.groupby(["pipeline", "subject"])["accuracy"].mean().unstack("pipeline")
    )
    if reference not in per_subj.columns:
        raise ValueError(f"reference {reference!r} not in results")
    ref = per_subj[reference]

    rows = []
    for name in per_subj.columns:
        if name == reference:
            continue
        a, b = per_subj[name], ref
        mask = a.notna() & b.notna()
        d = (a[mask] - b[mask]).to_numpy()
        if np.allclose(d, 0):
            stat, p = float("nan"), 1.0
        else:
            stat, p = wilcoxon(d)
        # Paired Cohen's d (effect size on the differences).
        eff = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan")
        rows.append({
            "pipeline": name,
            "vs": reference,
            "delta_acc": float(d.mean()),
            "wilcoxon_stat": stat,
            "p_value": float(p),
            "cohens_d": eff,
            "n_subjects": int(mask.sum()),
            "n_better": int((d > 0).sum()),
        })
    out = pd.DataFrame(rows).sort_values("delta_acc", ascending=False)
    # Holm-Bonferroni correction over the family of comparisons.
    order = np.argsort(out["p_value"].to_numpy())
    m = len(out)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * out["p_value"].to_numpy()[idx]
        running = max(running, min(val, 1.0))
        adj[idx] = running
    out["p_holm"] = adj
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Quantum vs classical EEG benchmark")
    ap.add_argument("--subjects", type=int, default=20,
                    help="number of PhysioNet subjects (from S001 upward)")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--qubits", type=int, default=N_QUBITS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--channels", type=str, default="motor8",
                    choices=["motor8", "motor16", "motor32", "all64"])
    ap.add_argument("--dataset", type=str, default="physionet",
                    choices=["physionet", "bci2a", "bci2b"],
                    help="physionet = EEGMMIDB (many subjects, 45 trials "
                         "each); bci2a/bci2b = BCI Competition IV via MOABB "
                         "(9 subjects, many trials each)")
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--subject-list", type=str, default=None,
                    help="explicit comma-separated subject ids (overrides "
                         "--start/--subjects); lets a long run be split into "
                         "batches that are merged afterwards")
    ap.add_argument("--tag", type=str, default=None,
                    help="override the output filename tag (batch runs)")
    ap.add_argument("--no-stats", action="store_true",
                    help="skip summary/tests (use when merging batches later)")
    ap.add_argument("--reference", type=str, default="classical/TS+LR",
                    help="baseline for the paired significance tests")
    args = ap.parse_args(argv)

    from .data import CHANNEL_SETS, MOABB_DATASETS, load_moabb

    chans = CHANNEL_SETS[args.channels]
    if args.subject_list:
        subjects = [int(x) for x in args.subject_list.split(",") if x.strip()]
    elif args.dataset != "physionet":
        subjects = None  # MOABB datasets define their own subject list
    else:
        subjects = list(range(args.start, args.start + args.subjects))

    n_req = len(subjects) if subjects else "all"
    print(f"Loading {n_req} subjects from {args.dataset} "
          f"({args.channels}, {len(chans)} ch) ...")
    if args.dataset == "physionet":
        eps = load_many(subjects, channels=chans)
    else:
        eps = load_moabb(args.dataset, subjects=subjects, channels=chans)
    print(f"  usable: {len(eps)} subjects, "
          f"{sum(len(e) for e in eps)} trials total")
    if not eps:
        print("No usable subjects loaded.")
        return 1

    pipelines = make_pipelines(n_qubits=args.qubits, seed=args.seed)
    grids = make_grids()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prefix = "" if args.dataset == "physionet" else f"{args.dataset}_"
    tag = args.tag or f"{prefix}{args.channels}_q{args.qubits}"
    partial = out / f"raw_folds_{tag}.partial.csv"

    all_rows = []
    t0 = time.perf_counter()
    for i, ep in enumerate(eps, 1):
        ts = time.perf_counter()
        rows = evaluate_subject(
            ep, pipelines, grids,
            n_splits=args.splits, n_repeats=args.repeats, seed=args.seed,
        )
        all_rows.extend(rows)
        best = max(
            {r["pipeline"]: r for r in rows}.keys(),
            key=lambda k: np.mean([r["accuracy"] for r in rows if r["pipeline"] == k]),
        )
        print(f"  [{i}/{len(eps)}] S{ep.subject:03d} n={len(ep):3d} "
              f"({time.perf_counter()-ts:5.1f}s)  best: {best}")
        # Checkpoint after every subject: a long run must survive being killed.
        pd.DataFrame(all_rows).to_csv(partial, index=False)

    df = pd.DataFrame(all_rows)
    df.to_csv(out / f"raw_folds_{tag}.csv", index=False)
    partial.unlink(missing_ok=True)

    summary = tests = None
    if not args.no_stats:
        summary = summarise(df)
        summary.to_csv(out / f"summary_{tag}.csv", index=False)
        if df["subject"].nunique() > 1:
            tests = paired_tests(df, args.reference)
            tests.to_csv(
                out / f"tests_vs_{args.reference.replace('/', '-')}_{tag}.csv",
                index=False)

    meta = {
        "dataset": args.dataset,
        "subjects_requested": subjects,
        "subjects_used": [e.subject for e in eps],
        "n_trials_per_subject": {str(e.subject): int(len(e)) for e in eps},
        "channels": chans,
        "n_qubits": args.qubits,
        "outer_cv": f"{args.splits}-fold x {args.repeats} repeats",
        "inner_cv": "4-fold GridSearchCV",
        "seed": args.seed,
        "reference": args.reference,
        "total_seconds": round(time.perf_counter() - t0, 1),
    }
    (out / f"meta_{tag}.json").write_text(json.dumps(meta, indent=2))

    pd.set_option("display.width", 200)
    if summary is not None:
        print("\n=== SUMMARY (mean over subjects) ===")
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if tests is not None:
        print(f"\n=== PAIRED TESTS vs {args.reference} ===")
        print(tests.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nWall clock: {meta['total_seconds']}s -> {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
