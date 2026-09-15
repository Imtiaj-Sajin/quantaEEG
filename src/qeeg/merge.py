"""Merge batched benchmark runs into one result set.

The full 30-subject nested-CV run is ~50 minutes of compute, longer than a
single foreground process is allowed here, so it is executed as a sequence of
small batches (``--subject-list`` + ``--tag``). This module concatenates their
per-fold CSVs and recomputes the summary and paired statistics over the pooled
subjects, which is identical to what a single long run would have produced:
subjects are evaluated independently, so batching changes nothing statistically.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

from .benchmark import paired_tests, summarise


def merge(results: Path, pattern: str, tag: str, reference: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(results / pattern)))
    if not files:
        raise SystemExit(f"no files matched {results / pattern}")
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    # A subject evaluated twice (e.g. a re-run batch) would double-weight it.
    df = df.drop_duplicates(subset=["subject", "pipeline", "fold"], keep="last")

    print(f"merged {len(files)} files -> {df['subject'].nunique()} subjects, "
          f"{df['pipeline'].nunique()} pipelines, {len(df)} rows")
    for f in files:
        print(f"    {Path(f).name}")

    df.to_csv(results / f"raw_folds_{tag}.csv", index=False)
    summary = summarise(df)
    summary.to_csv(results / f"summary_{tag}.csv", index=False)

    tests = None
    if df["subject"].nunique() > 1:
        tests = paired_tests(df, reference)
        tests.to_csv(
            results / f"tests_vs_{reference.replace('/', '-')}_{tag}.csv",
            index=False)

    # Carry the per-batch metadata through, so a merged run is described as
    # completely as a single one: trial counts per subject (the manuscript's
    # trial-count macro reads them), suite, seed, channels and CV protocol.
    # Settings must agree across batches; a mismatch means batches from
    # different configurations were mixed, which is refused.
    meta = {
        "merged_from": [Path(f).name for f in files],
        "subjects_used": sorted(int(s) for s in df["subject"].unique()),
        "n_subjects": int(df["subject"].nunique()),
        "reference": reference,
    }
    batch_metas = []
    for f in files:
        mp = Path(f).with_name(Path(f).name.replace("raw_folds_", "meta_", 1)
                               .replace(".csv", ".json"))
        if mp.exists():
            batch_metas.append(json.loads(mp.read_text()))
    if batch_metas:
        trials = {}
        for m in batch_metas:
            trials.update(m.get("n_trials_per_subject", {}))
        meta["n_trials_per_subject"] = dict(sorted(trials.items(), key=lambda kv: int(kv[0])))
        for key in ("dataset", "suite", "seed", "channels", "n_qubits",
                    "outer_cv", "inner_cv"):
            values = {json.dumps(m.get(key), sort_keys=True) for m in batch_metas}
            if len(values) > 1:
                raise SystemExit(f"refusing to merge: batches disagree on {key}: {values}")
            meta[key] = batch_metas[0].get(key)
        meta["total_seconds"] = round(sum(m.get("total_seconds") or 0 for m in batch_metas), 1)
    (results / f"meta_{tag}.json").write_text(json.dumps(meta, indent=2))

    pd.set_option("display.width", 200)
    print("\n=== SUMMARY (mean over subjects) ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if tests is not None:
        print(f"\n=== PAIRED TESTS vs {reference} ===")
        print(tests.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Merge batched benchmark results")
    ap.add_argument("--results", type=str, default="results")
    ap.add_argument("--pattern", type=str, default="raw_folds_batch*.csv")
    ap.add_argument("--tag", type=str, default="motor8_q4")
    ap.add_argument("--reference", type=str, default="classical/TS+LR")
    args = ap.parse_args(argv)
    merge(Path(args.results), args.pattern, args.tag, args.reference)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
