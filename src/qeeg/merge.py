"""Merge batched benchmark runs into one result set.

The full 30-subject nested-CV run is ~50 minutes of compute, longer than a
single foreground process is allowed here, so it is executed as a sequence of
small batches (``--subject-list`` + ``--tag``). This module concatenates their
per-fold CSVs and recomputes the summary and paired statistics over the pooled
subjects, which is identical to what a single long run would have produced --
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

    (results / f"meta_{tag}.json").write_text(json.dumps({
        "merged_from": [Path(f).name for f in files],
        "subjects_used": sorted(int(s) for s in df["subject"].unique()),
        "n_subjects": int(df["subject"].nunique()),
        "reference": reference,
    }, indent=2))

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
