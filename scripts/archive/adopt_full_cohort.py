"""Switch the manuscript from the first 30 PhysioNet subjects to all 104.

Every PhysioNet analysis was rerun on the full cohort. This moves the
30-subject files into results/n30/ (kept, not deleted: they are the published
numbers of the earlier draft) and puts the 104-subject results under the
canonical names the manuscript's table scripts read. IV-2a and Cho2017 results
are untouched.

    python scripts/adopt_full_cohort.py --dry-run     # show what would move
    python scripts/adopt_full_cohort.py               # do it

Refuses to run unless every expected input exists, so a half-finished queue
cannot leave the manuscript reading a mixture of cohorts.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
STAGE = RESULTS / "full104"
ARCHIVE = RESULTS / "n30"

# Batch tag prefix -> canonical tag for qeeg.merge.
MERGES = {
    "all_core": "motor8_q4",
    "all_ext": "refstate_motor8_q4",
    "all_ext_s1": "refstate_motor8_q4_seed1",
    "all_ext_s2": "refstate_motor8_q4_seed2",
    "all_16ch": "refstate_motor16_q4",
    "all_fb": "filterbank_motor8",
}
# Files produced in the staging folder, moved to results/ under the same name.
STAGED = [
    "transfer_folds_motor8.csv", "transfer_summary_motor8.csv",
    "transfer_frame_tests_motor8.csv",
    "shots_folds_motor8.csv", "shots_summary_motor8.csv",
    "concentration_raw.csv", "concentration_summary.csv",
    "concentration_decay.csv", "concentration_meta.json",
    "reference_gram_motor8.csv", "reference_gram_motor16.csv",
    "reference_gram_motor32.csv", "reference_gram_all64.csv",
]
# Canonical PhysioNet files to archive before the new ones land.
def archived_names() -> list[str]:
    names = []
    for tag in MERGES.values():
        names += [f"raw_folds_{tag}.csv", f"summary_{tag}.csv", f"meta_{tag}.json",
                  "tests_vs_classical-TS+LR_" + tag + ".csv"]
    names += STAGED + ["reference_gram_sweep.csv"]
    return names


def _run(module: str, *args) -> bool:
    """Run a qeeg module in the repository root with src on the path."""
    import os
    r = subprocess.run([sys.executable, "-m", module, *args], cwd=ROOT,
                       env={**os.environ, "PYTHONPATH": "src"},
                       capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-1200:], r.stderr[-1200:])
    return r.returncode == 0


def merge_staged(dry_run: bool) -> bool:
    """Combine the batched stage-2 runs into the files the tables read."""
    ok = True
    if list(STAGE.glob("transfer_folds_tr104_c??.csv")) and \
            not (STAGE / "transfer_folds_motor8.csv").exists():
        print("merge    transfer chunks -> transfer_folds_motor8.csv")
        ok &= dry_run or _run("qeeg.transfer", "--merge",
                              "transfer_folds_tr104_c*.csv",
                              "--out", str(STAGE), "--tag", "motor8")
    for prefix, tag in (("calib_cho", "cho2017"), ("calib_bci", "bci2a")):
        if list(RESULTS.glob(f"calib_folds_{prefix}_b??.csv")) and \
                not (RESULTS / f"calib_folds_{tag}.csv").exists():
            print(f"merge    {prefix} batches -> calib_folds_{tag}.csv")
            ok &= dry_run or _run("qeeg.calibration", "--merge",
                                  f"calib_folds_{prefix}_b*.csv",
                                  "--out", str(RESULTS), "--tag", tag)
    if list(RESULTS.glob("raw_folds_bci4_b??.csv")) and \
            not (RESULTS / "raw_folds_refstate_bci2a4_motor8_q4.csv").exists():
        print("merge    four-class IV-2a batches")
        ok &= dry_run or _run("qeeg.merge", "--results", str(RESULTS),
                              "--pattern", "raw_folds_bci4_b*.csv",
                              "--tag", "refstate_bci2a4_motor8_q4",
                              "--reference", "classical/TS+LR")
    return ok


def check(strict: bool) -> list[str]:
    missing = []
    for prefix in MERGES:
        n = len(list(RESULTS.glob(f"raw_folds_{prefix}_b??.csv")))
        expected = 10
        if n != expected:
            missing.append(f"{prefix}: {n} of {expected} batches")
    for name in STAGED:
        if not (STAGE / name).exists():
            missing.append(f"staging file missing: {name}")
    if missing and strict:
        print("refusing to adopt the full cohort; still missing:")
        for m in missing:
            print("  " + m)
    return missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="adopt even if some inputs are missing (not advised)")
    args = ap.parse_args(argv)

    if not merge_staged(args.dry_run):
        return 1
    missing = check(strict=not args.force)
    if missing and not args.force:
        return 1

    ARCHIVE.mkdir(exist_ok=True)
    # This script is not idempotent and must not pretend to be. A second run
    # would move the already-adopted 104-subject files on top of the 30-subject
    # archive and then regenerate them identically, so n30/ would silently hold
    # a copy of the current cohort under the name of the old one. That happened
    # on 2026-09-17 and the real archive had to be restored from the commit
    # before adoption.
    existing = [p.name for p in ARCHIVE.iterdir() if p.is_file()]
    if existing and not args.force:
        print(f"refusing: {ARCHIVE.relative_to(ROOT)} already holds "
              f"{len(existing)} files, so the archiving step has run before. "
              f"Re-running would overwrite the archive with the current "
              f"cohort. Use --force only if you are certain that is what you "
              f"want, and check the subject counts afterwards.")
        return 1
    for name in archived_names():
        src = RESULTS / name
        if src.exists():
            print(f"archive  {name} -> n30/")
            if not args.dry_run:
                shutil.move(str(src), str(ARCHIVE / name))

    env_note = "PYTHONPATH=src"
    for prefix, tag in MERGES.items():
        # The paired tests need a reference pipeline that exists in the suite.
        # Every filter-bank pipeline is prefixed FB-, so tangent space there is
        # classical/FB-TS+LR and the default would raise.
        reference = ("classical/FB-TS+LR" if "fb" in prefix
                     else "classical/TS+LR")
        print(f"merge    raw_folds_{prefix}_b*.csv -> {tag}  (vs {reference})")
        if not args.dry_run:
            r = subprocess.run(
                [sys.executable, "-m", "qeeg.merge", "--results", str(RESULTS),
                 "--pattern", f"raw_folds_{prefix}_b*.csv", "--tag", tag,
                 "--reference", reference],
                cwd=ROOT, env={**dict(__import__("os").environ), "PYTHONPATH": "src"},
                capture_output=True, text=True)
            if r.returncode:
                print(r.stdout[-1500:], r.stderr[-1500:])
                return 1

    for name in STAGED:
        src = STAGE / name
        if src.exists():
            print(f"adopt    full104/{name} -> results/{name}")
            if not args.dry_run:
                shutil.move(str(src), str(RESULTS / name))

    print("rebuild  reference_gram_sweep.csv")
    if not args.dry_run:
        r = subprocess.run([sys.executable, "-m", "qeeg.gram_sweep"], cwd=ROOT,
                           env={**dict(__import__("os").environ), "PYTHONPATH": "src"},
                           capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-800:], r.stderr[-800:])
            return 1
    print(f"\ndone ({env_note}). Next: qeeg.equivalence, make_tables.py, "
          "figures, then rebuild both PDFs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
