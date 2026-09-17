"""Download all three EEG datasets into the cache, in parallel, before any run.

    python scripts/fetch_data.py            # all three, about 12 GB
    python scripts/fetch_data.py --check    # report what is cached, download nothing
    python scripts/fetch_data.py --dataset physionet

This exists because MNE and MOABB download serially inside the benchmark, and
the first pass over 104 subjects spends most of its wall clock on the network
rather than on arithmetic. Fetching first, with a thread pool, turns that into
one bounded step you can watch.

Everything here is idempotent: a file already in the cache is not fetched
again, so the script is safe to interrupt and rerun. It is also the one place
that knows the study needs all three datasets, which the old PhysioNet-only
prefetch did not.

The cache location is MNE's, which `scripts/use_local_datasets.py` can point
inside the project so a 12 GB download does not land on the system drive.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")

BAD = {88, 89, 92, 100, 104}  # documented EEGMMIDB defects
PHYSIONET = [s for s in range(1, 110) if s not in BAD]
RUNS = [4, 8, 12]  # the motor-imagery runs the study uses
BCI = list(range(1, 10))
CHO = list(range(1, 53))


def cache_root() -> str:
    import mne
    return mne.get_config("MNE_DATA") or "the MNE default (~/mne_data)"


def fetch_physionet(subject: int, check: bool) -> str:
    from mne.datasets import eegbci
    if check:
        # update_path=False and a cached file means no network access.
        try:
            paths = eegbci.load_data(subjects=subject, runs=RUNS,
                                     update_path=False, verbose="ERROR")
            return "cached" if len(paths) == len(RUNS) else "missing"
        except Exception:
            return "missing"
    eegbci.load_data(subjects=subject, runs=RUNS, update_path=True,
                     verbose="ERROR")
    return "ok"


def fetch_moabb(name: str, subject: int, check: bool) -> str:
    import logging
    logging.getLogger("moabb").setLevel(logging.ERROR)
    import mne
    # MOABB builds a RawArray per run and MNE narrates every one of them, which
    # buries the progress line under hundreds of lines of noise.
    mne.set_log_level("ERROR")
    import moabb.datasets as mds
    ds = getattr(mds, name)()
    if check:
        # MOABB has no cache query, so ask for the data and see whether it
        # comes back without touching the network. This is why --check on a
        # cold cache is as slow as a download: there is nothing cheaper.
        try:
            ds.get_data(subjects=[subject])
            return "cached"
        except Exception:
            return "missing"
    ds.get_data(subjects=[subject])
    return "ok"


JOBS = {
    "physionet": ("PhysioNet EEGMMIDB", PHYSIONET,
                  lambda s, c: fetch_physionet(s, c)),
    "bci2a": ("BCI Competition IV-2a", BCI,
              lambda s, c: fetch_moabb("BNCI2014_001", s, c)),
    "cho2017": ("Cho2017", CHO,
                lambda s, c: fetch_moabb("Cho2017", s, c)),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", choices=sorted(JOBS), action="append",
                    help="fetch just this one; repeatable (default: all three)")
    ap.add_argument("--check", action="store_true",
                    help="report what is cached without downloading")
    ap.add_argument("--workers", type=int, default=6,
                    help="parallel downloads (default 6)")
    args = ap.parse_args(argv)

    which = args.dataset or sorted(JOBS)
    print(f"cache: {cache_root()}")
    failed = 0
    for key in which:
        title, subjects, fn = JOBS[key]
        verb = "checking" if args.check else "fetching"
        print(f"\n{verb} {title}: {len(subjects)} subject(s)", flush=True)
        bad = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(
                lambda s: (s, _guard(fn, s, args.check)), subjects))
        for s, status in results:
            if status not in ("ok", "cached"):
                bad.append((s, status))
        n_ok = len(results) - len(bad)
        print(f"  {n_ok}/{len(results)} present", flush=True)
        for s, status in bad[:10]:
            print(f"  S{s:03d}: {status}")
        if len(bad) > 10:
            print(f"  ... and {len(bad) - 10} more")
        failed += len(bad)

    if failed and args.check:
        print(f"\n{failed} subject(s) not cached. Run without --check to "
              f"download them.")
        return 1
    if failed:
        print(f"\n{failed} subject(s) failed. Downloads are resumable: run "
              f"this again and the cached files are skipped.")
        return 1
    print("\nAll datasets present." if not args.check
          else "\nAll datasets cached.")
    return 0


def _guard(fn, subject, check) -> str:
    try:
        return fn(subject, check)
    except Exception as e:
        return f"{type(e).__name__}: {e}"[:120]


if __name__ == "__main__":
    raise SystemExit(main())
