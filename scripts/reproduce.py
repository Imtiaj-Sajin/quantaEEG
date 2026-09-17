"""Run the entire study, from downloading the data to the built manuscript.

    python scripts/reproduce.py --plan     # print the stages, run nothing
    python scripts/reproduce.py --force    # recompute the study from raw data
    python scripts/reproduce.py            # run only what is missing
    python scripts/reproduce.py --from transfer     # resume from one stage
    python scripts/reproduce.py --only calibration  # run a single stage

Every stage declares the files it produces. A stage whose outputs all exist is
skipped, so the pipeline is safe to stop and restart: interrupt it, run it
again, and it continues where it left off. Within a stage the jobs run in
parallel, capped so the machine is not oversubscribed.

Read --force before using it. This repository commits results/, so a fresh
clone already satisfies every stage and the default run does almost nothing,
which is correct: the record is there and `run.py verify` reads it. --force
ignores what exists and recomputes from the raw recordings, overwriting
results/. That is the mode for actually reproducing the study.

This is the same work, in the same order, that produced the committed results.
The queue scripts it replaces are kept in scripts/archive/ as the historical
record.

Cost. On a 6-core desktop the whole thing is a few days of wall clock, almost
all of it in stages 1 and 5. The first run also downloads about 12 GB. Nothing
here needs a GPU: the quantum kernels are exact state-vector simulations of 3
to 6 qubits, which is linear algebra on 8x8 to 64x64 matrices.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

BAD = {88, 89, 92, 100, 104}
PHYSIONET = [s for s in range(1, 110) if s not in BAD]
CHO = list(range(1, 53))
BCI = list(range(1, 10))

# PhysioNet within-subject suites: (batch tag prefix, extra args, merged tag).
SUITES = [
    ("all_core", [], "motor8_q4"),
    ("all_ext", ["--suite", "extended"], "refstate_motor8_q4"),
    ("all_16ch", ["--suite", "extended", "--channels", "motor16"],
     "refstate_motor16_q4"),
    ("all_ext_s1", ["--suite", "extended", "--seed", "1"],
     "refstate_motor8_q4_seed1"),
    ("all_ext_s2", ["--suite", "extended", "--seed", "2"],
     "refstate_motor8_q4_seed2"),
    ("all_fb", ["--suite", "filterbank"], "filterbank_motor8"),
]


def chunks(xs, n):
    size = -(-len(xs) // n)
    return [xs[i:i + size] for i in range(0, len(xs), size)]


def py(*args: str) -> list[str]:
    return [sys.executable, "-u", "-m", *args]


class Job:
    def __init__(self, name, cmd, produces, cwd=ROOT):
        self.name, self.cmd, self.produces, self.cwd = name, cmd, produces, cwd

    def done(self) -> bool:
        # No declared output means the job is cheap and idempotent (prefetch
        # checks its own cache, the paper rebuild is a minute), so it always
        # runs. Without the first clause all([]) is True and these jobs would
        # silently never run at all.
        return bool(self.produces) and all(
            (ROOT / p).exists() for p in self.produces)


class Stage:
    def __init__(self, key, title, jobs, workers=1, note="", satisfied_by=()):
        self.key, self.title, self.jobs = key, title, jobs
        self.workers, self.note = workers, note
        # Files whose existence means this stage's work is already in the
        # record, even though its own per-batch outputs are gone. The batch
        # files are build intermediates: once merged they are deleted, and
        # without this a fresh checkout would be told to redo two days of
        # compute whose results are sitting right there.
        self.satisfied_by = tuple(satisfied_by)

    force = False

    def satisfied(self) -> bool:
        if Stage.force:
            return False
        return bool(self.satisfied_by) and all(
            (ROOT / p).exists() for p in self.satisfied_by)

    def pending(self):
        if self.satisfied():
            return []
        if Stage.force:
            return list(self.jobs)
        return [j for j in self.jobs if not j.done()]


def build_stages() -> list[Stage]:
    st: list[Stage] = []

    st.append(Stage("data", "Download the three datasets", [
        Job("fetch", [sys.executable, str(ROOT / "scripts" / "fetch_data.py")],
            [])], note="about 12 GB on the first run; cached files are skipped"))

    # 1. PhysioNet within-subject, six suites over ten batches each.
    jobs = []
    for prefix, extra, _ in SUITES:
        for i, part in enumerate(chunks(PHYSIONET, 10), 1):
            tag = f"{prefix}_b{i:02d}"
            jobs.append(Job(tag,
                            py("qeeg.benchmark", "--subject-list",
                               ",".join(map(str, part)), "--tag", tag,
                               "--no-stats", "--resume", *extra),
                            [f"results/raw_folds_{tag}.csv"]))
    st.append(Stage("physionet", "PhysioNet within-subject, 6 suites x 104 subjects",
                    jobs, workers=8, note="the long pole, about two days",
                    satisfied_by=[f"results/raw_folds_{t}.csv" for _, _, t in SUITES]))

    st.append(Stage("merge-physionet", "Merge the PhysioNet batches", [
        Job(tag, py("qeeg.merge", "--results", "results",
                    "--pattern", f"raw_folds_{prefix}_b*.csv", "--tag", tag,
                    "--reference",
                    "classical/FB-TS+LR" if "fb" in prefix else "classical/TS+LR"),
            [f"results/raw_folds_{tag}.csv"])
        for prefix, _, tag in SUITES], satisfied_by=[f"results/raw_folds_{t}.csv" for _, _, t in SUITES]))

    # 2. The other two datasets, within subject.
    st.append(Stage("other-datasets", "IV-2a and Cho2017, within subject", [
        Job("bci2a-core",
            py("qeeg.benchmark", "--dataset", "bci2a", "--tag", "bci2a_motor8_q4"),
            ["results/raw_folds_bci2a_motor8_q4.csv"]),
        Job("bci2a-ext",
            py("qeeg.benchmark", "--dataset", "bci2a", "--suite", "extended",
               "--tag", "refstate_bci2a_motor8_q4"),
            ["results/raw_folds_refstate_bci2a_motor8_q4.csv"]),
    ] + [
        Job(f"cho-b{i}",
            py("qeeg.benchmark", "--dataset", "cho2017", "--suite", "extended",
               "--subject-list", ",".join(map(str, part)),
               "--tag", f"cho_batch{chr(64 + i)}", "--no-stats", "--resume"),
            [f"results/raw_folds_cho_batch{chr(64 + i)}.csv"])
        for i, part in enumerate(chunks(CHO, 6), 1)
    ], workers=6, satisfied_by=[
        "results/raw_folds_bci2a_motor8_q4.csv",
        "results/raw_folds_refstate_bci2a_motor8_q4.csv",
        "results/raw_folds_refstate_cho2017_motor8_q4.csv"]))

    st.append(Stage("merge-cho", "Merge the Cho2017 batches", [
        Job("cho", py("qeeg.merge", "--results", "results",
                      "--pattern", "raw_folds_cho_batch?.csv",
                      "--tag", "refstate_cho2017_motor8_q4",
                      "--reference", "classical/TS+LR"),
            ["results/raw_folds_refstate_cho2017_motor8_q4.csv"])],
        satisfied_by=["results/raw_folds_refstate_cho2017_motor8_q4.csv"]))

    # 3. Four classes, one subject per process.
    st.append(Stage("fourclass", "Four-class IV-2a, extended suite", [
        Job(f"bci4_b{s:02d}",
            py("qeeg.benchmark", "--dataset", "bci2a", "--classes", "4",
               "--suite", "extended", "--subject-list", str(s),
               "--tag", f"bci4_b{s:02d}", "--no-stats", "--resume"),
            [f"results/raw_folds_bci4_b{s:02d}.csv"])
        for s in BCI], workers=6,
        satisfied_by=["results/raw_folds_refstate_bci2a4_motor8_q4.csv"]))
    st.append(Stage("merge-fourclass", "Merge the four-class batches", [
        Job("bci4", py("qeeg.merge", "--results", "results",
                       "--pattern", "raw_folds_bci4_b*.csv",
                       "--tag", "refstate_bci2a4_motor8_q4",
                       "--reference", "classical/TS+LR"),
            ["results/raw_folds_refstate_bci2a4_motor8_q4.csv"])],
        satisfied_by=["results/raw_folds_refstate_bci2a4_motor8_q4.csv"]))

    # 4. Transfer, split by held-out subject; memory-bound, so a low cap.
    st.append(Stage("transfer", "Cross-subject transfer, leave one out", [
        Job(f"tr_c{i:02d}",
            py("qeeg.transfer", "--subjects", "109", "--start", "1",
               "--heldout", ",".join(map(str, part)),
               "--out", "results/full104", "--tag", f"tr104_c{i:02d}"),
            [f"results/full104/transfer_folds_tr104_c{i:02d}.csv"])
        for i, part in enumerate(chunks(PHYSIONET, 6), 1)],
        workers=6,
        note="about 1.3 GB per process steady, far more while precomputing "
             "Gram matrices: do not raise the cap above 8 on 24 GB",
        satisfied_by=["results/transfer_folds_motor8.csv"]))
    st.append(Stage("merge-transfer", "Merge the transfer chunks", [
        Job("transfer", py("qeeg.transfer", "--merge",
                           "transfer_folds_tr104_*.csv",
                           "--out", "results/full104", "--tag", "motor8"),
            ["results/full104/transfer_folds_motor8.csv"])],
        satisfied_by=["results/transfer_folds_motor8.csv"]))

    st.append(Stage("crosssession", "Cross-session transfer on IV-2a", [
        Job("crosssession", py("qeeg.crosssession", "--dataset", "bci2a"),
            ["results/crosssession_folds_bci2a_motor8.csv"])]))

    # 5. Few-trial calibration.
    st.append(Stage("calibration", "Few-trial calibration sweep", [
        Job(f"calib_cho_b{i:02d}",
            py("qeeg.calibration", "--dataset", "cho2017", "--subject-list",
               ",".join(map(str, part)), "--tag", f"calib_cho_b{i:02d}"),
            [f"results/calib_folds_calib_cho_b{i:02d}.csv"])
        for i, part in enumerate(chunks(CHO, 8), 1)
    ] + [
        Job(f"calib_bci_b{s:02d}",
            py("qeeg.calibration", "--dataset", "bci2a", "--subject-list",
               str(s), "--tag", f"calib_bci_b{s:02d}"),
            [f"results/calib_folds_calib_bci_b{s:02d}.csv"])
        for s in BCI], workers=8, satisfied_by=[
            "results/calib_folds_cho2017.csv",
            "results/calib_folds_bci2a.csv"]))
    st.append(Stage("merge-calibration", "Merge the calibration batches", [
        Job("cho", py("qeeg.calibration", "--merge",
                      "calib_folds_calib_cho_b*.csv", "--tag", "cho2017"),
            ["results/calib_folds_cho2017.csv"]),
        Job("bci", py("qeeg.calibration", "--merge",
                      "calib_folds_calib_bci_b*.csv", "--tag", "bci2a"),
            ["results/calib_folds_bci2a.csv"])], satisfied_by=[
        "results/calib_folds_cho2017.csv", "results/calib_folds_bci2a.csv"]))

    # 6. Diagnostics.
    st.append(Stage("diagnostics", "Gram, concentration and finite shots", [
        Job(f"gram_{ch}",
            py("qeeg.reference", "--gram", "--subjects", "109", "--start", "1",
               "--channels", ch, "--out", f"results/reference_gram_{ch}.csv"),
            [f"results/reference_gram_{ch}.csv"])
        for ch in ("motor8", "motor16", "motor32", "all64")
    ] + [
        Job("concentration",
            py("qeeg.concentration", "--subjects", "109", "--start", "1",
               "--out", "results"),
            ["results/concentration_decay.csv"]),
        Job("shots",
            py("qeeg.shots", "--subjects", "109", "--start", "1",
               "--out", "results"),
            ["results/shots_folds_motor8.csv"]),
    ], workers=6))

    st.append(Stage("equivalence", "Two one-sided tests against the twin", [
        Job("equivalence", py("qeeg.equivalence"),
            ["results/equivalence_twin.csv"])]))

    st.append(Stage("paper", "Tables, macros, figures and the PDF", [
        Job("tables", [sys.executable, str(ROOT / "paper" / "make_tables.py")], []),
        Job("figures", [sys.executable, str(ROOT / "run.py"), "figures"], []),
        Job("build", [sys.executable, str(ROOT / "run.py"), "paper"], []),
        Job("verify", [sys.executable, str(ROOT / "run.py"), "verify"], []),
    ]))
    return st


def run_job(j: Job, env: dict) -> tuple[str, int, float]:
    t0 = time.perf_counter()
    log = RES / f"run_{j.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as fh:
        r = subprocess.run(j.cmd, cwd=j.cwd, env=env, stdout=fh,
                           stderr=subprocess.STDOUT)
    return j.name, r.returncode, time.perf_counter() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", action="store_true", help="print the stages only")
    ap.add_argument("--from", dest="start", help="resume from this stage key")
    ap.add_argument("--only", help="run just this stage key")
    ap.add_argument("--workers", type=int, default=None,
                    help="override the per-stage parallelism cap")
    ap.add_argument("--force", action="store_true",
                    help="recompute even where results already exist")
    args = ap.parse_args(argv)

    Stage.force = args.force
    stages = build_stages()
    keys = [s.key for s in stages]
    if args.only:
        # Without this check a mistyped key selects no stages at all and the
        # run ends with "All stages complete", which reads as success.
        if args.only not in keys:
            print(f"unknown stage {args.only!r}; keys are: {', '.join(keys)}")
            return 1
        stages = [s for s in stages if s.key == args.only]
    elif args.start:
        if args.start not in keys:
            print(f"unknown stage {args.start!r}; keys are: {', '.join(keys)}")
            return 1
        stages = stages[keys.index(args.start):]

    if args.plan:
        print(f"{'stage':20s} {'jobs':>5s} {'todo':>5s}  what it does")
        for s in stages:
            todo = len(s.pending())
            mark = "  (already in results/)" if s.satisfied() and not todo else ""
            print(f"{s.key:20s} {len(s.jobs):5d} {todo:5d}  {s.title}{mark}")
            if s.note:
                print(f"{'':20s} {'':11s}  ({s.note})")
        total = sum(len(s.pending()) for s in stages)
        n_sat = sum(1 for s in stages if s.satisfied())
        print(f"\n{total} job(s) to run.")
        if n_sat:
            print(f"{n_sat} stage(s) need nothing: their results are already in "
                  f"this checkout, because results/ is committed. That is what "
                  f"makes `python run.py verify` work on a fresh clone.\n"
                  f"To recompute them from the raw recordings instead, add "
                  f"--force. It overwrites files in results/, so commit or "
                  f"stash first, and budget a few days of CPU.")
        return 0

    if args.force:
        print("--force: recomputing regardless of what exists. This "
              "overwrites files in results/.\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    # One BLAS thread per process: several processes each spawning a full pool
    # ran slower than one until this was set.
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[v] = "1"

    for s in stages:
        todo = s.pending()
        if not todo:
            print(f"[skip] {s.key}: {s.title} (already complete)")
            continue
        w = args.workers or s.workers
        print(f"\n[run ] {s.key}: {s.title}  "
              f"({len(todo)} job(s), {w} at a time)")
        if s.note:
            print(f"       note: {s.note}")
        failed = []
        with ThreadPoolExecutor(max_workers=w) as ex:
            for name, rc, secs in ex.map(lambda j: run_job(j, env), todo):
                mark = "ok " if rc == 0 else "FAIL"
                print(f"       {mark} {name:22s} {secs / 60:7.1f} min")
                if rc:
                    failed.append(name)
        if failed:
            print(f"\n{len(failed)} job(s) failed in stage {s.key}: {failed}")
            print(f"Logs are in results/run_<job>.log. Fix, then rerun with "
                  f"--from {s.key}; finished jobs will skip themselves.")
            return 1
    print("\nAll stages complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
