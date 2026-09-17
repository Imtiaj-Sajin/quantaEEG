"""Re-split the remaining transfer work across more processes.

Why
---
Transfer runs as six chunk jobs, each doing every held-out subject in the
sensor frame and then again in the reference frame. Six processes on twelve
threads is fine while the calibration and four-class jobs share the machine,
but once those drain, half the CPU sits idle for hours while transfer finishes
alone.

The checkpoint design is one file per tag, so a tag's work cannot be split
across processes without two of them writing the same file. The way to
parallelise further is therefore to stop the chunk jobs, promote each
checkpoint to a finished part file (it holds real, complete rows: whole
subjects in whole frames), and start new jobs under new tags covering only the
subject-frame pairs nobody has done.

Nothing is recomputed and nothing is overwritten: the new jobs are given
exactly the subjects missing from the promoted files.

    python scripts/resplit_transfer.py            # print the plan
    python scripts/resplit_transfer.py --execute  # do it
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / "results" / "full104"
BACKUP = ROOT / "results" / "_resplit_backup"
BAD = {88, 89, 92, 100, 104}
PHYSIONET = [s for s in range(1, 110) if s not in BAD]
FRAMES = ("sensor", "reference")
WAYS = 2  # new jobs per chunk


def chunks(xs, n):
    size = -(-len(xs) // n)
    return [xs[i:i + size] for i in range(0, len(xs), size)]


def plan() -> list[dict]:
    """What each chunk has done, and what is left."""
    out = []
    for i, part in enumerate(chunks(PHYSIONET, 6), 1):
        tag = f"tr104_c{i:02d}"
        done_file = STAGE / f"transfer_folds_{tag}.csv"
        # Every file that already holds rows for this chunk: the finished one,
        # the checkpoint, and the part file a previous re-split promoted. All
        # of them count as done, or the new jobs would recompute them.
        srcs = [p for p in (
            done_file,
            STAGE / f"transfer_folds_{tag}.partial.csv",
            STAGE / f"transfer_folds_{tag}_part.csv",
        ) if p.exists()]
        srcs += sorted(STAGE.glob(f"transfer_folds_{tag}_???[0-9].csv"))
        done = set()
        for src in srcs:
            d = pd.read_csv(src)
            done |= {(str(r.frame), int(r.subject))
                     for r in d.itertuples(index=False)}
        src = srcs[0] if srcs else done_file
        todo = [(f, s) for f in FRAMES for s in part if (f, s) not in done]
        out.append({"tag": tag, "subjects": part, "src": src,
                    "complete": done_file.exists(), "done": done, "todo": todo})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--ways", type=int, default=WAYS,
                    help="new jobs per chunk (default 2)")
    args = ap.parse_args(argv)

    p = plan()
    total_todo = sum(len(c["todo"]) for c in p)
    print(f"{'chunk':8s} {'subjects':9s} {'done':>6s} {'left':>6s}  source")
    for c in p:
        print(f"  {c['tag']:6s} {len(c['subjects']):6d}    "
              f"{len(c['done']):6d} {len(c['todo']):6d}  "
              f"{c['src'].name if c['src'].exists() else 'nothing yet'}")
    print(f"\n{total_todo} subject-frame runs left, "
          f"currently across {len(p)} processes")

    # Frames left are almost always all-reference by this point, but do not
    # assume it: build one job per (chunk, way) over whatever is missing.
    jobs = []
    for c in p:
        if not c["todo"]:
            continue
        for f in FRAMES:
            subs = sorted({s for fr, s in c["todo"] if fr == f})
            if not subs:
                continue
            for w, group in enumerate(chunks(subs, args.ways), 1):
                if not group:
                    continue
                tag = f"{c['tag']}_{f[:3]}{w}"
                jobs.append({
                    "tag": tag, "frame": f, "subjects": group,
                    "cmd": (f"export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                            f"MKL_NUM_THREADS=1 PYTHONPATH=src; "
                            f"python -u -m qeeg.transfer --subjects 109 "
                            f"--start 1 --frames {f} --heldout "
                            f"{','.join(map(str, group))} "
                            f"--out {STAGE.relative_to(ROOT).as_posix()} "
                            f"--tag {tag} "
                            f"> results/run_{tag}.log 2>&1"),
                })
    print(f"would start {len(jobs)} jobs:")
    for j in jobs:
        print(f"  {j['tag']:18s} {j['frame']:9s} "
              f"{len(j['subjects']):2d} subjects: "
              f"{j['subjects'][0]}..{j['subjects'][-1]}")

    if not args.execute:
        print("\ndry run. Add --execute to stop the chunk jobs, promote their "
              "checkpoints and start these.")
        return 0

    # 1. Back up every checkpoint before anything can touch it.
    BACKUP.mkdir(parents=True, exist_ok=True)
    for c in p:
        if c["src"].exists():
            shutil.copy2(c["src"], BACKUP / c["src"].name)
    print(f"\nbacked up {len(list(BACKUP.glob('*.csv')))} files to "
          f"{BACKUP.relative_to(ROOT)}")

    # 2. Stop the chunk jobs. Their rows are already on disk.
    killed = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "$p = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
         "Where-Object { $_.CommandLine -match 'qeeg.transfer' }; "
         "$p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }; "
         "$p.Count"],
        capture_output=True, text=True).stdout.strip()
    print(f"stopped {killed} transfer processes")

    # 3. Promote each checkpoint to a part file, so the merge sees it. The rows
    # are complete subject-frame results; only the chunk is unfinished.
    for c in p:
        if c["complete"]:
            continue
        partial = STAGE / f"transfer_folds_{c['tag']}.partial.csv"
        if partial.exists():
            dest = STAGE / f"transfer_folds_{c['tag']}_part.csv"
            partial.rename(dest)
            print(f"  promoted {partial.name} -> {dest.name} "
                  f"({len(c['done'])} subject-frame rows)")

    # 4. Write the job list. It is NOT launched from here: a process started
    # from an editor or agent session is torn down with that session, which is
    # how two earlier attempts at this project's long runs died silently. The
    # queue goes through the Windows scheduler instead, via
    # scripts/run_resplit.sh.
    jobs_file = ROOT / "results" / "resplit_jobs.txt"
    jobs_file.write_text("\n".join(j["cmd"] for j in jobs) + "\n",
                         encoding="utf-8")
    print(f"wrote {len(jobs)} jobs to {jobs_file.relative_to(ROOT)}")
    print("start them with the scheduled task (no console, no session):")
    print("  bash scripts/run_resplit.sh      # or register it as a task")
    print("\nmerge when they finish with:")
    print("  PYTHONPATH=src python -m qeeg.transfer --merge "
          "'transfer_folds_tr104_*.csv' --out results/full104 --tag motor8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
