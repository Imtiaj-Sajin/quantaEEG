"""Keep the EEG downloads inside the project, in a gitignored datasets/ folder.

Why
---
The three datasets are about 12 GB and they live in ``~/mne_data`` by default,
which on this machine is the C: drive with under 7 GB free. Re-downloading them
costs hours (PhysioNet is roughly 50 s per file, three files per subject, 104
subjects). Moving the cache into ``datasets/`` next to the code puts the data
on the same drive as the project, frees the system drive, and makes it obvious
to anyone who clones the repo where the data is meant to go. ``datasets/`` is
in .gitignore: the files never reach the remote.

Nothing in the code reads a hard-coded path. MNE resolves the cache through two
config keys, ``MNE_DATA`` (which MOABB also honours, for IV-2a and Cho2017) and
``MNE_DATASETS_EEGBCI_PATH`` (for PhysioNet EEGMMIDB). This script moves the
folders and repoints both keys.

The hardlink trap
-----------------
MOABB builds a BIDS-style view of IV-2a and Cho2017 under ``NEMAR/``, whose
.mat files are *hardlinks* to the files in ``MNE-bnci-data`` and
``MNE-gigadb-data``, not copies. On disk the cache is 12 GB; add up the file
sizes naively and you get 22 GB, because the same 10 GB is counted twice.
Neither robocopy nor shutil preserves hardlinks, so a plain move would silently
turn one shared copy into two real ones and land 22 GB on the target drive.
This script therefore records every link's target first, moves the real data,
and then rebuilds ``NEMAR/`` as links into the moved files. It verifies the
link count afterwards: every rebuilt file must report two names, not one.

Usage
-----
    python scripts/use_local_datasets.py --check    # report, change nothing
    python scripts/use_local_datasets.py --move     # move, then repoint

--move refuses to run while any other Python process is alive, because MOABB
reads these files at job start and moving a file out from under a running
benchmark loses the run. Check with --check first.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "datasets"
KEYS = ("MNE_DATA", "MNE_DATASETS_EEGBCI_PATH")
# The real data, one folder per source. NEMAR is handled separately: it holds
# no data of its own, only links into these.
REAL = ("MNE-eegbci-data", "MNE-bnci-data", "MNE-gigadb-data")
LINKED = "NEMAR"


def _ps(script: str) -> str:
    r = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                       capture_output=True, text=True, timeout=600)
    return r.stdout


def _du(p: Path) -> int:
    """Bytes on disk, counting a file with several names only once."""
    seen, total = set(), 0
    for f in p.rglob("*"):
        try:
            st = f.stat()
        except OSError:
            continue
        if not f.is_file():
            continue
        key = (st.st_dev, st.st_ino)
        if st.st_nlink > 1:
            if key in seen:
                continue
            seen.add(key)
        total += st.st_size
    return total


def _gb(n: int) -> str:
    return f"{n / 1024 ** 3:.2f} GB"


def _other_pythons() -> int:
    """Count live python processes other than this one.

    tasklist under Git Bash has returned empty output on this machine, so ask
    PowerShell, which has been reliable.
    """
    out = _ps("(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\")"
              ".ProcessId -join ','")
    pids = {int(x) for x in out.strip().split(",") if x.strip().isdigit()}
    return len(pids - {os.getpid()})


def _src() -> Path:
    import mne
    return Path(mne.get_config("MNE_DATA") or (Path.home() / "mne_data"))


def _links(src: Path) -> dict[str, str]:
    """Map each linked file under NEMAR to its target, both relative to src.

    A hardlink has no direction: PowerShell reports the file's *other* names,
    so the target is whichever of them lies under one of the real folders.
    """
    base = src / LINKED
    if not base.exists():
        return {}
    out = _ps(f"Get-ChildItem -LiteralPath '{base}' -Recurse -File -Force | "
              f"ForEach-Object {{ $_.FullName + '|' + ($_.Target -join ';') }}")
    mapping = {}
    for line in out.splitlines():
        if "|" not in line:
            continue
        full, targets = line.split("|", 1)
        rel = str(Path(full).relative_to(src))
        for t in targets.split(";"):
            t = t.strip()
            if not t:
                continue
            try:
                trel = Path(t).relative_to(src)
            except ValueError:
                continue
            if trel.parts and trel.parts[0] in REAL:
                mapping[rel] = str(trel)
                break
    return mapping


def check() -> int:
    import mne
    src = _src()
    print(f"config    MNE_DATA = {mne.get_config('MNE_DATA')}")
    print(f"config    MNE_DATASETS_EEGBCI_PATH = "
          f"{mne.get_config('MNE_DATASETS_EEGBCI_PATH')}")
    print(f"project   {DEST}  {'exists' if DEST.exists() else 'not created yet'}")
    cached = 0
    for name in REAL + (LINKED,):
        for base, label in ((src, "cache"), (DEST, "project")):
            p = base / name
            if p.exists():
                n = _du(p)
                if base == src and name in REAL:
                    cached += n
                note = "" if name in REAL else "  (links, shared with the above)"
                print(f"  {label:8s} {name:18s} {_gb(n):>10s}{note}")
    links = _links(src)
    print(f"  {len(links)} hardlinked files under {LINKED}/ "
          f"(they cost no extra disk)")
    print(f"  real data still on the system drive: {_gb(cached)}")
    live = _other_pythons()
    print(f"  other python processes running: {live}"
          f"{'  (a move would be unsafe)' if live else ''}")
    return 0


def move() -> int:
    import mne
    live = _other_pythons()
    if live:
        print(f"refusing: {live} other python processes are running. These "
              f"datasets are read at job start, and moving a file out from "
              f"under a running benchmark loses the run. Wait for the queue "
              f"to drain, then rerun.")
        return 1
    src = _src()
    if src.resolve() == DEST.resolve():
        print("already pointed at the project folder; nothing to move")
        return 0
    links = _links(src)
    print(f"recorded {len(links)} hardlinks under {LINKED}/ before moving")
    DEST.mkdir(parents=True, exist_ok=True)

    for name in REAL:
        s, d = src / name, DEST / name
        if not s.exists():
            continue
        if d.exists():
            print(f"  {name}: already in the project, left the cache copy alone")
            continue
        print(f"  moving {name} ({_gb(_du(s))}) ...", flush=True)
        # robocopy rather than shutil.move: this is a cross-drive move of about
        # 10 GB, and robocopy retries, reports, and deletes each source file
        # only after it has copied it. /XJ so it never follows a junction.
        r = subprocess.run(["robocopy", str(s), str(d), "/E", "/MOVE", "/XJ",
                            "/R:2", "/W:2", "/NFL", "/NDL", "/NJH", "/NP"],
                           capture_output=True, text=True)
        if r.returncode >= 8:  # robocopy: under 8 is success
            print(f"    robocopy failed ({r.returncode}); source left in place")
            print(r.stdout[-2000:])
            return 1
        print(f"    done ({_gb(_du(d))} in the project)")

    # Rebuild the BIDS-style view as links into the moved data, rather than
    # copying 10 GB a second time.
    if (src / LINKED).exists() and not (DEST / LINKED).exists():
        print(f"  rebuilding {LINKED}/ as links into the moved data ...")
        made = copied = 0
        for f in sorted((src / LINKED).rglob("*")):
            if not f.is_file():
                continue
            rel = str(f.relative_to(src))
            dst = DEST / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            target = links.get(rel)
            if target and (DEST / target).exists():
                os.link(DEST / target, dst)
                made += 1
            else:
                shutil.copy2(f, dst)
                copied += 1
        bad = [rel for rel in links
               if (DEST / rel).exists() and (DEST / rel).stat().st_nlink < 2]
        if bad:
            print(f"    {len(bad)} files did not come back as links, e.g. "
                  f"{bad[0]}; leaving the old cache in place")
            return 1
        print(f"    {made} links, {copied} small files copied; removing the old "
              f"{LINKED}/")
        shutil.rmtree(src / LINKED)

    for k in KEYS:
        mne.set_config(k, str(DEST))
    print(f"repointed {' and '.join(KEYS)} to {DEST}")
    print("verify with: python scripts/use_local_datasets.py --check")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--move", action="store_true")
    args = ap.parse_args(argv)
    if args.move:
        return move()
    return check()


if __name__ == "__main__":
    sys.exit(main())
