"""Build the anonymised supplementary code archive for double-anonymous review.

Reviewers should be able to rerun the study without learning who wrote it.
The archive therefore carries only what reproduction needs (the qeeg package,
the table scripts, the batch helper, the pinned requirements and every result
CSV/JSON) plus a fresh README, and it leaves out everything that names a
person, an institution or the public repository: the project notes, git
history, logs with local user paths, compiled bytecode and the manuscript
source.

The file list is not trusted. Every archived file is scanned for identifying
strings, and the build fails if any is found, so a future edit that adds a
name to a docstring cannot slip into a review submission.

    python paper/make_anonymous_code_zip.py
    -> paper/build/supplementary_code.zip
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent

# Strings that would identify the authors, their institution or the repo.
# Matched case-insensitively against the text of every archived file.
IDENTIFYING = [
    "imtiaj", "sajin", "samiul", "saif", "chayon", "hasibur", "wahiduzzaman",
    "suva", "abha", "moula", "aiub", "iubat", "wshuvo", "quantaeeg",
    "github.com/", "0009-0009-2423-1835", "0009-0009-7116-7151",
    "0000-0003-3741-6651", "26-9408", "26-9409",
]

INCLUDE_GLOBS = [
    "src/qeeg/*.py",
    "paper/make_tables.py",
    "paper/cross_tables.py",
    "paper/reference_tables.py",
    "scripts/*.sh",
    "requirements.txt",
    "constraints.txt",
    "results/*.csv",
    "results/*.json",
]

README = """Supplementary code for the submitted manuscript
==================================================

This archive reproduces every table, figure and inline number in the
manuscript from the per-fold result files it contains. It has been anonymised
for double-anonymous review; the public repository will be linked on
publication.

Contents
--------
src/qeeg/        the analysis package: data loading, density-matrix and
                 circuit kernels, the reference-state whitening, every
                 benchmark suite, statistics and figure code
paper/           the scripts that turn results/*.csv into the LaTeX tables
                 and the macros quoted in the text
results/         per-fold scores and summaries for every experiment
scripts/         a helper for running dataset batches in parallel
requirements.txt pinned dependencies (constraints.txt pins scipy)

Reproduce the tables and inline numbers (seconds, no EEG download needed)
-------------------------------------------------------------------------
    pip install -r requirements.txt -c constraints.txt
    python paper/make_tables.py

This writes paper/tables_auto.tex and paper/macros_auto.tex.

Reproduce the figures
---------------------
    PYTHONPATH=src python -m qeeg.figures --paper
    PYTHONPATH=src python -m qeeg.figures_reference --paper

Rerun the experiments (downloads the public EEG datasets)
----------------------------------------------------------
    PYTHONPATH=src python -u -m qeeg.benchmark --subjects 30 --splits 5 --repeats 3
    PYTHONPATH=src python -u -m qeeg.benchmark --suite extended --subjects 30
    PYTHONPATH=src python -u -m qeeg.benchmark --dataset bci2a --suite extended
    PYTHONPATH=src python -u -m qeeg.benchmark --dataset cho2017 --suite extended
    PYTHONPATH=src python -u -m qeeg.transfer
    PYTHONPATH=src python -u -m qeeg.crosssession
    PYTHONPATH=src python -u -m qeeg.equivalence

Each module's docstring states its protocol. For parallel batches set
OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1 and MKL_NUM_THREADS=1, and merge
batch outputs with `python -m qeeg.merge`.
"""


def collect() -> list[Path]:
    files: set[Path] = set()
    for pattern in INCLUDE_GLOBS:
        files.update(p for p in ROOT.glob(pattern) if p.is_file())
    return sorted(files)


def leaks(text: str) -> list[str]:
    low = text.lower()
    return [s for s in IDENTIFYING if s in low]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(PAPER / "build" / "supplementary_code.zip"))
    args = ap.parse_args(argv)

    files = collect()
    problems = []
    for p in files:
        try:
            hit = leaks(p.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            hit = leaks(p.read_bytes().decode("latin-1"))
        if hit:
            problems.append(f"{p.relative_to(ROOT)}: {hit}")
    if leaks(README):
        problems.append(f"README: {leaks(README)}")
    if problems:
        print("REFUSING to build: identifying strings found")
        for line in problems:
            print("  " + line)
        return 1

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("README.txt", README)
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
    # Re-open and rescan what was actually written, names included.
    with zipfile.ZipFile(dest) as z:
        for name in z.namelist():
            if leaks(name) or leaks(z.read(name).decode("utf-8", "replace")):
                print(f"REFUSING: leak in archived member {name}")
                dest.unlink()
                return 1
    size_mb = dest.stat().st_size / 1e6
    print(f"wrote {dest} ({size_mb:.1f} MB, {len(files) + 1} files), "
          f"scanned clean for {len(IDENTIFYING)} identifying strings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
