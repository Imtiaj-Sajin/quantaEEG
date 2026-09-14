"""Pack a self-contained Overleaf upload of the manuscript.

Overleaf needs everything the build touches and nothing else: the class file
and its ORCID icon (IOP's, not on CTAN), the bibliography style, the generated
macro and table files, the .bib, and every figure PDF the document actually
includes. Figures are placed under ``figures/`` because ``\\graphicspath`` in
main.tex lists that directory last, after the two ``results/`` locations that
do not exist on Overleaf.

    python paper/make_overleaf_zip.py            # -> paper/build/quantaEEG-overleaf.zip
    python paper/make_overleaf_zip.py --check    # also unpack and compile it

The --check pass is the point: a zip that compiles here compiles on Overleaf.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent

CORE = ["main.tex", "macros_auto.tex", "tables_auto.tex", "refs.bib",
        "iopjournal.cls", "orcid.pdf", "iopart-num.bst"]
FIGURE_DIRS = [ROOT / "results" / "figures_paper", ROOT / "results" / "figures",
               PAPER / "figures"]


def referenced_figures() -> list[str]:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    tex = re.sub(r"(?m)%.*$", "", tex)
    return sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)))


def find_figure(name: str) -> Path:
    for d in FIGURE_DIRS:
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(f"figure {name} not found in {FIGURE_DIRS}")


def build_zip(dest: Path) -> list[str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    listing = []
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for name in CORE:
            src = PAPER / name
            if not src.exists():
                raise FileNotFoundError(
                    f"{src} missing; run get_iop_class.sh and make_tables.py first")
            z.write(src, name)
            listing.append(name)
        for fig in referenced_figures():
            src = find_figure(fig)
            arc = f"figures/{fig}"
            z.write(src, arc)
            listing.append(arc)
    return listing


def check(dest: Path) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(dest) as z:
            z.extractall(tmp)
        cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"]
        r = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
        log = (tmp / "main.log").read_text(encoding="utf-8", errors="replace") \
            if (tmp / "main.log").exists() else ""
        pages = re.search(r"Output written on .*?\((\d+) pages", log)
        warn = len(re.findall(r"LaTeX Warning", log))
        undef = len(re.findall(r"undefined", log))
        print(f"  check: exit={r.returncode} pages={pages.group(1) if pages else '?'} "
              f"latex-warnings={warn} undefined={undef}")
        if r.returncode or not pages or undef:
            print(r.stdout[-2000:])
            return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(PAPER / "build" / "quantaEEG-overleaf.zip"))
    ap.add_argument("--check", action="store_true",
                    help="unpack into a temp dir and compile with latexmk")
    args = ap.parse_args(argv)
    dest = Path(args.out)
    listing = build_zip(dest)
    print(f"wrote {dest} ({dest.stat().st_size // 1024} KB, {len(listing)} files)")
    for name in listing:
        print(f"  {name}")
    if args.check:
        if shutil.which("latexmk") is None:
            print("latexmk not on PATH; skipping compile check")
            return 0
        return check(dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
