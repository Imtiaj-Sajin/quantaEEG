"""Pack a self-contained, submission-safe upload of the manuscript.

The same archive serves Overleaf and IOP's ScholarOne system, so it follows
IOP's own upload rules (iopjournal-guidelines.pdf, section 1.1):

* every file sits in ONE flat directory, no subfolders;
* file names use only a-z, A-Z, 0-9 and underscore (plus the extension);
* every file the build touches is included.

That means the class and its ORCID icon (IOP's, not on CTAN), the
bibliography style, the generated macro and table files, the .bib, the
compiled .bbl (submission systems do not always run BibTeX), and every figure
PDF the document actually includes. Figures land beside main.tex, where
\\includegraphics finds them before it consults \\graphicspath.

    python paper/make_overleaf_zip.py            # -> paper/build/quantaEEG_submission.zip
    python paper/make_overleaf_zip.py --check    # also unpack and compile it

The --check pass is the point: a zip that compiles from a clean unpack here
compiles on Overleaf and in the submission system.
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
        "iopjournal.cls", "orcid.pdf", "iopart_num.bst"]
FIGURE_DIRS = [ROOT / "results" / "figures_paper", ROOT / "results" / "figures",
               PAPER / "figures"]
BBL = PAPER / "build" / "main.bbl"

# IOP: "only use characters a-z, A-Z, 0-9 and underscore. Do not use spaces."
SAFE_NAME = re.compile(r"^[A-Za-z0-9_]+\.[A-Za-z0-9]+$")


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
    sources: list[tuple[Path, str]] = []
    for name in CORE:
        src = PAPER / name
        if not src.exists():
            raise FileNotFoundError(
                f"{src} missing; run get_iop_class.sh and make_tables.py first")
        sources.append((src, name))
    if BBL.exists():
        sources.append((BBL, "main.bbl"))
    for fig in referenced_figures():
        sources.append((find_figure(fig), Path(fig).name))

    bad = [arc for _, arc in sources if not SAFE_NAME.match(arc)]
    if bad:
        raise ValueError(f"file names IOP will reject: {bad}")
    dupes = {a for _, a in sources if sum(1 for _, b in sources if b == a) > 1}
    if dupes:
        raise ValueError(f"name collision in a flat archive: {sorted(dupes)}")

    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc in sources:
            z.write(src, arc)
    return [arc for _, arc in sources]


def check(dest: Path) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(dest) as z:
            names = z.namelist()
            if any("/" in n for n in names):
                print("  check: archive contains a subfolder")
                return 1
            z.extractall(tmp)
        cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"]
        r = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
        log = (tmp / "main.log").read_text(encoding="utf-8", errors="replace") \
            if (tmp / "main.log").exists() else ""
        pages = re.search(r"Output written on .*?\((\d+) pages", log)
        warn = len(re.findall(r"LaTeX Warning", log))
        undef = len(re.findall(r"undefined", log))
        print(f"  check: exit={r.returncode} pages={pages.group(1) if pages else '?'} "
              f"latex-warnings={warn} undefined={undef} flat=yes")
        if r.returncode or not pages or undef:
            print(r.stdout[-2000:])
            return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(PAPER / "build" / "quantaEEG_submission.zip"))
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
