"""Pack the manuscript into ONE upload-ready archive: paper/build/submission.zip.

The same zip is what you upload to the journal and what you import into
Overleaf. It follows IOP's upload rules (iopjournal-guidelines.pdf,
section 1.1):

* every file sits in one flat directory, no subfolders;
* file names use only a-z, A-Z, 0-9 and underscore (plus the extension);
* every file the build touches is included: the class and its ORCID icon
  (IOP's, not on CTAN), the bibliography style, the generated macro and table
  files, the .bib, the compiled .bbl and each figure PDF the document uses.

    python paper/make_overleaf_zip.py            # build it
    python paper/make_overleaf_zip.py --check    # build it and compile it from a clean unpack

The --check pass is the point: a zip that compiles from an empty folder here
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

CORE = ["main.tex", "macros_auto.tex", "refs.bib",
        "iopjournal.cls", "orcid.pdf", "iopart_num.bst"]
# Tables are one generated file each, so that every table can be inputted
# beside the text that discusses it rather than queued at the top of Results.
# They are collected by glob because their number changes with the analysis.
TABLE_GLOB = "tab_*_auto.tex"
# The supplement's tables are generated under their own prefix, so the
# main glob does not see them.
SUPP_TABLE_GLOB = "supp_table_*_auto.tex"
FIGURE_DIRS = [ROOT / "results" / "figures_paper", ROOT / "results" / "figures",
               PAPER / "figures"]
def _bbl() -> Path | None:
    """The bibliography latexmk actually produced, never an empty leftover.

    latexmk is run in paper/, so paper/main.bbl is the live one. A zero-byte
    paper/build/main.bbl survives from an earlier -outdir attempt, which is the
    BibTeX path trap paper/README.md documents, and the zip was shipping it: an
    upload whose bibliography was empty. --check could not catch it, because
    refs.bib is in the archive and latexmk simply reran BibTeX in the temp
    directory, so the empty file never mattered there and would have mattered
    on the journal's system.
    """
    for p in (PAPER / "main.bbl", PAPER / "build" / "main.bbl"):
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


BBL = _bbl()

# IOP: "only use characters a-z, A-Z, 0-9 and underscore. Do not use spaces."
SAFE_NAME = re.compile(r"^[A-Za-z0-9_]+\.[A-Za-z0-9]+$")


def referenced_figures(*tex_files: str) -> list[str]:
    """Figures included by the given .tex files, comments stripped.

    Defaults to main.tex. The supplement is passed explicitly, because it
    includes four figures of its own that main.tex never mentions.
    """
    names: set[str] = set()
    for fname in (tex_files or ("main.tex",)):
        path = PAPER / fname
        if not path.exists():
            continue
        tex = re.sub(r"(?m)%.*$", "", path.read_text(encoding="utf-8"))
        names |= set(re.findall(
            r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex))
    return sorted(names)


def find_figure(name: str) -> Path:
    for d in FIGURE_DIRS:
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(f"figure {name} not found in {FIGURE_DIRS}")


def build_zip(dest: Path) -> list[str]:
    sources: list[tuple[Path, str]] = []
    for name in CORE:
        src = PAPER / name
        if not src.exists():
            raise FileNotFoundError(
                f"{src} missing; run get_iop_class.sh and make_tables.py first")
        sources.append((src, name))
    tables = sorted(PAPER.glob(TABLE_GLOB))
    if not tables:
        raise FileNotFoundError(
            f"no {TABLE_GLOB} in {PAPER}; run python paper/make_tables.py first")
    for t in tables:
        sources.append((t, t.name))
    if BBL is not None:
        sources.append((BBL, "main.bbl"))
    # main.tex cites supplementary tables and figures through xr, which reads
    # supplementary.aux; without it every such reference prints as ??.
    supp_aux = PAPER / "build" / "supplementary.aux"
    if not supp_aux.exists():
        raise FileNotFoundError(
            f"{supp_aux} missing; build supplementary.tex before packing")
    sources.append((supp_aux, "supplementary.aux"))

    # The supplement itself. Shipping only supplementary.aux resolves the
    # cross-references in main.tex and then sends a paper that cites four
    # tables and four figures nobody can read: a referee flagged exactly that
    # on the first submission. The source, its generated tables and the built
    # PDF all travel, so the supplement can be both read and rebuilt.
    supp_tex = PAPER / "supplementary.tex"
    if not supp_tex.exists():
        raise FileNotFoundError(f"{supp_tex} missing")
    sources.append((supp_tex, "supplementary.tex"))

    supp_tables = sorted(PAPER.glob(SUPP_TABLE_GLOB))
    if not supp_tables:
        raise FileNotFoundError(
            f"no {SUPP_TABLE_GLOB} in {PAPER}; run python paper/make_tables.py")
    for t in supp_tables:
        sources.append((t, t.name))

    supp_pdf = PAPER / "build" / "supplementary.pdf"
    if not supp_pdf.exists():
        raise FileNotFoundError(
            f"{supp_pdf} missing; build supplementary.tex before packing")
    sources.append((supp_pdf, "supplementary.pdf"))

    # Figures from both documents, deduplicated in case one is ever shared.
    for fig in referenced_figures("main.tex", "supplementary.tex"):
        sources.append((find_figure(fig), Path(fig).name))
    # The architecture figure's editable source travels with the manuscript, so
    # a co-author or a later reader can change it rather than being stuck with
    # a flat PDF. IOP's file names allow only letters, digits and underscore,
    # and .drawio is XML, so it ships as architecture_drawio.xml.
    drawio = ROOT / "figures" / "architecture.drawio"
    if drawio.exists():
        sources.append((drawio, "architecture_drawio.xml"))

    arcs = [arc for _, arc in sources]
    bad = [a for a in arcs if not SAFE_NAME.match(a)]
    if bad:
        raise ValueError(f"file names IOP will reject: {bad}")
    if len(set(arcs)) != len(arcs):
        raise ValueError("name collision in a flat archive")

    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc in sources:
            z.write(src, arc)
    return arcs


def check(dest: Path) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(dest) as z:
            if any("/" in n for n in z.namelist()):
                print("  check: archive contains a subfolder")
                return 1
            z.extractall(tmp)
        cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"]
        r = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True)
        log = (tmp / "main.log").read_text(encoding="utf-8", errors="replace") \
            if (tmp / "main.log").exists() else ""
        pages = re.search(r"Output written on .*?\((\d+)\s+pages", log, re.S)
        warn = len(re.findall(r"LaTeX Warning", log))
        over = len(re.findall(r"Overfull", log))
        undef = len(re.findall(r"undefined", log))
        print(f"  check: exit={r.returncode} pages={pages.group(1) if pages else '?'} "
              f"latex-warnings={warn} overfull={over} undefined={undef} flat=yes")
        if r.returncode or not pages or undef:
            print(r.stdout[-2000:])
            return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(PAPER / "build" / "submission.zip"))
    ap.add_argument("--check", action="store_true",
                    help="unpack into a temp dir and compile with latexmk")
    args = ap.parse_args(argv)
    dest = Path(args.out)
    arcs = build_zip(dest)
    print(f"wrote {dest} ({dest.stat().st_size // 1024} KB, {len(arcs)} files)")
    if args.check:
        if shutil.which("latexmk") is None:
            print("latexmk not on PATH; skipping compile check")
            return 0
        return check(dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
