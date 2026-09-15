"""Pack the manuscript into upload-ready archives: anonymous and named.

Two archives are built from the same main.tex, so they cannot drift apart:

  submission_anonymous.zip  for double-anonymous peer review. main.tex has
                            IOP's [anonymous] class option set, every
                            comment line removed, and the contents of the
                            author, affiliation, email, acknowledgement,
                            funding, roles and data commands replaced by a
                            placeholder, so neither the PDF nor the source
                            names anyone. Its compiled PDF text is scanned for
                            identifying strings and the build fails on a hit.
  manuscript_named.zip      the full manuscript with authors, for Overleaf and
                            for the final files IOP request after acceptance.

Both follow IOP's upload rules (iopjournal-guidelines.pdf, section 1.1): one
flat directory with no subfolders; file names using only a-z, A-Z, 0-9 and
underscore; every file the build touches included (class, ORCID icon,
bibliography style, generated macros and tables, .bib, compiled .bbl and each
figure PDF the document includes).

    python paper/make_overleaf_zip.py            # build both
    python paper/make_overleaf_zip.py --check    # also compile each from a clean unpack

The anonymised supplementary code archive is a separate upload, built by
make_anonymous_code_zip.py.
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
sys.path.insert(0, str(PAPER))
from make_anonymous_code_zip import IDENTIFYING  # noqa: E402

CORE = ["macros_auto.tex", "tables_auto.tex", "refs.bib",
        "iopjournal.cls", "orcid.pdf", "iopart_num.bst"]
FIGURE_DIRS = [ROOT / "results" / "figures_paper", ROOT / "results" / "figures",
               PAPER / "figures"]
BBL = PAPER / "build" / "main.bbl"

# IOP: "only use characters a-z, A-Z, 0-9 and underscore. Do not use spaces."
SAFE_NAME = re.compile(r"^[A-Za-z0-9_]+\.[A-Za-z0-9]+$")

# Commands whose argument names people or places. The class already hides
# their output under [anonymous]; the anonymous source hides them as well.
PERSONAL_COMMANDS = ["author", "affil", "email", "ack", "funding", "roles", "data"]
PLACEHOLDER = "Removed for anonymous review."

# In the anonymous scan, the bibliography legitimately contains other
# people's URLs, so the generic "github.com/" entry is narrowed.
ANON_SCAN = [s for s in IDENTIFYING if s != "github.com/"] + ["github.com/imtiaj"]


def referenced_figures(tex: str) -> list[str]:
    tex = re.sub(r"(?m)%.*$", "", tex)
    return sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)))


def find_figure(name: str) -> Path:
    for d in FIGURE_DIRS:
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(f"figure {name} not found in {FIGURE_DIRS}")


def _replace_argument(tex: str, cmd: str, replacement: str) -> tuple[str, int]:
    """Replace the brace-balanced argument of every \\cmd{...} occurrence."""
    out, i, n = [], 0, 0
    pat = re.compile(r"\\" + cmd + r"\{")
    while True:
        m = pat.search(tex, i)
        if not m:
            out.append(tex[i:])
            break
        depth, j = 1, m.end()
        while depth:
            c = tex[j]
            if c == "\\":
                j += 2
                continue
            depth += {"{": 1, "}": -1}.get(c, 0)
            j += 1
        out.append(tex[i:m.end()])
        out.append(replacement)
        out.append("}")
        i, n = j, n + 1
    return "".join(out), n


def anonymise_source(tex: str) -> str:
    # 1. The class option.
    tex, k = re.subn(r"\\documentclass\{iopjournal\}",
                     r"\\documentclass[anonymous]{iopjournal}", tex, count=1)
    if k != 1:
        raise ValueError("could not set the [anonymous] class option")
    # 2. Full-line comments, which carry names, ORCIDs and working notes.
    tex = "\n".join(line for line in tex.splitlines()
                    if not line.lstrip().startswith("%")) + "\n"
    # 3. Arguments of the personal commands (definitions in the preamble use
    #    \renewcommand{\email}, which has no brace directly after \email).
    for cmd in PERSONAL_COMMANDS:
        tex, n = _replace_argument(tex, cmd, PLACEHOLDER)
        if cmd in {"author", "email", "roles"} and n == 0:
            raise ValueError(f"\\{cmd}{{...}} not found; the source layout changed")
    # 4. The named branch of \codestatement.
    tex, k = re.subn(r"\\url\{https://github\.com/[^}]*\}", "a public repository", tex)
    return tex


def build_zip(dest: Path, anonymous: bool) -> list[str]:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    main_text = anonymise_source(tex) if anonymous else tex

    sources: list[tuple[Path | None, str]] = [(None, "main.tex")]
    for name in CORE:
        src = PAPER / name
        if not src.exists():
            raise FileNotFoundError(
                f"{src} missing; run get_iop_class.sh and make_tables.py first")
        sources.append((src, name))
    if BBL.exists():
        sources.append((BBL, "main.bbl"))
    for fig in referenced_figures(tex):
        sources.append((find_figure(fig), Path(fig).name))

    arcs = [arc for _, arc in sources]
    bad = [a for a in arcs if not SAFE_NAME.match(a)]
    if bad:
        raise ValueError(f"file names IOP will reject: {bad}")
    if len(set(arcs)) != len(arcs):
        raise ValueError("name collision in a flat archive")

    if anonymous:
        texts = {"main.tex": main_text}
        texts.update({arc: src.read_text(encoding="utf-8", errors="replace")
                      for src, arc in sources
                      if src is not None and arc.endswith((".tex", ".bib", ".bbl"))})
        hits = {arc: [s for s in ANON_SCAN if s in t.lower()] for arc, t in texts.items()}
        hits = {k: v for k, v in hits.items() if v}
        if hits:
            raise ValueError(f"identifying strings in anonymous source: {hits}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for src, arc in sources:
            if src is None:
                z.writestr(arc, main_text)
            else:
                z.write(src, arc)
    return arcs


def check(dest: Path, anonymous: bool) -> int:
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
        verdict = (f"  check {dest.name}: exit={r.returncode} "
                   f"pages={pages.group(1) if pages else '?'} latex-warnings={warn} "
                   f"overfull={over} undefined={undef} flat=yes")
        if anonymous:
            try:
                import pypdf
                text = "".join((p.extract_text() or "")
                               for p in pypdf.PdfReader(str(tmp / "main.pdf")).pages).lower()
                found = [s for s in ANON_SCAN if s in text]
                verdict += f" pdf-identity-leaks={found or 'none'}"
                if found:
                    print(verdict)
                    return 1
            except ImportError:
                verdict += " pdf-identity-scan=skipped (pip install pypdf)"
        print(verdict)
        if r.returncode or not pages or undef:
            print(r.stdout[-2000:])
            return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outdir", default=str(PAPER / "build"))
    ap.add_argument("--check", action="store_true",
                    help="unpack each archive into a temp dir and compile it")
    args = ap.parse_args(argv)
    outdir = Path(args.outdir)
    status = 0
    for name, anonymous in (("submission_anonymous.zip", True),
                            ("manuscript_named.zip", False)):
        dest = outdir / name
        arcs = build_zip(dest, anonymous)
        print(f"wrote {dest} ({dest.stat().st_size // 1024} KB, {len(arcs)} files)")
        if args.check:
            if shutil.which("latexmk") is None:
                print("latexmk not on PATH; skipping compile check")
            else:
                status |= check(dest, anonymous)
    return status


if __name__ == "__main__":
    sys.exit(main())
