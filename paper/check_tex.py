"""Static sanity checks on the manuscript, for use without a LaTeX toolchain.

Catches the failure modes that would otherwise only surface on first compile:
undefined custom macros, citations with no bib entry, dangling cross-references
and unbalanced environments.

    python paper/check_tex.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

BS = "\\"

HERE = Path(__file__).parent
tex = (HERE / "main.tex").read_text(encoding="utf-8")
tab = (HERE / "tables_auto.tex").read_text(encoding="utf-8")
bib = (HERE / "refs.bib").read_text(encoding="utf-8")

problems: list[str] = []


def used(cmd: str, text: str) -> bool:
    return re.search(re.escape(BS + cmd) + r"(?![A-Za-z])", text) is not None


# ---------------------------------------------------------------- macros
defined = set(re.findall(re.escape(BS + "newcommand{" + BS) + r"(\w+)}", tab))
used_defined = sorted(c for c in defined if used(c, tex))
unused = sorted(defined - set(used_defined))

# Any \Word macro in main.tex that is neither a known LaTeX/iopart command
# nor one of ours is a likely typo.
IOP_OK = {
    # iopart / LaTeX structural commands
    "TODO", "JNE", "Tref", "Fref", "Sref", "References", "Keywords",
    "LaTeX", "TeX",
    # standard math symbols that happen to be capitalised
    "Delta", "Gamma", "Lambda", "Omega", "Phi", "Pi", "Psi", "Sigma",
    "Theta", "Upsilon", "Xi",
}
candidates = set(re.findall(re.escape(BS) + r"([A-Z][A-Za-z]+)(?![A-Za-z])", tex))
unknown = sorted(c for c in candidates if c not in defined and c not in IOP_OK)

print(f"macros defined in tables_auto.tex : {len(defined)}")
print(f"macros used in main.tex           : {len(used_defined)}")
print(f"  {used_defined}")
if unused:
    print(f"defined but unused (harmless)     : {unused}")
if unknown:
    print(f"UNKNOWN capitalised commands      : {unknown}")
    problems.append(f"unknown commands: {unknown}")

# ------------------------------------------------------------- citations
cites: set[str] = set()
for m in re.findall(re.escape(BS + "cite{") + r"([^}]*)}", tex):
    cites |= {c.strip() for c in m.split(",") if c.strip()}
keys = set(re.findall(r"@\w+\{([^,]+),", bib))

print(f"\ncitations used                    : {len(cites)}")
missing = sorted(cites - keys)
orphan = sorted(keys - cites)
print(f"missing from refs.bib             : {missing or 'none'}")
print(f"in refs.bib but never cited       : {orphan or 'none'}")
if missing:
    problems.append(f"missing bib entries: {missing}")

# ---------------------------------------------------------------- labels
labels = set(re.findall(re.escape(BS + "label{") + r"([^}]*)}", tex + tab))
refs: set[str] = set()
for pat in ("Tref", "Fref", "Sref", "ref", "eref"):
    refs |= set(re.findall(re.escape(BS + pat + "{") + r"([^}]*)}", tex))

print(f"\nlabels defined                    : {sorted(labels)}")
print(f"refs used                         : {sorted(refs)}")
dangling = sorted(refs - labels)
unref = sorted(labels - refs)
print(f"dangling refs                     : {dangling or 'none'}")
print(f"labels never referenced           : {unref or 'none'}")
if dangling:
    problems.append(f"dangling refs: {dangling}")

# ---------------------------------------------------------- environments
print("\nenvironment balance:")
both = tex + tab
for env in ("document", "abstract", "table", "figure", "tabular",
            "indented", "equation", "align"):
    b = len(re.findall(re.escape(BS + "begin{" + env + "}"), both))
    e = len(re.findall(re.escape(BS + "end{" + env + "}"), both))
    flag = "" if b == e else "   <-- MISMATCH"
    print(f"  {env:10s} begin={b} end={e}{flag}")
    if b != e:
        problems.append(f"unbalanced environment: {env} ({b}/{e})")

for name, text in (("main.tex", tex), ("tables_auto.tex", tab)):
    ok = text.count("{") == text.count("}")
    print(f"braces balanced in {name:16s}: {ok} "
          f"({text.count('{')} open / {text.count('}')} close)")
    if not ok:
        problems.append(f"unbalanced braces in {name}")

# ------------------------------------------------------------- figures
figs = set(re.findall(re.escape(BS + "includegraphics") + r"(?:\[[^\]]*\])?\{([^}]*)\}", tex))
print(f"\nfigures referenced                : {sorted(figs)}")
for f in sorted(figs):
    hit = list((HERE.parent / "results" / "figures").glob(f + "*")) or \
          list((HERE / "figures").glob(f + "*"))
    status = "found" if hit else "MISSING"
    print(f"  {f:32s} {status}")
    if not hit:
        problems.append(f"missing figure: {f}")

print()
if problems:
    print(f"FAILED: {len(problems)} problem(s)")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)
print("All static checks passed.")
