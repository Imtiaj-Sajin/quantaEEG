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
# One generated file per table, so each can be inputted beside the text that
# discusses it. The checks below want them as one body of text.
tab = "\n".join(p.read_text(encoding="utf-8")
                for p in sorted(HERE.glob("tab_*_auto.tex")))
if not tab.strip():
    raise SystemExit("no tab_*_auto.tex found: run python paper/make_tables.py")
mac = (HERE / "macros_auto.tex").read_text(encoding="utf-8")
bib = (HERE / "refs.bib").read_text(encoding="utf-8")

# Generated content is split: macros are read in the preamble, table floats in
# the body. Most checks care about the union.
gen = mac + "\n" + tab

problems: list[str] = []


def used(cmd: str, text: str) -> bool:
    return re.search(re.escape(BS + cmd) + r"(?![A-Za-z])", text) is not None


# ---------------------------------------------------------------- macros
defined = set(re.findall(re.escape(BS + "newcommand{" + BS) + r"(\w+)}", gen))
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

print(f"macros defined in macros_auto.tex : {len(defined)}")
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
labels = set(re.findall(re.escape(BS + "label{") + r"([^}]*)}", tex + gen))
refs: set[str] = set()
# iopart supplies both cases of each cross-reference macro and the manuscript
# uses the lowercase ones (\sref, \eref) far more than the capitalised forms.
# Listing only the capitalised spellings silently missed every \sref in the
# document, which is exactly the dangling-reference class this script exists
# to catch, so match the macro name case-insensitively.
for pat in ("Tref", "Fref", "Sref", "Eref", "ref"):
    refs |= set(re.findall(
        re.escape(BS) + pat + r"\{([^}]*)\}", tex, flags=re.IGNORECASE))

# Labels defined in the supplementary document, cited from main.tex through
# xr with the prefix S-.
supp_text = ""
for f in [HERE / "supplementary.tex", *sorted(HERE.glob("supp_table_*_auto.tex"))]:
    if f.exists():
        supp_text += f.read_text(encoding="utf-8")
labels |= {"S-" + l for l in re.findall(re.escape(BS + "label{") + r"([^}]*)}", supp_text)}

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
both = tex + gen
for env in ("document", "abstract", "table", "figure", "tabular",
            "indented", "equation", "align"):
    b = len(re.findall(re.escape(BS + "begin{" + env + "}"), both))
    e = len(re.findall(re.escape(BS + "end{" + env + "}"), both))
    flag = "" if b == e else "   <-- MISMATCH"
    print(f"  {env:10s} begin={b} end={e}{flag}")
    if b != e:
        problems.append(f"unbalanced environment: {env} ({b}/{e})")

for name, text in (("main.tex", tex), ("tables_auto.tex", tab),
                   ("macros_auto.tex", mac)):
    ok = text.count("{") == text.count("}")
    print(f"braces balanced in {name:16s}: {ok} "
          f"({text.count('{')} open / {text.count('}')} close)")
    if not ok:
        problems.append(f"unbalanced braces in {name}")

# --------------------------------------------------------------- dashes
# House rule: no em-dashes anywhere in the manuscript or the project. That
# covers the Unicode character, LaTeX's --- ligature, \textemdash, and a
# spaced double hyphen used as punctuation. Ranges (8--30) are en-dashes and
# are fine. Comments are stripped first so banner rules like %% ---- pass.
print("\nem-dash rule:")
EM_CHARS = (chr(0x2014), chr(0x2015))  # em-dash, horizontal bar


def _strip_tex_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


dash_hits = []
for name, text in (("main.tex", tex), ("tables_auto.tex", tab),
                   ("macros_auto.tex", mac), ("refs.bib", bib),
                   ("supplementary material", supp_text)):
    body = _strip_tex_comments(text)
    for i, line in enumerate(body.splitlines(), 1):
        if any(c in line for c in EM_CHARS) or "\\textemdash" in line \
                or re.search(r"(?<!-)---(?!-)", line) \
                or re.search(r"\S\s+--\s+\S", line):
            dash_hits.append(f"{name}:{i}: {line.strip()[:80]}")
# Repo-wide: the Unicode character in any text file we own.
for p in HERE.parent.rglob("*"):
    if any(part in {".git", "__pycache__", "build"} for part in p.parts) or not p.is_file():
        continue
    if p.suffix not in {".py", ".md", ".tex", ".bib", ".sh", ".txt", ".csv", ".json"}:
        continue
    try:
        s = p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        continue
    if any(c in s for c in EM_CHARS):
        dash_hits.append(f"{p.relative_to(HERE.parent)}: contains an em-dash character")
print(f"  {'none found' if not dash_hits else str(len(dash_hits)) + ' found'}")
for h in dash_hits[:20]:
    print(f"    {h}")
if dash_hits:
    problems.append(f"em-dashes present: {len(dash_hits)}")

# ------------------------------------------------- rendering regressions
# Each of these shipped in a clean build and was quoted back by a referee.
# They are prose around generated numbers, so no macro could catch them.
print("\nrendering regressions:")
macro_val = dict(re.findall(r"(?m)^" + re.escape(BS + "newcommand{" + BS)
                            + r"(\w+)\}\{(.*)\}\s*$", mac))
prose = _strip_tex_comments(tex)
regress: list[str] = []


def _num(v: str) -> float | None:
    v = v.replace(BS + "ensuremath{-}", "-")
    m = re.fullmatch(r"\s*[+-]?\d+(?:\.\d+)?\s*", v)
    return float(v) if m else None


# 1. A relation typed in front of a macro that already carries one, which
#    renders "all p <= < 0.001".
for rel, name in re.findall(r"(\\le\b|\\leq\b|\\ge\b|\\geq\b|=|<|>)\s*"
                            + re.escape(BS) + r"(\w+)", prose):
    v = macro_val.get(name, "")
    if v.startswith(BS + "ensuremath{") and re.search(r"[<>=]|\\le|\\ge", v[:30]):
        regress.append(f"relation '{rel}' typed before \\{name}, which already "
                       f"carries one ({v})")

# 2. "up to \Macro" where the macro is zero: "dipping by up to 0 %".
for name in re.findall(r"up to\s+" + re.escape(BS) + r"(\w+)", prose):
    x = _num(macro_val.get(name, ""))
    if x is not None and x == 0:
        regress.append(f"'up to \\{name}' renders as up to 0")

# 3. The equivalence margin described as pre-specified; section 2.9 and the
#    TOST tables say it was not pre-registered.
for text, where in ((prose, "main.tex"), (gen, "generated tables")):
    if re.search(r"pre-?specified[^.]{0,60}margin|pre-?specified\s*\$?\\pm",
                 text, re.I):
        regress.append(f"{where}: the equivalence margin is called pre-specified")

# 4. A generated quantity said to be inside the margin when it is not:
#    "The largest difference, 0.029, is inside the +-0.02 margin".
margin = _num(macro_val.get("EquivMargin", ""))
for name in re.findall(re.escape(BS) + r"(\w+)(?:\{\})?,?\s+(?:is|lies|stays)\s+"
                       r"inside\s+the\s+\$\\pm\\EquivMargin", prose):
    x = _num(macro_val.get(name, ""))
    if margin is not None and x is not None and abs(x) >= margin:
        regress.append(f"\\{name} = {x} is described as inside the "
                       f"+-{margin:g} margin")

# 5. "conventionally significant" right after a p-value that is not.
for name, after in re.findall(r"\$p" + re.escape(BS) + r"(\w+)\$(.{0,160})",
                              prose.replace("\n", " ")):
    v = macro_val.get(name, "")
    m = re.search(r"(\d+\.\d+)\s*$", v)
    if (m and "=" in v and float(m.group(1)) >= 0.05
            and re.search(r"\bconventionally significant", after)
            and not re.search(r"\bnot\b", after.split("significant")[0])):
        regress.append(f"p = {m.group(1)} (\\{name}) is called conventionally "
                       f"significant")

print(f"  {'none found' if not regress else str(len(regress)) + ' found'}")
for r in regress:
    print(f"    {r}")
problems.extend(regress)

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

# ------------------------------------------------- generated table files
# Each table is its own file so it can sit beside the text that discusses it.
# The risk of that arrangement is a table that is generated and then never
# inputted, which vanishes from the manuscript without any error.
gen = {p.stem for p in HERE.glob("tab_*_auto.tex")}
inputs = set(re.findall(re.escape(BS + "input") + r"\{(tab_\w+_auto)\}", tex))
print(f"generated table files             : {len(gen)}")
orphaned = gen - inputs
if orphaned:
    print(f"  NOT INPUTTED: {sorted(orphaned)}")
    problems.append(f"generated but never inputted: {sorted(orphaned)}")
for name in sorted(inputs):
    if name not in gen:
        problems.append(f"input of a table file that is not generated: {name}")
    elif tex.count(BS + "input{" + name + "}") > 1:
        problems.append(f"table inputted more than once: {name}")
if not orphaned and not (inputs - gen):
    print(f"  all {len(gen)} inputted exactly once")

print()
if problems:
    print(f"FAILED: {len(problems)} problem(s)")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)
print("All static checks passed.")
