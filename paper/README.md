# Manuscript

Draft targeting **Journal of Neural Engineering** (IOP Publishing, Q1), 
`iopart.cls`, structured abstract (*Objective / Approach / Main results /
Significance*), Harvard-numeric references via `iopart-num`.

## Files

| File | Role |
|---|---|
| `main.tex` | The manuscript. |
| `macros_auto.tex` | **Auto-generated: never edit.** Every number quoted in the prose, as `\newcommand` macros. Read in the preamble. |
| `tables_auto.tex` | **Auto-generated: never edit.** The four table floats. Read in the body. |
| `make_tables.py` | Regenerates both of the above from `results/*.csv`. |
| `check_tex.py` | Static checks with no LaTeX needed: undefined macros, unresolved citations, dangling refs, unbalanced environments, missing figures. |
| `get_iop_class.sh` | Fetches the IOP class files (not on CTAN, not committed). |
| `refs.bib` | Bibliography. **Read its header before submitting.** |
| `build/main.pdf` | Compiled output. |

## The one rule

**No number is ever typed into `main.tex` by hand.** Figures in the prose are
LaTeX macros (`\PrimaryDelta`, `\BestClassicalAcc`, …) defined in
`macros_auto.tex` and generated from the result CSVs. After any new benchmark
run:

```bash
python paper/make_tables.py     # rewrites macros_auto.tex + tables_auto.tex
python paper/check_tex.py       # confirm nothing broke
```

and the whole manuscript, tables and prose, is consistent by construction.
This matters because the study is ongoing: numbers will change, and a
hand-transcribed manuscript silently rots.

## Building

**The draft compiles cleanly: 12 pages, 0 undefined references, 0 overfull
boxes, 0 BibTeX warnings.**

The build uses [Tectonic](https://tectonic-typesetting.github.io/): a single
self-contained binary that fetches only the TeX packages it needs, requires no
admin rights, and needs no TeX installation. Download the Windows zip from its
[releases page](https://github.com/tectonic-typesetting/tectonic/releases) and
unpack it anywhere.

```bash
bash paper/get_iop_class.sh                  # once: fetch IOP class files
PYTHONPATH=src python -m qeeg.figures --paper   # paper-mode figures
python paper/make_tables.py                  # tables + inline macros
cd paper && tectonic -X compile main.tex --outdir build
```

Output: `paper/build/main.pdf`.

MiKTeX or TeX Live work too (`latexmk -pdf main.tex`), but note that
**`iopart.cls` is not on CTAN**, IOP distributes it separately, so no TeX
distribution installs it automatically. `get_iop_class.sh` fetches it plus
`iopart10/12.clo`, `iopams.sty`, `setstack.sty`, `harvard.sty` and
`iopart-num.bst`. These are third-party files and are **not committed** to the
repo; for a real submission take the official copy from
[IOP's author pages](https://publishingsupport.iopscience.iop.org/questions/latex-template/).

### Two class quirks worth knowing

- **Do not `\usepackage{amsmath}`.** It clashes with `iopart.cls`'s own
  `\equation*`. Load `iopams` instead, which pulls in amsmath/amssymb in the
  order the class expects.
- **No `align` environment.** `iopart` provides `eqnarray` for multi-line
  displays; `align` is undefined and will halt the build.

### Figures

`\graphicspath` prefers `../results/figures_paper/`, produced by
`python -m qeeg.figures --paper`. Paper mode drops the in-figure title and
explanatory note (the LaTeX `\caption` supplies both, having them twice is a
journal-convention error) and uses a white canvas so the figure does not sit on
the page as a tinted block. The default mode keeps titles and captions for
standalone viewing, and writes to `results/figures/`.

## Before submission: checklist

- [ ] **Verify every `[CHECK]` entry in `refs.bib`.** Those are canonical works
      cited from standing knowledge; the papers are right, but volume/issue/page
      metadata was not re-checked against the publisher record. A DOI lookup per
      entry is enough. (The two entries that were missing author lists have now
      been verified and completed.)
- [ ] Fill the affiliation in `main.tex` (currently a placeholder).
- [ ] Complete the `\ack` section (funding, compute).
- [ ] Add co-authors if applicable.
- [ ] Compile once and read the PDF end to end.
- [ ] Check JNE's current author guidelines, word limits and the structured
      abstract format are enforced and do change.

## Scope note carried into the paper

The manuscript states explicitly that at 8–64 channels every kernel evaluated
is classically computable in O(n³) and that **no quantum speedup is claimed**.
Keep that. It is the sentence that separates this from the literature it
critiques, and removing it would make the paper indefensible.

## Status

Complete first draft, **compiling to 12 pages**: abstract, introduction,
methods, results, discussion, limitations, conclusion, data-availability
statement, four tables, three figures, 14 references. Reports a **negative
result with an identified mechanism**, the framing is deliberate and is argued
for in [../RESEARCH.md](../RESEARCH.md) §6.

Not yet included, and the obvious next additions:

- Cross-subject transfer experiment (RESEARCH.md §6, Option C), the strongest
  candidate for a positive finding, and the setting the discussion flags.
- BCI Competition IV-2a replication for comparability with published claims.
- Shot-noise analysis.
