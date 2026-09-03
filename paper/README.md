# Manuscript

Draft targeting **Journal of Neural Engineering** (IOP Publishing, Q1), 
`iopart.cls`, structured abstract (*Objective / Approach / Main results /
Significance*), Harvard-numeric references via `iopart-num`.

## Files

| File | Role |
|---|---|
| `main.tex` | The manuscript. |
| `macros_auto.tex` | **Auto-generated: never edit.** Every number quoted in the prose, as `\newcommand` macros. Read in the preamble. |
| `tables_auto.tex` | **Auto-generated: never edit.** The ten table floats. Read in the body. |
| `cross_tables.py` | Second-dataset (IV-2a) tables and macros. |
| `reference_tables.py` | Reference-frame tables and macros: frame effect, twin control, transfer, shots. |
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

**The draft compiles cleanly: 25 pages, 0 undefined references, 0 overfull
boxes, 0 BibTeX warnings** (MiKTeX/`latexmk`, 2026-09-03).

### The `-outdir` BibTeX trap

Building with `-outdir=build` runs BibTeX *inside* `build/`, where it cannot
see `refs.bib` and silently emits an empty bibliography; the failure surfaces
much later as `Something's wrong--perhaps a missing \item` from `main.bbl`,
which points nowhere near the real cause. Set the search paths:

```bash
export PATH="$PATH:/c/Users/User/AppData/Local/Programs/MiKTeX/miktex/bin/x64"
export BIBINPUTS="<abs-path-to>/paper;"
export TEXINPUTS="<abs-path-to>/paper;"
latexmk -pdf -interaction=nonstopmode -outdir=build main.tex
```

(The trailing `;` matters on Windows: it means "then search the normal
places".) Confirm the build really succeeded by checking the *final* pass,
not the accumulated `latexmk` log, which contains first-pass warnings that
later passes resolve:

```bash
grep -E "Warning" build/main.log | grep -viE "font|miktex|update"   # expect none
grep -c bibitem build/main.bbl                                       # expect 15
```

The build uses [Tectonic](https://tectonic-typesetting.github.io/): a single
self-contained binary that fetches only the TeX packages it needs, requires no
admin rights, and needs no TeX installation. Download the Windows zip from its
[releases page](https://github.com/tectonic-typesetting/tectonic/releases) and
unpack it anywhere.

```bash
bash paper/get_iop_class.sh                  # once: fetch IOP class files
PYTHONPATH=src python -m qeeg.figures --paper            # figures 1-4
PYTHONPATH=src python -m qeeg.figures_reference --paper  # figures 5-9
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
`python -m qeeg.figures --paper` (figures 1--4) and
`python -m qeeg.figures_reference --paper` (figures 5--9: the circuit feature
maps, the reference-state schematic and invariance check, the frame effect, the
classical-twin control, and transfer plus shot budget). Paper mode drops the in-figure title and
explanatory note (the LaTeX `\caption` supplies both, having them twice is a
journal-convention error) and uses a white canvas so the figure does not sit on
the page as a tinted block. The default mode keeps titles and captions for
standalone viewing, and writes to `results/figures/`.

## Before submission: checklist

- [x] **Verify every `[CHECK]` entry in `refs.bib`.** Done 2026-09-03: all ten
      were resolved to a DOI and confirmed field by field against the CrossRef
      record. No entry carries the flag any more; add it back for any reference
      cited from memory in future.
- [x] **Compile at least once.** Done 2026-09-03, the first successful build.
      It caught `\tfrac`, undefined under `iopams`, which the static checker
      cannot see.
- [ ] **Fill the affiliation in `main.tex`** (still a placeholder).
- [ ] **Complete the `\ack` section** (funding, compute).
- [ ] Add co-authors if applicable.
- [ ] **Read the PDF end to end.** It compiles and every number is generated
      from a CSV, but nobody has yet read it as a reader would.
- [ ] Check JNE's current author guidelines: word limits and the structured
      abstract format are enforced, and do change.

## Scope note carried into the paper

The manuscript states explicitly that at 8–64 channels every kernel evaluated
is classically computable in O(n³) and that **no quantum speedup is claimed**.
Keep that. It is the sentence that separates this from the literature it
critiques, and removing it would make the paper indefensible.

## Status

Complete draft, **compiling to 25 pages**: abstract, introduction, methods,
results, discussion, limitations, conclusion, data-availability statement,
ten tables, nine figures, 15 references.

The argument is a **negative result with an identified mechanism**, and the
mechanism changed on 2026-09-03. It is no longer kernel concentration but an
**invariance mismatch**: the quantum quantities are invariant under unitary
conjugation while EEG's nuisances act by congruence, so the original benchmark
was in part measuring a difference of invariance groups. Correcting it reverses
the headline comparison; the classical-twin control then shows the gain is not
quantum. See [../RESEARCH.md](../RESEARCH.md) §4.6–§4.10.

Every experiment the argument needs is done: PhysioNet at three and five
qubits, IV-2a, cross-subject transfer, filter-bank/FBCSP baselines, and a
shot-noise analysis. The remaining work is editorial rather than
computational — the three unticked checklist items above.
