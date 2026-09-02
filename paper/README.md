# Manuscript

Draft targeting **Journal of Neural Engineering** (IOP Publishing, Q1) —
`iopart.cls`, structured abstract (*Objective / Approach / Main results /
Significance*), Harvard-numeric references via `iopart-num`.

## Files

| File | Role |
|---|---|
| `main.tex` | The manuscript. |
| `tables_auto.tex` | **Auto-generated — never edit.** All four tables plus every number quoted in the prose. |
| `make_tables.py` | Regenerates `tables_auto.tex` from `results/*.csv`. |
| `refs.bib` | Bibliography. **Read its header before submitting.** |

## The one rule

**No number is ever typed into `main.tex` by hand.** Figures in the prose are
LaTeX macros (`\PrimaryDelta`, `\BestClassicalAcc`, …) defined in
`tables_auto.tex` and generated from the result CSVs. After any new benchmark
run:

```bash
python paper/make_tables.py     # rewrites tables_auto.tex from results/
```

and the whole manuscript — tables and prose — is consistent by construction.
This matters because the study is ongoing: numbers will change, and a
hand-transcribed manuscript silently rots.

## Building

No LaTeX toolchain is installed on the development machine, so **this draft has
not been compiled**. To build:

```bash
# Windows: install MiKTeX (https://miktex.org) -- it fetches iopart
#          and iopart-num automatically on first build.
# Linux:   sudo apt install texlive-full   (or texlive + texlive-publishers)

python paper/make_tables.py
cd paper
latexmk -pdf main.tex
```

`iopart.cls` and `iopart-num.bst` ship with TeX Live (`texlive-publishers`) and
are on CTAN; MiKTeX installs them on demand. They can also be downloaded from
IOP's author pages. Expect to fix minor issues on the first compile — the draft
is written to the class conventions but has never been run through it.

Figures resolve via `\graphicspath` to `../results/figures/` and are the vector
PDFs written by `python -m qeeg.figures`.

## Before submission — checklist

- [ ] **Verify every `[CHECK]` entry in `refs.bib`.** Those are canonical works
      cited from standing knowledge; the papers are right, but volume/issue/page
      metadata was not re-checked against the publisher record. A DOI lookup per
      entry is enough. Two entries also need author lists completed.
- [ ] Fill the affiliation in `main.tex` (currently a placeholder).
- [ ] Complete the `\ack` section (funding, compute).
- [ ] Add co-authors if applicable.
- [ ] Compile once and read the PDF end to end.
- [ ] Check JNE's current author guidelines — word limits and the structured
      abstract format are enforced and do change.

## Scope note carried into the paper

The manuscript states explicitly that at 8–64 channels every kernel evaluated
is classically computable in O(n³) and that **no quantum speedup is claimed**.
Keep that. It is the sentence that separates this from the literature it
critiques, and removing it would make the paper indefensible.

## Status

Complete first draft: abstract, introduction, methods, results, discussion,
limitations, conclusion, data-availability statement, four tables, three
figures. Reports a **negative result with an identified mechanism** — the
framing is deliberate and is argued for in [../RESEARCH.md](../RESEARCH.md) §6.

Not yet included, and the obvious next additions:

- Cross-subject transfer experiment (RESEARCH.md §6, Option C) — the strongest
  candidate for a positive finding, and the setting the discussion flags.
- BCI Competition IV-2a replication for comparability with published claims.
- Shot-noise analysis.
