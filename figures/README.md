# figures/

The architecture figure, its editable source, and the alternatives.

## What the paper uses

| File | What it is |
|---|---|
| **`architecture.drawio`** | **the source.** Open this in diagrams.net to change the figure |
| **`architecture.pdf`** | the vector the manuscript includes, exported from diagrams.net |
| `architecture_preview.png` | raster, for a quick look without opening a PDF |
| `architecture_caption.md` | the caption, and every number kept out of the drawing |

`paper/main.tex` includes `architecture.pdf`, found through its graphicspath via
`results/figures/`. The same PDF is copied there, and the `.drawio` travels
inside `paper/build/submission.zip` as `architecture_drawio.xml` so anyone who
receives the submission can edit the figure.

> **After editing the figure, re-export the PDF.** The `.drawio` and the `.pdf`
> are two separate files and nothing in the build checks that they agree, so an
> edit that is not re-exported leaves the old figure in the paper.

## Other versions, kept on purpose

`versions/` holds layouts that are not currently used but are worth being able
to go back to:

| File | What it is |
|---|---|
| `versions/architecture_tworow.drawio` | a line-art alternative: five lettered panels **a** to **e** in two rows, mostly black and grey, colour only where it carries meaning |
| `versions/architecture_tworow.pdf` | the same, as vector |
| `versions/architecture_tworow.svg` | the same, as SVG |
| `versions/architecture_tworow_preview.png` | a raster preview, open this first |

To switch the paper to that version, copy its PDF over the one the manuscript
reads and rebuild:

```bash
cp figures/versions/architecture_tworow.pdf results/figures/architecture.pdf
python run.py paper
```

The caption in `main.tex` would then need its panel names changing back to
letters, since that version labels panels **a** to **e** rather than titling
them.

## Generators

| File | What it does |
|---|---|
| `src/scene.py` | drawing primitives and the SVG exporter that both paths share |
| `src/make_architecture.py` | builds the two-row alternative in `versions/` |
| `src/drawio_to_pdf.py` | fallback: renders any `.drawio` to a single-page PDF |

The fallback exists because an earlier export declared a 1560 by 400 page while
the drawing spanned about 840 by 750 at an offset, so diagrams.net produced four
page fragments with the figure cut across them. The current export is a correct
single page and is used as it comes. Run the fallback only if that recurs:

```bash
python figures/src/drawio_to_pdf.py figures/architecture.drawio \
    --out figures/architecture
```

It substitutes a superscript italic *T* for U+22A4 and a plain arrow for
U+21A6, because Arial carries neither; diagrams.net renders both correctly
through Liberation Sans, which is a reason to prefer its own export.
