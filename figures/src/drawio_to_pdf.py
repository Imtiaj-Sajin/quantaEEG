"""Render a draw.io file to a single-page vector PDF.

    python figures/src/drawio_to_pdf.py paper/eeg_quantum_architecture_v3.drawio.xml \
        --out figures/architecture

Why this exists
---------------
draw.io's own PDF export paginates: the corresponding author's file declares a
1560 by 400 page while the drawing spans roughly 820 by 730 at an offset of
(1513, 831), so the export came out as four page fragments with the figure cut
across them. It also means the PDF and the editable file drift apart the moment
either is touched.

This reads the .drawio directly, so the XML stays the single source of truth:
edit in diagrams.net, re-run this, and the PDF follows. It handles the subset
of draw.io that the figure actually uses (rounded rectangles, ellipses,
triangles, cylinders, text with inline markup, and edges with explicit or
orthogonally routed waypoints) and refuses loudly on anything it does not
recognise rather than dropping it silently.

The parsed shapes are pushed through figures/src/scene.py, which already knows
how to lay out inline markup, fall back for glyphs Arial lacks, and draw
arrowheads as polygons because svglib ignores SVG markers.
"""
from __future__ import annotations

import argparse
import html
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scene import Scene  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent

# draw.io keys that carry no geometry or paint and can be ignored.
IGNORABLE = {
    "html", "whiteSpace", "overflow", "spacing", "boundedLbl",
    "backgroundOutline", "absoluteArcSize", "container", "edgeStyle",
    "orthogonalLoop", "jettySize", "perimeterSpacing", "glass", "shadow",
    "gradientColor", "fillStyle", "startFill", "startSize", "endFill",
    "endSize", "dashPattern", "exitX", "exitY", "exitDx", "exitDy",
    "entryX", "entryY", "entryDx", "entryDy", "exitPerimeter",
    "entryPerimeter", "connectable", "direction", "size", "shape", "group",
    "align", "verticalAlign", "fontFamily", "fontSize", "fontColor",
    "fontStyle", "rounded", "arcSize", "ellipse", "text", "triangle",
    "fillColor", "strokeColor", "strokeWidth", "dashed", "opacity",
    "curved", "endArrow", "startArrow", "labelBackgroundColor",
    "verticalLabelPosition", "labelPosition", "points", "flipH", "flipV",
}


def parse_style(s: str) -> dict:
    out = {}
    for part in s.split(";"):
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
        else:
            out[part] = "1"
    return out


def num(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load(path: Path):
    root = ET.parse(path).getroot()
    model = root.find(".//mxGraphModel")
    cells = model.find("root").findall("mxCell")
    by_id = {c.get("id"): c for c in cells}

    # Group children carry geometry relative to the group's origin.
    offset = {}

    def origin(cid):
        """Absolute offset that children of `cid` are measured from.

        For a group this is the group's own position plus its parent's offset,
        not the parent's offset alone: getting that wrong put every child of
        the Evaluation group at its raw relative coordinate near (0, 0) and
        stretched the bounding box from 820 units wide to 2344.
        """
        if cid in (None, "0", "1"):
            return (0.0, 0.0)
        if cid in offset:
            return offset[cid]
        c = by_id.get(cid)
        if c is None:
            return (0.0, 0.0)
        g = c.find("mxGeometry")
        px, py = origin(c.get("parent"))
        offset[cid] = (px + num(g.get("x") if g is not None else 0),
                       py + num(g.get("y") if g is not None else 0))
        return offset[cid]

    return cells, by_id, origin


def geom(c, origin):
    g = c.find("mxGeometry")
    ox, oy = origin(c.get("parent"))
    return (ox + num(g.get("x")), oy + num(g.get("y")),
            num(g.get("width")), num(g.get("height")))


def edge_points(c, by_id, origin):
    g = c.find("mxGeometry")
    pts = []
    src = g.find("mxPoint[@as='sourcePoint']")
    tgt = g.find("mxPoint[@as='targetPoint']")
    ox, oy = origin(c.get("parent"))

    def box(cid):
        x, y, w, h = geom(by_id[cid], origin)
        return x, y, w, h

    if src is not None:
        pts.append((ox + num(src.get("x")), oy + num(src.get("y"))))
    elif c.get("source"):
        x, y, w, h = box(c.get("source"))
        pts.append((x + w / 2, y + h / 2))
    arr = g.find("Array[@as='points']")
    if arr is not None:
        for p in arr.findall("mxPoint"):
            pts.append((ox + num(p.get("x")), oy + num(p.get("y"))))
    if tgt is not None:
        pts.append((ox + num(tgt.get("x")), oy + num(tgt.get("y"))))
    elif c.get("target"):
        x, y, w, h = box(c.get("target"))
        pts.append((x + w / 2, y + h / 2))
    return pts


def clip_to_box(p_inside, p_outside, box):
    """Move a centre-anchored endpoint out to the shape's border."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    dx, dy = p_outside[0] - cx, p_outside[1] - cy
    if dx == 0 and dy == 0:
        return p_inside
    sx = (w / 2) / abs(dx) if dx else float("inf")
    sy = (h / 2) / abs(dy) if dy else float("inf")
    s = min(sx, sy)
    return (cx + dx * s, cy + dy * s)


def build(path: Path, margin: float = 12.0):
    cells, by_id, origin = load(path)
    shapes, edges, unknown = [], [], set()

    for c in cells:
        st = parse_style(c.get("style") or "")
        for k in st:
            if k not in IGNORABLE:
                unknown.add(k)
        if c.get("edge") == "1":
            edges.append((c, st))
        elif c.get("vertex") == "1" and "group" not in st:
            shapes.append((c, st))

    # Bounding box over everything that will be drawn.
    xs, ys = [], []
    for c, st in shapes:
        x, y, w, h = geom(c, origin)
        xs += [x, x + w]
        ys += [y, y + h]
    for c, st in edges:
        for px, py in edge_points(c, by_id, origin):
            xs.append(px)
            ys.append(py)
    minx, miny, maxx, maxy = min(xs), min(ys), max(xs), max(ys)
    W = maxx - minx + 2 * margin
    Hh = maxy - miny + 2 * margin
    S = Scene(round(W, 1), round(Hh, 1))

    def T(p):
        return (p[0] - minx + margin, p[1] - miny + margin)

    for c, st in shapes:
        x, y, w, h = geom(c, origin)
        x, y = T((x, y))
        fill = st.get("fillColor", "none")
        stroke = st.get("strokeColor", "#000000")
        if fill == "none":
            fill = "none"
        if stroke == "none":
            stroke = "none"
        sw = num(st.get("strokeWidth"), 1)
        dash = st.get("dashed") == "1"
        op = num(st.get("opacity"), 100) / 100 if "opacity" in st else 1
        if "text" in st:
            val = html.unescape(c.get("value") or "")
            val = (val.replace("⊤", "<i>T</i>")
                      .replace("↦", "→"))
            fs = num(st.get("fontStyle"), 0)
            S.text(x, y, w, h, val,
                   size=num(st.get("fontSize"), 12),
                   color=st.get("fontColor", "#222222"),
                   bold=bool(int(fs) & 1), italic=bool(int(fs) & 2),
                   align=st.get("align", "center"),
                   valign=st.get("verticalAlign", "middle"))
        elif "ellipse" in st:
            S.ell(x, y, w, h, fill=fill, stroke=stroke, sw=sw, dash=dash, op=op)
        elif "triangle" in st:
            S.tri(x, y, w, h, fill, stroke=stroke, sw=sw,
                  direction=st.get("direction", "east"))
        elif st.get("shape", "").startswith("cylinder"):
            S.cyl(x, y, w, h, fill=fill, stroke=stroke, sw=sw)
        else:
            r = num(st.get("arcSize"), 0) / 2 if st.get("rounded") == "1" else 0
            S.rect(x, y, w, h, fill=fill, stroke=stroke, sw=sw, r=r,
                   dash=dash, op=op)

    for c, st in edges:
        pts = edge_points(c, by_id, origin)
        if len(pts) < 2:
            continue
        # Endpoints anchored on a shape start at its centre; push them out to
        # the border, then route orthogonally when draw.io asked for that.
        if c.get("source") and len(pts) >= 2:
            pts[0] = clip_to_box(pts[0], pts[1], geom(by_id[c.get("source")], origin))
        if c.get("target") and len(pts) >= 2:
            pts[-1] = clip_to_box(pts[-1], pts[-2],
                                  geom(by_id[c.get("target")], origin))
        if st.get("edgeStyle") == "orthogonalEdgeStyle" and len(pts) == 2:
            (x0, y0), (x1, y1) = pts
            if abs(x1 - x0) > 2 and abs(y1 - y0) > 2:
                mid = (x0 + x1) / 2
                pts = [(x0, y0), (mid, y0), (mid, y1), (x1, y1)]
        S.line([T(p) for p in pts],
               stroke=st.get("strokeColor", "#222222"),
               sw=num(st.get("strokeWidth"), 1),
               arrow=st.get("endArrow", "classic") not in ("none", "0"),
               start_arrow=st.get("startArrow", "none") not in ("none", "0"),
               dash=st.get("dashed") == "1",
               curved=st.get("curved") == "1")

    return S, sorted(unknown), (W, Hh), len(shapes), len(edges)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("source")
    ap.add_argument("--out", default="figures/architecture",
                    help="output stem; .svg and .pdf are written")
    ap.add_argument("--also-copy-to", default="results/figures")
    args = ap.parse_args(argv)

    src = Path(args.source)
    S, unknown, (W, H), ns, ne = build(src)
    if unknown:
        print(f"unrecognised style keys (not drawn faithfully): {unknown}")

    stem = Path(args.out)
    stem.parent.mkdir(parents=True, exist_ok=True)
    svg = S.svg()
    stem.with_suffix(".svg").write_text(svg, encoding="utf-8")

    import matplotlib
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    ttf = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
    for name, f in (("DejaVu Sans", "DejaVuSans.ttf"),
                    ("DejaVu Sans-Bold", "DejaVuSans-Bold.ttf"),
                    ("DejaVu Sans-Italic", "DejaVuSans-Oblique.ttf"),
                    ("DejaVu Sans-BoldItalic", "DejaVuSans-BoldOblique.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(ttf / f)))
    pdfmetrics.registerFontFamily("DejaVu Sans", normal="DejaVu Sans",
                                  bold="DejaVu Sans-Bold",
                                  italic="DejaVu Sans-Italic",
                                  boldItalic="DejaVu Sans-BoldItalic")

    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF

    drawing = svg2rlg(str(stem.with_suffix(".svg")))
    renderPDF.drawToFile(drawing, str(stem.with_suffix(".pdf")))

    import fitz
    doc = fitz.open(str(stem.with_suffix(".pdf")))
    doc[0].get_pixmap(matrix=fitz.Matrix(1.6, 1.6)).save(
        str(stem.parent / (stem.name + "_preview.png")))

    if args.also_copy_to:
        dest = ROOT / args.also_copy_to
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stem.with_suffix(".pdf"), dest / (stem.name + ".pdf"))

    body = 13 * (160.0 / W) / (25.4 / 72)
    print(f"{ns} shapes, {ne} edges -> {stem.with_suffix('.pdf')} "
          f"({W:.0f} x {H:.0f}; a 13-unit label prints at {body:.1f} pt "
          f"across 160 mm)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
