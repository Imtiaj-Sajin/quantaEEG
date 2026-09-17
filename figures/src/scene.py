"""A tiny scene graph that exports the same drawing to draw.io and to SVG.

One source, two outputs, so the editable file and the published vector can
never drift apart.

Portability notes for this machine
----------------------------------
The reference implementation targeted Linux. Two changes were needed:

* Fonts. Liberation Sans is metric-compatible with Arial and is what the SVG
  asks for; on Windows we measure with Arial itself, which has the same
  metrics, and fall back to the DejaVu Sans that ships inside matplotlib for
  the handful of glyphs Arial lacks.
* f-strings. The original emitted text with a backslash escape inside an
  f-string expression, which is a syntax error before Python 3.12. The
  substitution is done before the f-string here.
"""
import html
import re
from pathlib import Path

FONT_SVG = "Helvetica"          # Arial metrics; renderers substitute sensibly
FONT_DIO = "Helvetica"

_WIN = Path("C:/Windows/Fonts")


def _font_path(bold: bool, fallback: bool) -> str:
    """Arial for the body, DejaVu for glyphs Arial does not carry."""
    if fallback:
        import matplotlib
        d = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
        return str(d / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"))
    return str(_WIN / ("arialbd.ttf" if bold else "arial.ttf"))


class Scene:
    def __init__(s, W, H):
        s.W, s.H, s.els, s.n, s.markers = W, H, [], 2, {}

    def _id(s):
        s.n += 1
        return f"c{s.n}"

    # ---------------------------------------------------------- primitives
    def rect(s, x, y, w, h, fill="none", stroke="none", sw=1, r=0, dash=False,
             op=1, shadow=False):
        s.els.append(dict(k="rect", x=x, y=y, w=w, h=h, fill=fill, stroke=stroke,
                          sw=sw, r=r, dash=dash, op=op, shadow=shadow))

    def ell(s, x, y, w, h, fill="none", stroke="none", sw=1, dash=False, op=1):
        s.els.append(dict(k="ell", x=x, y=y, w=w, h=h, fill=fill, stroke=stroke,
                          sw=sw, dash=dash, op=op))

    def circ(s, cx, cy, r, **kw):
        s.ell(cx - r, cy - r, 2 * r, 2 * r, **kw)

    def tri(s, x, y, w, h, fill, stroke="none", sw=1, direction="north"):
        s.els.append(dict(k="tri", x=x, y=y, w=w, h=h, fill=fill, stroke=stroke,
                          sw=sw, d=direction))

    def cyl(s, x, y, w, h, fill, stroke, sw=1.2):
        s.els.append(dict(k="cyl", x=x, y=y, w=w, h=h, fill=fill, stroke=stroke,
                          sw=sw))

    def text(s, x, y, w, h, t, size=12, color="#222222", bold=False,
             italic=False, align="center", valign="middle"):
        s.els.append(dict(k="text", x=x, y=y, w=w, h=h, t=t, size=size,
                          color=color, bold=bold, italic=italic, align=align,
                          valign=valign))

    def line(s, pts, stroke="#222222", sw=1.5, arrow=False, dash=False,
             curved=False, start_arrow=False):
        s.els.append(dict(k="line", pts=pts, stroke=stroke, sw=sw, arrow=arrow,
                          dash=dash, curved=curved, sa=start_arrow))

    # ------------------------------------------------------------- draw.io
    def drawio(s):
        out = []
        for e in s.els:
            i = s._id()
            k = e["k"]
            if k == "line":
                st = (f"html=1;rounded=0;strokeColor={e['stroke']};"
                      f"strokeWidth={e['sw']};")
                st += ("endArrow=block;endFill=1;endSize=%d;" % (4 + 2 * e["sw"])
                       if e["arrow"] else "endArrow=none;")
                st += "startArrow=block;startFill=1;" if e["sa"] else "startArrow=none;"
                if e["dash"]:
                    st += "dashed=1;"
                if e["curved"]:
                    st += "curved=1;"
                p = e["pts"]
                wp = "".join(f'<mxPoint x="{a:.1f}" y="{b:.1f}"/>' for a, b in p[1:-1])
                arr = f'<Array as="points">{wp}</Array>' if wp else ""
                out.append(
                    f'<mxCell id="{i}" style="{st}" edge="1" parent="1">'
                    f'<mxGeometry relative="1" as="geometry">'
                    f'<mxPoint x="{p[0][0]:.1f}" y="{p[0][1]:.1f}" as="sourcePoint"/>'
                    f'<mxPoint x="{p[-1][0]:.1f}" y="{p[-1][1]:.1f}" as="targetPoint"/>'
                    f'{arr}</mxGeometry></mxCell>')
                continue
            val = ""
            if k == "rect":
                st = f"rounded={1 if e['r'] else 0};absoluteArcSize=1;arcSize={e['r'] * 2};"
            elif k == "ell":
                st = "ellipse;"
            elif k == "tri":
                st = f"triangle;direction={e['d']};"
            elif k == "cyl":
                st = ("shape=cylinder3;boundedLbl=1;backgroundOutline=1;size=%d;"
                      % max(3, e["h"] // 6))
            if k == "text":
                st = (f"text;html=1;whiteSpace=wrap;overflow=visible;"
                      f"align={e['align']};verticalAlign={e['valign']};"
                      f"fontFamily={FONT_DIO};fontSize={e['size']};"
                      f"fontColor={e['color']};"
                      f"fontStyle={(1 if e['bold'] else 0) + (2 if e['italic'] else 0)};"
                      f"spacing=0;")
                val = html.escape(e["t"], quote=True)
            else:
                f = e["fill"]
                st += f"fillColor={f};" if f != "none" else "fillColor=none;"
                st += (f"strokeColor={e['stroke']};strokeWidth={e['sw']};"
                       if e["stroke"] != "none" else "strokeColor=none;")
                if e.get("dash"):
                    st += "dashed=1;dashPattern=6 4;"
                if e.get("op", 1) != 1:
                    st += f"opacity={int(e['op'] * 100)};"
                if e.get("shadow"):
                    st += "shadow=1;"
                st += "html=1;"
            out.append(
                f'<mxCell id="{i}" value="{val}" style="{st}" vertex="1" parent="1">'
                f'<mxGeometry x="{e["x"]:.1f}" y="{e["y"]:.1f}" '
                f'width="{e["w"]:.1f}" height="{e["h"]:.1f}" as="geometry"/></mxCell>')
        s.n = 2
        body = "".join(out)
        return (f'<mxfile host="app.diagrams.net"><diagram id="arch" name="Architecture">'
                f'<mxGraphModel dx="{s.W}" dy="{s.H}" grid="0" gridSize="10" guides="1" '
                f'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
                f'pageWidth="{s.W}" pageHeight="{s.H}" background="#FFFFFF" math="0" '
                f'shadow="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
                f'{body}</root></mxGraphModel></diagram></mxfile>')

    # ----------------------------------------------------------------- SVG
    def _marker(s, color):
        mid = "m" + color.strip("#")
        s.markers[mid] = color
        return mid

    def _smooth(s, p):
        d = f"M{p[0][0]:.1f},{p[0][1]:.1f}"
        for i in range(len(p) - 1):
            p0 = p[i - 1] if i else p[i]
            p1, p2 = p[i], p[i + 1]
            p3 = p[i + 2] if i + 2 < len(p) else p2
            c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
            c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
            d += (f" C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} "
                  f"{p2[0]:.1f},{p2[1]:.1f}")
        return d

    # Glyphs Arial does not carry, or carries badly.
    FB = set("\u22a4\u21a6\u2208\u27e8\u27e9\u208a\u208b\u0303\u2248\u21c4\u2713\u2717\u221d\u00f7\u03c1\u03c3")

    def _runs(s, t):
        """Split a string into runs that share one font."""
        out, cur, fb = [], "", None
        for k, ch in enumerate(t):
            f = ch in s.FB or (k + 1 < len(t) and t[k + 1] == "\u0303")
            if fb is None or f == fb:
                cur += ch
            else:
                out.append((cur, fb))
                cur = ch
            fb = f
        out.append((cur, fb))
        return out

    def _w(s, t, size, bold):
        return sum(s._w1(r, size, bold, fb) for r, fb in s._runs(t))

    def _w1(s, t, size, bold, fb):
        from PIL import ImageFont
        key = (size, bold, fb)
        if not hasattr(s, "_fc"):
            s._fc = {}
        if key not in s._fc:
            s._fc[key] = ImageFont.truetype(_font_path(bold, fb),
                                            size=int(round(size * 10)))
        return s._fc[key].getlength(t.replace("\u0303", "")) / 10

    def _svgtext(s, e):
        lines = e["t"].split("<br>")
        size = e["size"]
        lh = size * 1.28
        total = lh * len(lines)
        if e["valign"] == "middle":
            y0 = e["y"] + (e["h"] - total) / 2 + size * 0.98
        elif e["valign"] == "top":
            y0 = e["y"] + size * 0.98
        else:
            y0 = e["y"] + e["h"] - total + size * 0.98
        res = []
        for li, ln in enumerate(lines):
            parts = re.split(r"(</?(?:b|i|sup|sub)>)", ln)
            bold, ital, shift = e["bold"], e["italic"], 0
            spans = []
            for p in parts:
                if p in ("<b>", "</b>"):
                    bold = p == "<b>"
                    continue
                if p in ("<i>", "</i>"):
                    ital = p == "<i>"
                    continue
                if p in ("<sup>", "<sub>"):
                    shift = -1 if p == "<sup>" else 1
                    continue
                if p in ("</sup>", "</sub>"):
                    shift = 0
                    continue
                if not p:
                    continue
                spans.append([p, size * (0.68 if shift else 1), bold, ital, shift])
            xs, cur, prev = [], 0.0, None
            for sp in spans:
                # A subscript straight after a superscript stacks under it.
                if sp[4] == 1 and prev is not None and prev[4] == -1:
                    x_here = prev[5]
                    cur = max(cur, x_here + s._w(sp[0], sp[1], sp[2]))
                    sp.append(x_here)
                    prev = sp
                    continue
                sp.append(cur)
                cur += s._w(sp[0], sp[1], sp[2])
                prev = sp
            width = cur
            X0 = {"center": e["x"] + (e["w"] - width) / 2,
                  "left": e["x"],
                  "right": e["x"] + e["w"] - width}[e["align"]]
            yb = y0 + li * lh
            for t, fs, b, it, sh, xo in spans:
                dy = -size * 0.38 if sh == -1 else (size * 0.25 if sh == 1 else 0)
                for r, fb in s._runs(t):
                    # Escaped and space-substituted before the f-string: a
                    # backslash inside an f-string expression is a syntax
                    # error before Python 3.12.
                    body = html.escape(r).replace(" ", "\u00a0")
                    fam = "DejaVu Sans" if fb else FONT_SVG
                    res.append(
                        f'<text x="{X0 + xo:.1f}" y="{yb + dy:.1f}" fill="{e["color"]}" '
                        f'font-family="{fam}" font-size="{fs:.1f}" '
                        f'font-weight="{"bold" if b else "normal"}" '
                        f'font-style="{"italic" if it else "normal"}" '
                        f'xml:space="preserve">{body}</text>')
                    xo += s._w1(r, fs, b, fb)
        return "".join(res)

    @staticmethod
    def _head(a, b, color, sw):
        """A filled triangle at b, pointing away from a."""
        import math as _m
        ang = _m.atan2(b[1] - a[1], b[0] - a[0])
        ln = 4.0 + 2.4 * sw
        wd = 1.6 + 1.1 * sw
        tip = b
        back = (b[0] - ln * _m.cos(ang), b[1] - ln * _m.sin(ang))
        left = (back[0] - wd * _m.sin(ang), back[1] + wd * _m.cos(ang))
        right = (back[0] + wd * _m.sin(ang), back[1] - wd * _m.cos(ang))
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (tip, left, right))
        return f'<polygon points="{pts}" fill="{color}" stroke="none"/>'

    def svg(s):
        o = []
        for e in s.els:
            k = e["k"]
            dash = ' stroke-dasharray="6 4"' if e.get("dash") else ""
            common = ""
            if k in ("rect", "ell", "tri", "cyl"):
                common = (f'fill="{e["fill"]}" stroke="{e["stroke"]}" '
                          f'stroke-width="{e["sw"]}"{dash} opacity="{e.get("op", 1)}"')
            if k == "rect":
                if e.get("shadow"):
                    o.append(f'<rect x="{e["x"] + 3}" y="{e["y"] + 3}" '
                             f'width="{e["w"]}" height="{e["h"]}" rx="{e["r"]}" '
                             f'fill="#000" opacity="0.08"/>')
                o.append(f'<rect x="{e["x"]}" y="{e["y"]}" width="{e["w"]}" '
                         f'height="{e["h"]}" rx="{e["r"]}" {common}/>')
            elif k == "ell":
                o.append(f'<ellipse cx="{e["x"] + e["w"] / 2}" cy="{e["y"] + e["h"] / 2}" '
                         f'rx="{e["w"] / 2}" ry="{e["h"] / 2}" {common}/>')
            elif k == "tri":
                x, y, w, h = e["x"], e["y"], e["w"], e["h"]
                pts = {"north": [(x, y + h), (x + w / 2, y), (x + w, y + h)],
                       "south": [(x, y), (x + w, y), (x + w / 2, y + h)],
                       "east": [(x, y), (x + w, y + h / 2), (x, y + h)],
                       "west": [(x + w, y), (x, y + h / 2), (x + w, y + h)]}[e["d"]]
                pstr = " ".join(f"{a},{b}" for a, b in pts)
                o.append(f'<polygon points="{pstr}" {common}/>')
            elif k == "cyl":
                x, y, w, h = e["x"], e["y"], e["w"], e["h"]
                ry = max(3, h // 6) / 2 + 1
                o.append(f'<path d="M{x},{y + ry} L{x},{y + h - ry} '
                         f'A{w / 2},{ry} 0 0 0 {x + w},{y + h - ry} '
                         f'L{x + w},{y + ry}" {common}/>')
                o.append(f'<ellipse cx="{x + w / 2}" cy="{y + ry}" rx="{w / 2}" '
                         f'ry="{ry}" {common}/>')
            elif k == "text":
                o.append(s._svgtext(e))
            elif k == "line":
                p = e["pts"]
                d = (s._smooth(p) if e["curved"]
                     else "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in p))
                o.append(f'<path d="{d}" fill="none" stroke="{e["stroke"]}" '
                         f'stroke-width="{e["sw"]}" stroke-linejoin="round" '
                         f'stroke-linecap="round"{dash}/>')
                # Arrowheads are drawn as polygons rather than declared as SVG
                # <marker>s. svglib, which is what converts this to PDF here,
                # ignores markers entirely, so every arrow in the figure came
                # out as a plain line.
                if e["arrow"]:
                    o.append(s._head(p[-2], p[-1], e["stroke"], e["sw"]))
                if e["sa"]:
                    o.append(s._head(p[1], p[0], e["stroke"], e["sw"]))
        defs = "".join(
            f'<marker id="{m}" viewBox="0 0 10 10" refX="8" refY="5" '
            f'markerWidth="5" markerHeight="5" orient="auto">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
            f'<marker id="s{m}" viewBox="0 0 10 10" refX="2" refY="5" '
            f'markerWidth="5" markerHeight="5" orient="auto">'
            f'<path d="M10,0 L0,5 L10,10 z" fill="{c}"/></marker>'
            for m, c in s.markers.items())
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{s.W}" '
                f'height="{s.H}" viewBox="0 0 {s.W} {s.H}"><defs>{defs}</defs>'
                f'<rect width="100%" height="100%" fill="#fff"/>{"".join(o)}</svg>')
