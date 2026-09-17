"""Build the architecture figure: one scene graph, three outputs.

    python figures/src/make_architecture.py

Writes figures/architecture.drawio (editable, native shapes),
figures/architecture.svg, figures/architecture.pdf (vector) and
figures/architecture_preview.png, and copies the PDF to results/figures/ so the
manuscript's graphicspath picks it up.

Why two rows rather than one
----------------------------
A single row of five panels is the better composition on a screen, and it is
illegible on the page. This manuscript is single-column IOP at 160 mm, so a
1560-unit canvas shrinks to about 41 mm tall and a 10-unit label renders at
2.9 pt, well under the 6 pt most journals require. Folding the same five panels
into two rows on an 880-unit canvas puts body labels near 6.4 pt and panel
titles near 8.7 pt, which survives print.

Two portability notes
---------------------
cairosvg binds to a native libcairo that Windows does not ship, so the PDF is
produced with svglib and reportlab and the preview is rasterised from that PDF,
which also guarantees the preview shows exactly what the vector shows. And
svglib ignores SVG markers, so arrowheads are drawn as polygons in scene.py.
"""
import math
import os
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene import Scene  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "figures" / "versions"
OUT.mkdir(parents=True, exist_ok=True)

W, H = 880, 752
S = Scene(W, H)

INK, GRAY, LG, FILL = "#222222", "#707070", "#cfcfcf", "#f3f3f3"
Q, QL = "#2f5d9e", "#e5edf7"   # quantum, reference frame, chosen electrodes
T = "#a4503a"                   # metric-matched twin
K = "#555555"                   # classical, sensor frame
P = "#6a5a9c"                   # circuit kernels

# Type scale, chosen so the smallest label clears 6 pt at 160 mm.
F_LET, F_PAN, F_GRP, F_BODY, F_SMALL, F_MATH, F_BIG = 20, 17, 14, 13, 11.5, 17, 27


def txt(x, y, w, h, t, size=F_BODY, color=INK, **kw):
    S.text(x, y, w, h, t, size, color, **kw)


def L(pts, c=INK, sw=1, **kw):
    S.line(pts, stroke=c, sw=sw, **kw)


def panel(x, y, w, letter, title):
    txt(x, y, 22, 26, letter, F_LET, INK, bold=True, align="left")
    txt(x + 26, y + 1, w - 26, 26, title, F_PAN, INK, bold=True, align="left")
    L([(x, y + 30), (x + w, y + 30)], LG, 1)


def bloch(cx, cy, r, vecs, c=Q):
    S.circ(cx, cy, r, fill="#ffffff", stroke=c, sw=1.2)
    S.ell(cx - r, cy - r * 0.3, 2 * r, r * 0.6, stroke=c, sw=0.9, dash=True)
    L([(cx, cy - r), (cx, cy + r)], LG, 0.9)
    for i, a in enumerate(vecs):
        a = math.radians(a)
        L([(cx, cy), (cx + 0.86 * r * math.cos(a), cy + 0.86 * r * math.sin(a))],
          [c, T][i % 2], 1.9, arrow=True)
    S.circ(cx, cy, 2.2, fill=c)


def ellipse_pts(cx, cy, a, b, ang, n=40):
    out = []
    for k in range(n + 1):
        t = 2 * math.pi * k / n
        ex, ey = a * math.cos(t), b * math.sin(t)
        r = math.radians(ang)
        out.append((cx + ex * math.cos(r) - ey * math.sin(r),
                    cy + ex * math.sin(r) + ey * math.cos(r)))
    return out


# ############################ ROW 1 #######################################
R1 = 12

# ================================================================== a  EEG
ax, aw = 16, 264
panel(ax, R1, aw, "a", "EEG")

hx, hy, hr = ax + 74, 128, 52
S.tri(hx - 8, hy - hr - 11, 16, 13, fill="#ffffff", stroke=INK, sw=1.1)
S.ell(hx - hr - 5, hy - 11, 10, 22, fill="#ffffff", stroke=INK, sw=1.1)
S.ell(hx + hr - 5, hy - 11, 10, 22, fill="#ffffff", stroke=INK, sw=1.1)
S.circ(hx, hy, hr, fill="#ffffff", stroke=INK, sw=1.4)
sel = {(-0.32, -0.37), (-0.32, 0.0), (-0.32, 0.37), (0.0, -0.45), (0.0, 0.0),
       (0.0, 0.45), (0.32, -0.37), (0.32, 0.37)}
grid = [(-0.66, [-0.3, 0, 0.3]),
        (-0.32, [-0.72, -0.37, 0, 0.37, 0.72]),
        (0.0, [-0.82, -0.45, 0, 0.45, 0.82]),
        (0.32, [-0.72, -0.37, 0, 0.37, 0.72]),
        (0.66, [-0.3, 0, 0.3])]
for ry, xs in grid:
    for rx in xs:
        px, py = hx + rx * hr, hy + ry * hr
        if (ry, rx) in sel:
            S.circ(px, py, 5.0, fill=Q)
        else:
            S.circ(px, py, 3.0, fill="#ffffff", stroke="#9a9a9a", sw=0.9)

random.seed(2)
tx0, tw = ax + 150, 104
for i in range(8):
    y = 86 + i * 14
    ph, pts = random.random() * 6, []
    for k in range(41):
        t = k / 40
        v = (0.55 * math.sin(2 * math.pi * 3.2 * t + ph)
             + 0.30 * math.sin(2 * math.pi * 7 * t + 2 * ph)
             + 0.15 * math.sin(2 * math.pi * 13 * t))
        pts.append((tx0 + tw * t, y + 5.2 * v))
    L(pts, Q if i % 2 == 0 else INK, 0.9, curved=True)
L([(tx0, 206), (tx0 + tw, 206)], INK, 0.9)
L([(tx0, 202), (tx0, 210)], INK, 0.9)
L([(tx0 + tw, 202), (tx0 + tw, 210)], INK, 0.9)
txt(tx0, 208, tw, 16, "1 s", F_SMALL, GRAY)

fx, fy = ax + 8, 232
L([(fx, fy + 34), (fx + 104, fy + 34)], GRAY, 0.9)
L([(fx + 2, fy + 33), (fx + 22, fy + 32), (fx + 32, fy + 6), (fx + 70, fy + 6),
   (fx + 80, fy + 32), (fx + 102, fy + 33)], Q, 1.6, curved=True)
txt(fx + 112, fy + 10, 140, 20, "8–30 Hz · 8 ch", F_BODY, INK, align="left")

for i, name in enumerate(["EEGMMIDB", "BCI IV 2a", "Cho2017"]):
    cxx = ax + 12 + i * 86
    S.cyl(cxx, 302, 19, 26, fill="#ffffff", stroke=INK, sw=1.1)
    txt(cxx - 14, 330, 62, 16, name, F_SMALL, INK)

# ================================================================ b  STATE
bx, bw = 296, 244
panel(bx, R1, bw, "b", "State")


def blue(v):
    a, b = (255, 255, 255), (32, 72, 138)
    return "#%02x%02x%02x" % tuple(int(a[k] + v * (b[k] - a[k])) for k in range(3))


cs = 16
hm_x, hm_y = bx + 6, 58
for i in range(8):
    for j in range(8):
        v = math.exp(-abs(i - j) / 2.4) * (0.8 + 0.2 * math.cos(i + j))
        v = min(1, max(0.03, v + 0.06 * math.sin(3.1 * (i + j))))
        S.rect(hm_x + j * cs, hm_y + i * cs, cs, cs, fill=blue(v),
               stroke="#ffffff", sw=0.6)
S.rect(hm_x, hm_y, 8 * cs, 8 * cs, stroke=INK, sw=0.9)
txt(hm_x + 8 * cs + 12, hm_y + 42, 70, 32, "<i>C</i>", F_BIG, INK, align="left")
txt(hm_x + 8 * cs + 12, hm_y + 76, 70, 18, "8 × 8", F_SMALL, GRAY, align="left")

L([(hm_x + 64, 192), (hm_x + 64, 222)], INK, 1.3, arrow=True)
txt(hm_x + 74, 196, 90, 22, "÷ tr <i>C</i>", F_BODY, GRAY, align="left")

ry0 = 250
for q in range(3):
    y = ry0 + q * 24
    L([(bx + 8, y), (bx + 48, y)], INK, 1.1)
    L([(bx + 104, y), (bx + 140, y)], INK, 1.1)
S.rect(bx + 48, ry0 - 16, 56, 80, fill=QL, stroke=Q, sw=1.4, r=3)
txt(bx + 48, ry0 - 16, 56, 80, "<i>ρ</i>", F_BIG, Q)
bloch(bx + 186, ry0 + 24, 32, [-55])
txt(bx + 146, ry0 + 62, 80, 18, "3 qubits", F_SMALL, Q)

# ================================================================ c  FRAME
cx0, cwid = 556, 308
panel(cx0, R1, cwid, "c", "Frame")
txt(cx0, 46, cwid, 26, "<i>C</i> → <i>A C A</i><sup><i>T</i></sup>",
    F_MATH, INK)
bwid = 146
for i, (title, col) in enumerate([("sensor", K), ("reference", Q)]):
    px = cx0 + i * (bwid + 16)
    S.rect(px, 80, bwid, 262, fill="#ffffff", stroke=LG, sw=1.1, r=3)
    S.rect(px, 80, bwid, 3.5, fill=col)
    txt(px, 90, bwid, 22, title, F_GRP, col, bold=True)
    mx, my = px + bwid / 2, 166
    if i == 0:
        L(ellipse_pts(mx, my, 44, 18, -20), col, 1.4, curved=True)
        L(ellipse_pts(mx, my, 44, 18, -58), "#a6a6a6", 1.3, curved=True, dash=True)
        L([(mx + 38, my - 36), (mx + 50, my - 22), (mx + 47, my - 6)], GRAY, 1.2,
          curved=True, arrow=True)
        txt(px, 218, bwid, 24, "<i>ρ</i> = <i>C</i> / tr <i>C</i>", F_BODY, INK)
        txt(px, 252, bwid, 24, "<i>O C O</i><sup><i>T</i></sup> only", F_BODY, K)
        S.circ(mx, 308, 11, stroke=K, sw=1.5)
        L([(mx - 7, 301), (mx + 7, 315)], K, 1.5)
    else:
        L(ellipse_pts(mx - 36, my, 30, 13, -25), GRAY, 1.3, curved=True)
        L([(mx - 2, my), (mx + 16, my)], INK, 1.2, arrow=True)
        txt(mx - 10, my - 24, 26, 16, "<i>W</i>", F_BODY, INK)
        S.circ(mx + 40, my, 21, stroke=col, sw=1.5)
        txt(px, 218, bwid, 24, "<i>W</i> = <i>M</i><sup>−1/2</sup>", F_BODY, Q)
        txt(px, 252, bwid, 24, "<i>A C A</i><sup><i>T</i></sup>", F_BODY, Q)
        S.circ(mx, 308, 11, fill=Q)
        L([(mx - 5, 308), (mx - 1, 312), (mx + 6, 303)], "#ffffff", 2.0)

L([(ax + aw + 2, 150), (bx - 6, 150)], INK, 1.5, arrow=True)
L([(bx + bw + 2, 150), (cx0 - 6, 150)], INK, 1.5, arrow=True)

# elbow carrying the flow from the end of row 1 into the start of row 2
L([(cx0 + cwid / 2, 348), (cx0 + cwid / 2, 366), (ax + 140, 366), (ax + 140, 386)],
  INK, 1.5, arrow=True)

# ############################ ROW 2 #######################################
R2 = 392

# ============================================================== d  KERNELS
dx, dwid = 16, 548
panel(dx, R2, dwid, "d", "Kernels")
cw, gp = 128, 12
gx = [dx + 4 + i * (cw + gp) for i in range(4)]
groups = [("Classical", K, ["CSP + LDA", "TS + LR", "MDM", "FBCSP"]),
          ("Twin", T, ["Riemann", "log-Eucl."]),
          ("Quantum", Q, ["HS", "Fidelity", "Bures", "QRE"]),
          ("Circuit", P, ["IQP", "IQP − ent.", "PCA"])]
IY, IH = R2 + 62, 104
for i, (title, c, items) in enumerate(groups):
    X = gx[i]
    S.rect(X, R2 + 38, cw, 3.5, fill=c)
    txt(X, R2 + 44, cw, 20, title, F_GRP, c, bold=True)
    S.rect(X, IY, cw, IH, fill=FILL, r=3)
    if i == 0:                                   # two classes, one boundary
        random.seed(5)
        for k in range(14):
            cl = k % 2
            px = X + 18 + random.random() * 38 + cl * 46
            py = IY + 16 + random.random() * 70
            S.circ(px, py, 3.4, fill=K if cl else "#ffffff", stroke=K, sw=1.1)
        L([(X + 40, IY + 96), (X + 88, IY + 8)], INK, 1.2, dash=True)
    elif i == 1:                                 # a manifold and a geodesic
        top = [(X + 8, IY + 38), (X + 42, IY + 18), (X + 84, IY + 30),
               (X + 120, IY + 12)]
        bot = [(X + 10, IY + 86), (X + 44, IY + 66), (X + 86, IY + 78),
               (X + 122, IY + 58)]
        L(top, c, 1.2, curved=True)
        L(bot, c, 1.2, curved=True)
        L([top[0], bot[0]], c, 1.2)
        L([top[-1], bot[-1]], c, 1.2)
        L([(X + 30, IY + 54), (X + 64, IY + 40), (X + 100, IY + 44)], c, 2.2,
          curved=True)
        S.circ(X + 30, IY + 54, 4.2, fill=c)
        S.circ(X + 100, IY + 44, 4.2, fill=c)
    elif i == 2:                                 # two states on a Bloch sphere
        bloch(X + cw / 2, IY + IH / 2, 38, [-62, -12])
    else:                                        # a small circuit
        ys = [IY + 26, IY + 52, IY + 78]
        for yy in ys:
            L([(X + 8, yy), (X + cw - 8, yy)], INK, 1.0)
            S.rect(X + 14, yy - 7, 14, 14, fill="#ffffff", stroke=c, sw=1.1)
            S.rect(X + 34, yy - 7, 14, 14, fill="#ffffff", stroke=c, sw=1.1)
            S.rect(X + 98, yy - 7, 14, 14, fill="#ffffff", stroke=INK, sw=0.9)
            L([(X + 101, yy + 4), (X + 109, yy - 4)], INK, 0.9)
        for cxp, a, b in [(X + 64, 0, 1), (X + 80, 1, 2)]:
            L([(cxp, ys[a]), (cxp, ys[b] + 5)], c, 1.2)
            S.circ(cxp, ys[a], 3.0, fill=c)
            S.circ(cxp, ys[b], 5.4, stroke=c, sw=1.2)
    y = IY + IH + 10
    for it in items:
        txt(X, y, cw, 20, it, F_BODY, INK)
        y += 21

by = R2 + 268
bx1, bx2 = gx[1] - 4, gx[2] + cw + 4
L([(bx1, by - 9), (bx1, by), (bx2, by), (bx2, by - 9)], INK, 1.3)
L([((bx1 + bx2) / 2, by), ((bx1 + bx2) / 2, by + 9)], INK, 1.3)
txt(bx1 - 40, by + 12, bx2 - bx1 + 80, 22, "only the metric differs", F_BODY,
    INK, bold=True)

sx = dx + dwid / 2
S.rect(sx - 68, R2 + 312, 136, 28, fill="#ffffff", stroke=INK, sw=1.2, r=14)
txt(sx - 68, R2 + 312, 136, 28, "shared SVM", F_BODY, INK, bold=True)
L([(gx[0], R2 + 326), (sx - 74, R2 + 326)], LG, 1)
L([(sx + 74, R2 + 326), (gx[3] + cw, R2 + 326)], LG, 1)

# =========================================================== e  EVALUATION
ex, ewid = 580, 284
panel(ex, R2, ewid, "e", "Evaluation")

ox, oy = ex + 4, R2 + 44
for rr in range(3):
    for f in range(5):
        S.rect(ox + f * 19, oy + rr * 18, 16, 14,
               fill=INK if f == (rr * 2 + 1) % 5 else "#d8d8d8", r=1.5)
L([(ox + 95, oy), (ox + 148, oy - 4)], GRAY, 0.8, dash=True)
L([(ox + 95, oy + 14), (ox + 148, oy + 48)], GRAY, 0.8, dash=True)
S.rect(ox + 148, oy - 4, 106, 52, stroke=GRAY, sw=0.9, r=3)
for rr in range(2):
    for f in range(4):
        S.rect(ox + 156 + f * 23, oy + 4 + rr * 20, 19, 14,
               fill=Q if f == rr else QL, r=1.5)
txt(ox, oy + 56, 95, 18, "5 × 3", F_SMALL, GRAY)
txt(ox + 148, oy + 56, 106, 18, "4-fold", F_SMALL, GRAY)

px0, py0 = ex + 10, R2 + 128
random.seed(9)
for k in range(9):
    a = py0 + 8 + random.random() * 72
    b = a + random.uniform(-11, 11)
    L([(px0 + 16, a), (px0 + 78, b)], "#bcbcbc", 0.9)
    S.circ(px0 + 16, a, 3.0, fill=T)
    S.circ(px0 + 78, b, 3.0, fill=Q)
txt(px0 - 6, py0 + 88, 44, 16, "twin", F_SMALL, T)
txt(px0 + 56, py0 + 88, 46, 16, "quantum", F_SMALL, Q)
for i, t in enumerate(["Wilcoxon", "Holm", "TOST"]):
    txt(ex + 124, py0 + 6 + i * 28, 150, 22, t, F_BODY, INK, align="left")

sy = R2 + 244
for i, (t, sub) in enumerate([("subject", "within / LOO"), ("session", "cross"),
                              ("trials", "few"), ("qubits", "3 → 6")]):
    gxc = ex + 34 + i * 68
    S.circ(gxc, sy + 15, 15, stroke=INK, sw=1.1)
    txt(gxc - 34, sy + 36, 68, 16, t, F_SMALL, INK)
    txt(gxc - 34, sy + 52, 68, 16, sub, 10, GRAY)
g0 = ex + 34
S.circ(g0 - 5, sy + 11, 3.4, fill=INK)
S.circ(g0 + 5, sy + 11, 3.4, fill="#bbbbbb")
S.rect(g0 - 7, sy + 17, 14, 6, fill=INK, r=2)
g1 = g0 + 68
L([(g1 - 7, sy + 15), (g1 + 7, sy + 15)], INK, 1.4, arrow=True)
S.rect(g1 - 9, sy + 8, 3.5, 14, fill=INK)
g2 = g0 + 136
for k in range(3):
    S.rect(g2 - 7 + k * 5, sy + 9, 3.5, 12, fill=INK if k < 1 else "#bbbbbb")
g3 = g0 + 204
for k in range(3):
    S.circ(g3 - 7 + k * 7, sy + 15, 2.6, fill=Q)

L([(dx + dwid + 2, R2 + 120), (ex - 6, R2 + 120)], INK, 1.5, arrow=True)


# ------------------------------------------------------------------ write
(OUT / "architecture_tworow.drawio").write_text(S.drawio(), encoding="utf-8")
svg = S.svg()
(OUT / "architecture_tworow.svg").write_text(svg, encoding="utf-8")

# svglib resolves font-family by name against reportlab's registry, so the
# DejaVu fallback the scene graph asks for has to be registered.
import matplotlib  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402

_ttf = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
for _name, _file in (("DejaVu Sans", "DejaVuSans.ttf"),
                     ("DejaVu Sans-Bold", "DejaVuSans-Bold.ttf"),
                     ("DejaVu Sans-Italic", "DejaVuSans-Oblique.ttf"),
                     ("DejaVu Sans-BoldItalic", "DejaVuSans-BoldOblique.ttf")):
    pdfmetrics.registerFont(TTFont(_name, str(_ttf / _file)))
pdfmetrics.registerFontFamily("DejaVu Sans", normal="DejaVu Sans",
                              bold="DejaVu Sans-Bold",
                              italic="DejaVu Sans-Italic",
                              boldItalic="DejaVu Sans-BoldItalic")

from svglib.svglib import svg2rlg  # noqa: E402
from reportlab.graphics import renderPDF  # noqa: E402

drawing = svg2rlg(str(OUT / "architecture_tworow.svg"))
renderPDF.drawToFile(drawing, str(OUT / "architecture_tworow.pdf"))

import fitz  # noqa: E402

_doc = fitz.open(str(OUT / "architecture_tworow.pdf"))
_doc[0].get_pixmap(matrix=fitz.Matrix(1.6, 1.6)).save(
    str(OUT / "architecture_tworow_preview.png"))

# This is the alternative layout, kept so the corresponding author can switch
# back to it. It is deliberately NOT copied into results/figures: the figure
# the manuscript uses is built from figures/architecture.drawio by
# figures/src/drawio_to_pdf.py.

_pt = F_BODY * (160.0 / W) / (25.4 / 72)
print(f"wrote figures/versions/architecture_tworow.* (alternative layout); "
      f"body labels render at {_pt:.1f} pt at 160 mm")
