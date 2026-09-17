"""Build the architecture figure: one scene graph, three outputs.

    python figures/src/make_architecture.py

Writes figures/architecture.drawio (editable, native shapes),
figures/architecture.svg, figures/architecture.pdf (vector) and
figures/architecture_preview.png, and copies the PDF to results/figures/ so
the manuscript's graphicspath picks it up.

cairosvg is not usable here: it binds to a native libcairo that Windows does
not ship, so the SVG is rasterised and converted with svglib and reportlab
instead.
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
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

W, H = 1560, 400
S = Scene(W, H)
INK, GRAY, LG, FILL = "#222222", "#707070", "#cfcfcf", "#f3f3f3"
Q, QL = "#2f5d9e", "#e5edf7"   # quantum, reference frame, selected electrodes
T = "#a4503a"                   # metric-matched twin
K = "#555555"                   # classical, sensor frame
P = "#6a5a9c"                   # circuit kernels


def txt(x, y, w, h, t, size=12, color=INK, **kw):
    S.text(x, y, w, h, t, size, color, **kw)


def L(pts, c=INK, sw=1, **kw):
    S.line(pts, stroke=c, sw=sw, **kw)


def panel(x, w, letter, title):
    txt(x, 14, 20, 24, letter, 18, INK, bold=True, align="left")
    txt(x + 22, 14, w - 22, 24, title, 15, INK, bold=True, align="left")
    L([(x, 44), (x + w, 44)], LG, 1)


def bloch(cx, cy, r, vecs, c=Q):
    S.circ(cx, cy, r, fill="#ffffff", stroke=c, sw=1.1)
    S.ell(cx - r, cy - r * 0.3, 2 * r, r * 0.6, stroke=c, sw=0.8, dash=True)
    L([(cx, cy - r), (cx, cy + r)], LG, 0.8)
    for i, a in enumerate(vecs):
        a = math.radians(a)
        L([(cx, cy), (cx + 0.85 * r * math.cos(a), cy + 0.85 * r * math.sin(a))],
          [c, T][i % 2], 1.6, arrow=True)
    S.circ(cx, cy, 2, fill=c)


def ellipse_pts(cx, cy, a, b, ang):
    out = []
    for k in range(33):
        t = 2 * math.pi * k / 32
        ex, ey = a * math.cos(t), b * math.sin(t)
        r = math.radians(ang)
        out.append((cx + ex * math.cos(r) - ey * math.sin(r),
                    cy + ex * math.sin(r) + ey * math.cos(r)))
    return out


# ================================================================== a  EEG
x0, w = 20, 270
panel(x0, w, "a", "EEG")
hx, hy, r = x0 + 70, 150, 58
S.tri(hx - 8, hy - r - 11, 16, 13, fill="#ffffff", stroke=INK, sw=1)
S.ell(hx - r - 5, hy - 10, 10, 20, fill="#ffffff", stroke=INK, sw=1)
S.ell(hx + r - 5, hy - 10, 10, 20, fill="#ffffff", stroke=INK, sw=1)
S.circ(hx, hy, r, fill="#ffffff", stroke=INK, sw=1.2)
sel = {(-0.32, -0.37), (-0.32, 0.0), (-0.32, 0.37), (0.0, -0.45), (0.0, 0.0),
       (0.0, 0.45), (0.32, -0.37), (0.32, 0.37)}
grid = [(-0.66, [-0.3, 0, 0.3]),
        (-0.32, [-0.72, -0.37, 0, 0.37, 0.72]),
        (0.0, [-0.82, -0.45, 0, 0.45, 0.82]),
        (0.32, [-0.72, -0.37, 0, 0.37, 0.72]),
        (0.66, [-0.3, 0, 0.3])]
for ry, xs in grid:
    for rx in xs:
        px, py = hx + rx * r, hy + ry * r
        if (ry, rx) in sel:
            S.circ(px, py, 4.5, fill=Q)
        else:
            S.circ(px, py, 2.8, fill="#ffffff", stroke="#9a9a9a", sw=0.8)
random.seed(2)
for i in range(8):
    y = 92 + i * 16
    ph, pts = random.random() * 6, []
    for k in range(41):
        t = k / 40
        v = (0.55 * math.sin(2 * math.pi * 3.2 * t + ph)
             + 0.3 * math.sin(2 * math.pi * 7 * t + 2 * ph)
             + 0.15 * math.sin(2 * math.pi * 13 * t))
        pts.append((x0 + 156 + 110 * t, y + 5.5 * v))
    L(pts, Q if i % 2 == 0 else INK, 0.8, curved=True)
L([(x0 + 156, 228), (x0 + 266, 228)], INK, 0.8)
txt(x0 + 156, 230, 110, 14, "1 s", 10, GRAY)
for i, name in enumerate(["EEGMMIDB", "BCI IV 2a", "Cho2017"]):
    cx = x0 + 10 + i * 88
    S.cyl(cx, 272, 18, 24, fill="#ffffff", stroke=INK, sw=1)
    txt(cx + 24, 276, 64, 16, name, 10.5, INK, align="left")
fx, fy = x0 + 6, 322
L([(fx, fy + 34), (fx + 110, fy + 34)], GRAY, 0.8)
L([(fx + 2, fy + 33), (fx + 26, fy + 32), (fx + 36, fy + 6), (fx + 72, fy + 6),
   (fx + 82, fy + 32), (fx + 108, fy + 33)], Q, 1.4, curved=True)
txt(fx + 120, fy + 10, 140, 18, "8–30 Hz · 8 ch", 11.5, INK, align="left")

# ================================================================ b  STATE
x0, w = 320, 230
panel(x0, w, "b", "State")


def blue(v):
    a, b = (255, 255, 255), (32, 72, 138)
    return "#%02x%02x%02x" % tuple(int(a[k] + v * (b[k] - a[k])) for k in range(3))


hm_x, hm_y, cs = x0 + 4, 70, 15
for i in range(8):
    for j in range(8):
        v = math.exp(-abs(i - j) / 2.4) * (0.8 + 0.2 * math.cos(i + j))
        v = min(1, max(0.03, v + 0.06 * math.sin(3.1 * (i + j))))
        S.rect(hm_x + j * cs, hm_y + i * cs, cs, cs, fill=blue(v),
               stroke="#ffffff", sw=0.5)
S.rect(hm_x, hm_y, 8 * cs, 8 * cs, stroke=INK, sw=0.8)
txt(hm_x + 8 * cs + 10, hm_y + 44, 80, 30, "<i>C</i>", 22, INK, align="left")
txt(hm_x + 8 * cs + 10, hm_y + 74, 80, 18, "8 × 8", 11, GRAY, align="left")

L([(hm_x + 60, 200), (hm_x + 60, 236)], INK, 1.2, arrow=True)
txt(hm_x + 70, 208, 80, 20, "÷ tr <i>C</i>", 12, GRAY, align="left")

ry0 = 268
for q in range(3):
    y = ry0 + q * 26
    L([(x0 + 4, y), (x0 + 40, y)], INK, 1)
    L([(x0 + 92, y), (x0 + 130, y)], INK, 1)
S.rect(x0 + 40, ry0 - 14, 52, 80, fill=QL, stroke=Q, sw=1.2, r=2)
txt(x0 + 40, ry0 - 14, 52, 80, "<i>ρ</i>", 24, Q)
bloch(x0 + 178, ry0 + 26, 34, [-55])
txt(x0 + 134, ry0 + 64, 90, 16, "3 qubits", 11, Q)

# ================================================================ c  FRAME
x0, w = 580, 280
panel(x0, w, "c", "Frame")
txt(x0, 56, w, 26, "<i>C</i> → <i>A C A</i><sup><i>T</i></sup>", 16, INK)
bw = 132
for i, (title, col) in enumerate([("sensor", K), ("reference", Q)]):
    bx = x0 + i * (bw + 16)
    S.rect(bx, 96, bw, 270, fill="#ffffff", stroke=LG, sw=1, r=3)
    S.rect(bx, 96, bw, 3, fill=col)
    txt(bx, 106, bw, 20, title, 13, col, bold=True)
    cx, cy = bx + bw / 2, 190
    if i == 0:
        L(ellipse_pts(cx, cy, 40, 17, -20), col, 1.2, curved=True)
        L(ellipse_pts(cx, cy, 40, 17, -55), "#9a9a9a", 1.2, curved=True, dash=True)
        L([(cx + 34, cy - 34), (cx + 44, cy - 20), (cx + 42, cy - 6)], GRAY, 1,
          curved=True, arrow=True)
        txt(bx, 262, bw, 22, "<i>ρ</i> = <i>C</i> / tr <i>C</i>", 13, INK)
        txt(bx, 300, bw, 22, "<i>O C O</i><sup><i>T</i></sup> only", 13, K)
        S.circ(cx, 340, 8, stroke=K, sw=1.2)
        L([(cx - 5, 335), (cx + 5, 345)], K, 1.2)
    else:
        L(ellipse_pts(cx - 32, cy, 30, 12, -25), GRAY, 1.1, curved=True)
        L([(cx - 4, cy), (cx + 14, cy)], INK, 1, arrow=True)
        txt(cx - 8, cy - 20, 24, 14, "<i>W</i>", 11, INK)
        S.circ(cx + 36, cy, 19, stroke=col, sw=1.3)
        txt(bx, 262, bw, 22, "<i>W</i> = <i>M</i><sup>−1/2</sup>", 13, Q)
        txt(bx, 300, bw, 22, "<i>A C A</i><sup><i>T</i></sup>", 13, Q)
        S.circ(cx, 340, 8, fill=Q)
        L([(cx - 4, 340), (cx - 1, 343), (cx + 4, 336)], "#ffffff", 1.6)

# ============================================================ d  PIPELINES
x0, w = 890, 390
panel(x0, w, "d", "Kernels")
cw, gp = 88, 12
gx = [x0 + 4 + i * (cw + gp) for i in range(4)]
groups = [("Classical", K, ["CSP", "TS", "MDM", "FBCSP"]),
          ("Twin", T, ["Riemann", "log-Eucl."]),
          ("Quantum", Q, ["HS", "Fidelity", "Bures", "QRE"]),
          ("Circuit", P, ["IQP", "IQP − ent.", "PCA"])]
for i, (title, c, items) in enumerate(groups):
    X = gx[i]
    S.rect(X, 58, cw, 3, fill=c)
    txt(X, 64, cw, 20, title, 12.5, c, bold=True)
    iy = 92
    S.rect(X, iy, cw, 84, fill=FILL, r=2)
    if i == 0:
        random.seed(5)
        for k in range(12):
            cl = k % 2
            px = X + 12 + random.random() * 26 + cl * 36
            py = iy + 14 + random.random() * 56
            S.circ(px, py, 2.8, fill=K if cl else "#ffffff", stroke=K, sw=1)
        L([(X + 30, iy + 78), (X + 62, iy + 6)], INK, 1, dash=True)
    elif i == 1:
        L([(X + 6, iy + 32), (X + 30, iy + 16), (X + 58, iy + 24), (X + 82, iy + 10)],
          c, 1, curved=True)
        L([(X + 8, iy + 70), (X + 32, iy + 54), (X + 60, iy + 62), (X + 84, iy + 46)],
          c, 1, curved=True)
        L([(X + 6, iy + 32), (X + 8, iy + 70)], c, 1)
        L([(X + 82, iy + 10), (X + 84, iy + 46)], c, 1)
        L([(X + 22, iy + 44), (X + 44, iy + 34), (X + 68, iy + 36)], c, 1.8, curved=True)
        S.circ(X + 22, iy + 44, 3.5, fill=c)
        S.circ(X + 68, iy + 36, 3.5, fill=c)
    elif i == 2:
        bloch(X + cw / 2, iy + 42, 30, [-60, -15])
    else:
        ys = [iy + 20, iy + 42, iy + 64]
        for yy in ys:
            L([(X + 4, yy), (X + cw - 4, yy)], INK, 0.9)
            S.rect(X + 8, yy - 6, 12, 12, fill="#ffffff", stroke=c, sw=1)
            S.rect(X + 24, yy - 6, 12, 12, fill="#ffffff", stroke=c, sw=1)
            S.rect(X + 68, yy - 6, 12, 12, fill="#ffffff", stroke=INK, sw=0.8)
            L([(X + 70, yy + 3), (X + 78, yy - 3)], INK, 0.8)
        for cxp, a, b in [(X + 46, 0, 1), (X + 58, 1, 2)]:
            L([(cxp, ys[a]), (cxp, ys[b] + 4)], c, 1)
            S.circ(cxp, ys[a], 2.5, fill=c)
            S.circ(cxp, ys[b], 4.5, stroke=c, sw=1)
    y = 188
    for it in items:
        txt(X, y, cw, 20, it, 11.5, INK)
        y += 22
by = 296
bx1, bx2 = gx[1], gx[2] + cw
L([(bx1, by - 8), (bx1, by), (bx2, by), (bx2, by - 8)], INK, 1)
L([((bx1 + bx2) / 2, by), ((bx1 + bx2) / 2, by + 8)], INK, 1)
txt(bx1 - 30, by + 12, bx2 - bx1 + 60, 20, "only the metric differs", 12.5, INK,
    bold=True)
sx = x0 + w / 2
S.rect(sx - 60, 344, 120, 26, fill="#ffffff", stroke=INK, sw=1, r=13)
txt(sx - 60, 344, 120, 26, "shared SVM", 12, INK, bold=True)
L([(gx[0], 357), (sx - 64, 357)], LG, 1)
L([(sx + 64, 357), (gx[3] + cw, 357)], LG, 1)

# =========================================================== e  EVALUATION
x0, w = 1310, 230
panel(x0, w, "e", "Evaluation")
ox, oy = x0, 66
for rr in range(3):
    for f in range(5):
        S.rect(ox + f * 17, oy + rr * 16, 14, 12,
               fill=INK if f == (rr * 2 + 1) % 5 else "#d8d8d8", r=1)
L([(ox + 34 + 14, oy), (ox + 128, oy - 2)], GRAY, 0.6, dash=True)
L([(ox + 34 + 14, oy + 12), (ox + 128, oy + 40)], GRAY, 0.6, dash=True)
S.rect(ox + 128, oy - 2, 92, 42, stroke=GRAY, sw=0.8, r=2)
for rr in range(2):
    for f in range(4):
        S.rect(ox + 135 + f * 20, oy + 4 + rr * 17, 17, 12,
               fill=Q if f == rr else QL, r=1)
txt(ox, oy + 50, 90, 14, "5 × 3", 11, GRAY)
txt(ox + 128, oy + 50, 92, 14, "4-fold", 11, GRAY)

px0, py0 = x0 + 10, 170
random.seed(9)
for k in range(9):
    a = py0 + 8 + random.random() * 76
    b = a + random.uniform(-10, 10)
    L([(px0 + 14, a), (px0 + 70, b)], "#b8b8b8", 0.8)
    S.circ(px0 + 14, a, 2.6, fill=T)
    S.circ(px0 + 70, b, 2.6, fill=Q)
txt(px0 - 6, py0 + 90, 40, 14, "twin", 10, T)
txt(px0 + 50, py0 + 90, 44, 14, "quantum", 10, Q)
for i, t in enumerate(["Wilcoxon", "Holm", "TOST"]):
    txt(x0 + 116, py0 + 8 + i * 26, 110, 20, t, 12, INK, align="left")

sy = 300
for i, (t, sub) in enumerate([("subject", "within / LOO"), ("session", "cross"),
                              ("trials", "few"), ("qubits", "3 → 6")]):
    cx = x0 + 28 + i * 58
    S.circ(cx, sy + 14, 13, stroke=INK, sw=1)
    txt(cx - 30, sy + 32, 60, 14, t, 10, INK)
    txt(cx - 30, sy + 46, 60, 14, sub, 9, GRAY)
c0 = x0 + 28
S.circ(c0 - 4, sy + 11, 3, fill=INK)
S.circ(c0 + 4, sy + 11, 3, fill="#bbbbbb")
S.rect(c0 - 6, sy + 16, 12, 5, fill=INK, r=2)
c1 = c0 + 58
L([(c1 - 6, sy + 14), (c1 + 6, sy + 14)], INK, 1.2, arrow=True)
S.rect(c1 - 8, sy + 8, 3, 12, fill=INK)
c2 = c0 + 116
for k in range(3):
    S.rect(c2 - 6 + k * 4, sy + 9, 3, 10, fill=INK if k < 1 else "#bbbbbb")
c3 = c0 + 174
for k in range(3):
    S.circ(c3 - 6 + k * 6, sy + 14, 2.2, fill=Q)

for xa, xb in [(290, 320), (550, 580), (860, 890), (1280, 1310)]:
    L([(xa + 6, 200), (xb - 4, 200)], INK, 1.3, arrow=True)


# ------------------------------------------------------------------ write
(OUT / "architecture.drawio").write_text(S.drawio(), encoding="utf-8")
svg = S.svg()
(OUT / "architecture.svg").write_text(svg, encoding="utf-8")

# svglib resolves font-family by name against reportlab's registry, so the
# DejaVu fallback the scene graph asks for has to be registered or every glyph
# Arial lacks comes out as an empty box.
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

drawing = svg2rlg(str(OUT / "architecture.svg"))
renderPDF.drawToFile(drawing, str(OUT / "architecture.pdf"))

# The preview is rasterised from the PDF with PyMuPDF rather than by
# reportlab's renderPM, whose backend needs a native Cairo this machine does
# not have. Going through the PDF also means the preview shows exactly what
# the published vector shows, which is the point of looking at it.
import fitz  # noqa: E402

_doc = fitz.open(str(OUT / "architecture.pdf"))
_doc[0].get_pixmap(matrix=fitz.Matrix(1.2, 1.2)).save(
    str(OUT / "architecture_preview.png"))

# The manuscript's graphicspath includes results/figures, so publish there too.
dest = ROOT / "results" / "figures"
dest.mkdir(parents=True, exist_ok=True)
shutil.copy2(OUT / "architecture.pdf", dest / "architecture.pdf")
print("wrote figures/architecture.{drawio,svg,pdf,_preview.png} "
      "and results/figures/architecture.pdf")
