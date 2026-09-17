"""Two pipeline figures: a one-line overview, and its kernel block expanded.

    python figures/src/make_architecture_parts.py    # real-data panels first
    python figures/src/make_pipeline_figures.py      # then this

Outputs, in figures/experiments/architecture_parts/:

  fig_pipeline_overview.drawio   EEG, channel selection, band-pass, epoching,
                                 covariance, the "Kernel families" block,
                                 kernel matrix, SVM, predicted class
  fig_kernel_families.drawio     the block opened up: 1 quantum density-matrix
                                 kernels, 2 quantum circuit kernels, 3 the
                                 classical twin, from C in to K out

The block's three numbered rows are the three numbered lanes of the second
figure, so a reader can map one onto the other. What each lane states is taken
from the code (see make_pipeline_diagram.py, which this supersedes, for the
provenance of every formula). Export each file to PDF with diagrams.net
(File, Export as, PDF, Crop) or draw.io --export --format pdf --crop.
"""
from __future__ import annotations

import base64
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARTS = HERE.parent / "experiments" / "architecture_parts"

INK, MUTED = "#263238", "#546e7a"

TEXT = (f"text;html=1;whiteSpace=wrap;align=center;verticalAlign=middle;"
        f"fontFamily=Helvetica;fontColor={INK};")
PRE = (f"rounded=1;arcSize=10;whiteSpace=wrap;html=1;fillColor=#f5fbff;"
       f"strokeColor=#9cc8ea;fontFamily=Helvetica;fontColor={INK};fontSize=12;")
FRAME = (f"rounded=1;arcSize=10;whiteSpace=wrap;html=1;fillColor=#ffffff;"
         f"strokeColor=#a5d6b4;fontFamily=Helvetica;fontColor={INK};fontSize=12;")
ITEM = (f"rounded=1;arcSize=12;whiteSpace=wrap;html=1;fillColor=#ede9fb;"
        f"strokeColor=#b8aee6;fontFamily=Helvetica;fontColor={INK};fontSize=11;")
TWIN_ITEM = (f"rounded=1;arcSize=12;whiteSpace=wrap;html=1;fillColor=#fbeee8;"
             f"strokeColor=#e0b4a0;fontFamily=Helvetica;fontColor={INK};fontSize=11;")
LANE = ("rounded=1;arcSize=2;whiteSpace=wrap;html=1;fillColor=none;"
        "strokeColor=#9fa8da;dashed=1;")
TWIN_LANE = ("rounded=1;arcSize=3;whiteSpace=wrap;html=1;fillColor=none;"
             "strokeColor=#d7a58f;dashed=1;")
GROUP = ("rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#fafaff;"
         "strokeColor=#cfc8f0;")
BLOCK = ("rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#fafaff;"
         "strokeColor=#9fa8da;dashed=1;strokeWidth=1.4;")
SWAPBOX = ("rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#ffffff;"
           "strokeColor=#9f8fe0;dashed=1;")
SVMBOX = ("rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#ffffff;"
          "strokeColor=#9f8fe0;")
PRED = (f"rounded=1;arcSize=12;whiteSpace=wrap;html=1;fillColor=#fff2cc;"
        f"strokeColor=#d6b656;fontFamily=Helvetica;fontColor={INK};fontSize=13;")
IMAGE = "shape=image;imageAspect=0;aspect=fixed;verticalLabelPosition=bottom;verticalAlign=top;"
EDGE = (f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;html=1;"
        f"endArrow=block;endFill=1;strokeColor={INK};strokeWidth=1.3;")
CALLOUT = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;dashed=1;"
           "endArrow=open;endFill=0;strokeColor=#7e57c2;strokeWidth=1.1;")


def png(name: str) -> str:
    return "data:image/png," + base64.b64encode((PARTS / f"{name}.png").read_bytes()).decode()


def port(exit_=None, entry=None) -> str:
    s = ""
    if exit_:
        s += f"exitX={exit_[0]};exitY={exit_[1]};exitDx=0;exitDy=0;"
    if entry:
        s += f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
    return s


class Diagram:
    def __init__(self, name: str, page_w: int, page_h: int):
        self.root = ET.Element("mxfile", host="app.diagrams.net")
        d = ET.SubElement(self.root, "diagram", name=name, id=name)
        m = ET.SubElement(d, "mxGraphModel", dx="0", dy="0", grid="0",
                          gridSize="10", guides="1", tooltips="1", connect="1",
                          arrows="1", fold="1", page="1", pageScale="1",
                          pageWidth=str(page_w), pageHeight=str(page_h),
                          background="#FFFFFF", math="1", shadow="0")
        self.top = ET.SubElement(m, "root")
        ET.SubElement(self.top, "mxCell", id="0")
        ET.SubElement(self.top, "mxCell", id="1", parent="0")

    def v(self, cid, value, style, x, y, w, h):
        c = ET.SubElement(self.top, "mxCell", id=cid, value=value, style=style,
                          vertex="1", parent="1")
        ET.SubElement(c, "mxGeometry", x=f"{x:.2f}", y=f"{y:.2f}",
                      width=f"{w:.2f}", height=f"{h:.2f}", **{"as": "geometry"})

    def img(self, cid, name, x, y, w, h):
        self.v(cid, "", IMAGE + "image=" + png(name) + ";", x, y, w, h)

    def e(self, cid, src, tgt, ports="", points=(), style=EDGE):
        c = ET.SubElement(self.top, "mxCell", id=cid, value="", style=style + ports,
                          edge="1", parent="1", source=src, target=tgt)
        g = ET.SubElement(c, "mxGeometry", relative="1", **{"as": "geometry"})
        if points:
            arr = ET.SubElement(g, "Array", **{"as": "points"})
            for x, y in points:
                ET.SubElement(arr, "mxPoint", x=f"{x:.2f}", y=f"{y:.2f}")

    def title(self, cid, text, x, y, w, size=13, align="left"):
        self.v(cid, f"<b>{text}</b>", TEXT + f"fontSize={size};align={align};",
               x, y, w, 20)

    def write(self, path: Path):
        ET.indent(self.root, space="  ")
        ET.ElementTree(self.root).write(path, encoding="utf-8", xml_declaration=False)
        print("wrote", path)


BLOCK_NAME = "Kernel families"
ROWS = [("1", "Density-matrix kernels", ITEM),
        ("2", "Circuit kernels", ITEM),
        ("3", "Classical twin", TWIN_ITEM)]


# ======================================================== figure A: overview
def overview() -> None:
    d = Diagram("overview", 1370, 270)
    pitch, c0, m = 22.0, 52.0, 0.6
    mid = c0 + 3.5 * pitch
    tr_h = pitch * (7 + 2 * m)
    tr_w = tr_h * 2.4 / 2.6
    tr_y = c0 - tr_h * m / (7 + 2 * m)

    d.title("t-eeg", "Motor-imagery EEG", 20, 14, 220)
    for i, ch in enumerate(["FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CP4"]):
        d.v(f"ch-{ch}", ch, TEXT + "fontSize=11;align=right;", 20, c0 + i * pitch - 8, 34, 16)
    d.img("eeg", "1_eeg_traces", 58, tr_y, tr_w, tr_h)
    d.v("x-label", r"\(X \in \mathbb{R}^{8 \times T}\)", TEXT + "fontSize=13;",
        58, tr_y + tr_h + 2, tr_w, 22)

    x = 58 + tr_w + 32
    prev = "eeg"
    for cid, label in [("chsel", "Channel selection"), ("bpf", "Band-pass filter"),
                       ("epoch", "Epoching")]:
        d.v(cid, label, PRE + "horizontal=0;", x, mid - 62, 42, 124)
        d.e(f"a-{cid}", prev, cid, port((1, 0.5), (0, 0.5)))
        prev, x = cid, x + 42 + 30

    x += 8
    d.title("t-cov", "Covariance matrix", x - 25, mid - 76, 150, align="center")
    d.img("cov", "2_covariance", x, mid - 52, 104, 104)
    d.v("cov-eq", r"\(C \in \mathbb{S}^{8}_{++}\)", TEXT + "fontSize=13;",
        x - 10, mid + 54, 124, 22)
    d.e("a-cov", prev, "cov", port((1, 0.5), (0, 0.5)))

    x += 104 + 40
    bw, bh = 212, 150
    d.v("block", "", BLOCK, x, mid - bh / 2, bw, bh)
    d.v("block-title", f"<b>{BLOCK_NAME}</b>", TEXT + "fontSize=13;",
        x, mid - bh / 2 + 6, bw, 22)
    for i, (num, name, style) in enumerate(ROWS):
        d.v(f"row-{num}", f"<b>{num}</b>&nbsp;&nbsp;{name}", style + "fontSize=12;",
            x + 14, mid - bh / 2 + 36 + i * 37, bw - 28, 29)
    d.e("a-block", "cov", "block", port((1, 0.5), (0, 0.5)))

    x += bw + 40
    d.title("t-K", "Kernel matrix", x - 15, mid - 76, 140, align="center")
    d.img("K", "4_kernel_matrix", x, mid - 52, 104, 104)
    d.v("K-eq", r"\(K_{ij} = k(\cdot_i, \cdot_j)\)", TEXT + "fontSize=13;",
        x - 18, mid + 54, 140, 22)
    d.e("a-K", "block", "K", port((1, 0.5), (0, 0.5)))

    x += 104 + 40
    sw, sh = 176, 176
    d.v("svm", "", SVMBOX, x, mid - sh / 2, sw, sh)
    d.v("svm-title", "<b>SVM</b> (precomputed kernel)", TEXT + "fontSize=12;",
        x, mid - sh / 2 + 6, sw, 20)
    iw = sw - 24
    d.img("svm-img", "5_svm", x + 12, mid - sh / 2 + 32, iw, iw * 1132 / 1203)
    d.e("a-svm", "K", "svm", port((1, 0.5), (0, 0.5)))

    x += sw + 40
    d.v("pred", "<b>Predicted class</b>", PRED, x, mid - 24, 140, 48)
    d.e("a-pred", "svm", "pred", port((1, 0.5), (0, 0.5)))
    d.write(PARTS / "fig_pipeline_overview.drawio")


# ================================================ figure B: the block expanded
def kernel_families() -> None:
    d = Diagram("kernel-families", 1150, 660)
    ox = 70                                   # lanes sit right of the C input
    LX, LW = 60 + ox, 860

    d.title("t-block", BLOCK_NAME, 20, 12, 300, size=15)
    v, img, e, title = d.v, d.img, d.e, d.title

    # ---- lane 1: quantum density-matrix kernels --------------------------
    l1y, l1h = 50, 238
    v("lane1", "", LANE, LX, l1y, LW, l1h)
    title("t-lane1", "1&nbsp;&nbsp;Quantum density-matrix kernels", LX + 12, l1y + 6, 400)
    v("frame", r"<b>Reference frame</b><br>\(W = M^{-1/2}\)<br>\(\tilde C = W C W\)",
      FRAME, 80 + ox, l1y + 56, 124, 76)
    v("frame-note", r"\(M\): Fréchet mean of the training covariances<br>"
                    r"sensor frame: \(W = I\)",
      TEXT + f"fontSize=9;fontColor={MUTED};", 70 + ox, l1y + 136, 144, 44)
    title("t-rho", "Density matrix", 246 + ox, l1y + 30, 120, size=12, align="center")
    img("rho", "3_density_matrix", 256 + ox, l1y + 52, 100, 100)
    v("rho-eq", r"\(\rho = \tilde C / \operatorname{tr}\tilde C\)<br>"
                r"\(\rho \succ 0,\ \operatorname{tr}\rho = 1\)",
      TEXT + "fontSize=12;", 226 + ox, l1y + 156, 160, 44)
    e("k1", "frame", "rho", port((1, 0.5), (0, 0.5)))

    g1x, g1y, g1w, g1h = 410 + ox, l1y + 34, 200, 196
    v("kd-box", "", GROUP, g1x, g1y, g1w, g1h)
    v("kd-title", r"\(k(\rho_i, \rho_j)\), one per run", TEXT + "fontSize=12;",
      g1x, g1y + 4, g1w, 22)
    for i, name in enumerate(["Hilbert–Schmidt overlap", "Fidelity", "HS-RBF",
                              "Bures-RBF", "QRE-RBF"]):
        v(f"kd-{i}", name, ITEM, g1x + 14, g1y + 32 + i * 32, g1w - 28, 26)
    e("k2", "rho", "kd-box", port((1, 0.5), (0, 0.44)))

    sx, sy, sw_, sh_ = 632 + ox, l1y + 44, 278, 150
    v("swap-box", "", SWAPBOX, sx, sy, sw_, sh_)
    v("swap-title", "<b>SWAP test</b> for the overlap", TEXT + "fontSize=12;",
      sx, sy + 4, sw_, 20)
    swap_h = 112
    swap_w = swap_h * 1988 / 2243
    img("swap", "7_swap_test", sx + 8, sy + 28, swap_w, swap_h)
    v("swap-eq", r"\(P(a{=}0) = \tfrac{1}{2}\,(1 + \operatorname{tr}\rho\sigma)\)"
                 r"<br><span style='font-size:10px;color:#546e7a'>how the overlap would be "
                 r"measured on hardware; simulated in the finite-shot analysis</span>",
      TEXT + "fontSize=11;", sx + swap_w + 14, sy + 34, sw_ - swap_w - 20, 104)
    e("callout", "kd-0", "swap-box", port((1, 0.5), (0, 0.3)),
      points=[(618 + ox, g1y + 32 + 13), (618 + ox, sy + 0.3 * sh_)], style=CALLOUT)

    # ---- lane 2: quantum circuit kernels ---------------------------------
    l2y, l2h = 308, 172
    v("lane2", "", LANE, LX, l2y, LW, l2h)
    title("t-lane2", "2&nbsp;&nbsp;Quantum circuit kernels", LX + 12, l2y + 6, 400)
    v("t-lane2-sub", "4 qubits; IQP, ring CNOT, and the entanglement ablation",
      TEXT + f"fontSize=10;fontColor={MUTED};align=left;", LX + 206, l2y + 7, 400, 18)
    for i, (cid, label) in enumerate([("ts", "Tangent space"),
                                      ("pca", "Standardise, PCA to 4"),
                                      ("ang", r"Angles \(x \in [0,\pi]^4\)")]):
        v(cid, label, PRE + "fontSize=11;", 80 + ox, l2y + 38 + i * 42, 140, 26)
    e("c1", "ts", "pca", port((0.5, 1), (0.5, 0)))
    e("c2", "pca", "ang", port((0.5, 1), (0.5, 0)))
    cw = 290
    ch_ = cw * 1213 / 3283
    img("circ", "6_circuit_iqp", 250 + ox, l2y + 30, cw, ch_)
    v("circ-note", r"feature map \(U(x)\), one of two layers shown; "
                   r"\(z_{i,i+1} = (\pi - x_i)(\pi - x_{i+1})\)",
      TEXT + f"fontSize=10;fontColor={MUTED};", 240 + ox, l2y + 32 + ch_, 310, 30)
    e("c3", "ang", "circ", port((1, 0.5), (0, 0.5)),
      points=[(236 + ox, l2y + 38 + 2 * 42 + 13), (236 + ox, l2y + 30 + ch_ / 2)])
    g2x, g2y, g2w = 580 + ox, l2y + 30, 220
    v("kc-box", "", GROUP, g2x, g2y, g2w, 130)
    v("kc-title", r"\(k(x_i, x_j) = |\langle \psi(x_j) \mid \psi(x_i) \rangle|^2\)",
      TEXT + "fontSize=12;", g2x, g2y + 4, g2w, 24)
    for i, name in enumerate(["IQP feature map", "Ring CNOT", "No entanglers (ablation)"]):
        v(f"kc-{i}", name, ITEM, g2x + 14, g2y + 34 + i * 31, g2w - 28, 25)
    e("c4", "circ", "kc-box", port((1, 0.5), (0, 0.5)))

    # ---- lane 3: the classical twin --------------------------------------
    l3y, l3h = 500, 112
    v("lane3", "", TWIN_LANE, LX, l3y, LW, l3h)
    title("t-lane3", "3&nbsp;&nbsp;Classical twin (control)", LX + 12, l3y + 6, 240)
    v("t-lane3-sub", "same covariances, same frame, same SVM and tuning budget: "
                     "only the metric differs",
      TEXT + f"fontSize=10;fontColor={MUTED};align=left;", LX + 200, l3y + 7, 460, 18)
    v("twin-frame", r"Same reference frame<br>\(\tilde C = W C W\)",
      FRAME + "strokeColor=#e0b4a0;fontSize=11;", 80 + ox, l3y + 42, 140, 48)
    g3x, g3y, g3w = 410 + ox, l3y + 34, 300
    v("kt-box", "", GROUP + "fillColor=#fffaf8;strokeColor=#ecc9b8;", g3x, g3y, g3w, 70)
    v("kt-0", r"Riemannian: \(k(C_i, C_j) = \operatorname{tr}[\log\tilde C_i \,\log\tilde C_j]\)",
      TWIN_ITEM, g3x + 12, g3y + 8, g3w - 24, 26)
    v("kt-1", "log-Euclidean, reported alongside", TWIN_ITEM + "fontColor=#6d4c41;",
      g3x + 12, g3y + 38, g3w - 24, 24)
    e("t1", "twin-frame", "kt-box", port((1, 0.5), (0, 0.5)))

    # ---- C in, K out: the same symbols as the overview's arrows ----------
    cy = l2y + l2h / 2
    v("t-C", "Covariance", TEXT + "fontSize=11;", 8, cy - 58, 84, 18)
    img("C", "2_covariance", 20, cy - 36, 60, 60)
    v("C-sym", r"\(C\)", TEXT + "fontSize=14;", 20, cy + 26, 60, 20)
    BUS_X = 104
    for cid, tgt, ty in (("fork1", "frame", l1y + 56 + 38),
                         ("fork2", "ts", l2y + 38 + 13),
                         ("fork3", "twin-frame", l3y + 42 + 24)):
        e(cid, "C", tgt, port((1, 0.5), (0, 0.5)),
          points=[(BUS_X, cy - 6), (BUS_X, ty)])

    kx = LX + LW + 50
    v("t-Kout", "Kernel matrix", TEXT + "fontSize=11;", kx - 14, cy - 58, 90, 18)
    img("Kout", "4_kernel_matrix", kx, cy - 36, 60, 60)
    v("K-sym", r"\(K\)", TEXT + "fontSize=14;", kx, cy + 26, 60, 20)
    MERGE_X = LX + LW + 24
    for cid, src, ey, sy_ in (("m1", "kd-box", 0.9, g1y + 0.9 * g1h),
                              ("m2", "kc-box", 0.5, g2y + 65),
                              ("m3", "kt-box", 0.5, g3y + 35)):
        e(cid, src, "Kout", port((1, ey), (0, 0.5)),
          points=[(MERGE_X, sy_), (MERGE_X, cy - 6)])
    d.write(PARTS / "fig_kernel_families.drawio")


if __name__ == "__main__":
    overview()
    kernel_families()
