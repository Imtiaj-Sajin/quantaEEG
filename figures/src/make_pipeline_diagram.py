"""The quantum-kernel EEG pipeline diagram, built from scratch as a .drawio file.

    python figures/src/make_architecture_parts.py     # the real-data panels first
    python figures/src/make_pipeline_diagram.py       # then this

Output: figures/experiments/architecture_parts/quantum_kernel_eeg_pipeline_final.drawio.
Export it to PDF with diagrams.net (File, Export as, PDF, Crop) or the draw.io
command line: draw.io --export --format pdf --crop <file>.

It follows the corresponding author's diagram (colours, fonts, reading order)
and the code, not a schematic idea of it:

  front end   channel selection, band-pass, epoching, OAS covariance C
              (src/qeeg/data.py picks channels, filters, then epochs)
  lane 1      quantum density-matrix kernels: reference frame W = M^(-1/2)
              (sensor frame W = I), rho = WCW / tr WCW, five kernels
  lane 2      quantum circuit kernels: tangent space, standardise, PCA to 4,
              angles in [0, pi], feature map U(x), state overlap; IQP, ring
              CNOT and the no-entangler ablation
  lane 3      the classical twin: pyriemann's Riemannian kernel centred on the
              same Frechet mean, tr[log(WC_iW) log(WC_jW)], with the
              log-Euclidean kernel reported alongside
  merge       one Gram matrix, projected to the PSD cone when needed, one SVM
              on the precomputed kernel, nested cross-validation

The SWAP test is drawn as a callout on the Hilbert-Schmidt overlap kernel, not
as a pipeline stage: it is how tr(rho sigma) would be estimated on hardware,
studied in the finite-shot analysis, and no classifier in the benchmark runs it.
"""
from __future__ import annotations

import base64
import html
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARTS = HERE.parent / "experiments" / "architecture_parts"
OUT = PARTS / "quantum_kernel_eeg_pipeline_final.drawio"

PAGE_W, PAGE_H = 1330, 900

INK, MUTED, LINK = "#263238", "#546e7a", "#1f5fa8"
BLUE, ORANGE = "#2a78d6", "#eb6834"

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
SWAPBOX = ("rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#ffffff;"
           "strokeColor=#9f8fe0;dashed=1;")
SVMBOX = ("rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#ffffff;"
          "strokeColor=#9f8fe0;")
PRED = (f"rounded=1;arcSize=12;whiteSpace=wrap;html=1;fillColor=#fff2cc;"
        f"strokeColor=#d6b656;fontFamily=Helvetica;fontColor={INK};fontSize=13;")
NOTE = (f"rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#f7f7f7;"
        f"strokeColor=#cfcfcf;fontFamily=Helvetica;fontColor={INK};fontSize=10;"
        f"align=left;verticalAlign=top;spacingLeft=8;spacingTop=4;")
IMAGE = "shape=image;imageAspect=0;aspect=fixed;verticalLabelPosition=bottom;verticalAlign=top;"
EDGE = (f"edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;html=1;"
        f"endArrow=block;endFill=1;strokeColor={INK};strokeWidth=1.3;")
CALLOUT = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;dashed=1;"
           "endArrow=open;endFill=0;strokeColor=#7e57c2;strokeWidth=1.1;")

root_el = ET.Element("mxfile", host="app.diagrams.net")
diagram = ET.SubElement(root_el, "diagram", name="Pipeline", id="qeeg-pipeline")
model = ET.SubElement(diagram, "mxGraphModel", dx="0", dy="0", grid="0",
                      gridSize="10", guides="1", tooltips="1", connect="1",
                      arrows="1", fold="1", page="1", pageScale="1",
                      pageWidth=str(PAGE_W), pageHeight=str(PAGE_H),
                      background="#FFFFFF", math="1", shadow="0")
top = ET.SubElement(model, "root")
ET.SubElement(top, "mxCell", id="0")
ET.SubElement(top, "mxCell", id="1", parent="0")


def png(name: str) -> str:
    data = base64.b64encode((PARTS / f"{name}.png").read_bytes()).decode()
    return "data:image/png," + data


def v(cid, value, style, x, y, w, h):
    c = ET.SubElement(top, "mxCell", id=cid, value=value, style=style,
                      vertex="1", parent="1")
    ET.SubElement(c, "mxGeometry", x=f"{x:.2f}", y=f"{y:.2f}", width=f"{w:.2f}",
                  height=f"{h:.2f}", **{"as": "geometry"})
    return (x, y, w, h)


def img(cid, name, x, y, w, h):
    return v(cid, "", IMAGE + "image=" + png(name) + ";", x, y, w, h)


def e(cid, src, tgt, ports="", points=(), style=EDGE, value=""):
    c = ET.SubElement(top, "mxCell", id=cid, value=value, style=style + ports,
                      edge="1", parent="1", source=src, target=tgt)
    g = ET.SubElement(c, "mxGeometry", relative="1", **{"as": "geometry"})
    if points:
        arr = ET.SubElement(g, "Array", **{"as": "points"})
        for x, y in points:
            ET.SubElement(arr, "mxPoint", x=f"{x:.2f}", y=f"{y:.2f}")


def port(exit_=None, entry=None):
    s = ""
    if exit_:
        s += f"exitX={exit_[0]};exitY={exit_[1]};exitDx=0;exitDy=0;"
    if entry:
        s += f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
    return s


def title(cid, text, x, y, w, size=13, align="left", color=INK):
    v(cid, f"<b>{text}</b>", TEXT + f"fontSize={size};align={align};fontColor={color};",
      x, y, w, 20)


# =========================================================== front end
# EEG: channel labels centred on the traces. The trace image puts trace i at
# (m + i) / (7 + 2m) of its height, m = 0.6, and has width / height = 2.4 / 2.6.
title("t-eeg", "Motor-imagery EEG", 20, 20, 240, align="left")
m, pitch, c0 = 0.6, 24.0, 64.0
tr_h = pitch * 7 * (7 + 2 * m) / 7
tr_w = tr_h * 2.4 / 2.6
tr_y = c0 - tr_h * m / (7 + 2 * m)
for i, ch in enumerate(["FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CP4"]):
    v(f"ch-{ch}", ch, TEXT + "fontSize=11;align=right;", 20, c0 + i * pitch - 8, 34, 16)
img("eeg", "1_eeg_traces", 58, tr_y, tr_w, tr_h)
v("x-label", r"\(X \in \mathbb{R}^{8 \times T}\)", TEXT + "fontSize=13;",
  58, tr_y + tr_h + 4, tr_w, 22)

mid = c0 + 3.5 * pitch                       # vertical centre of the front end
pre_x0 = 58 + tr_w + 34
for k, (cid, label) in enumerate([("chsel", "Channel selection"),
                                  ("bpf", "Band-pass filter"),
                                  ("epoch", "Epoching")]):
    v(cid, label, PRE + "horizontal=0;", pre_x0 + k * 72, mid - 66, 42, 132)
e("f1", "eeg", "chsel", port((1, 0.5), (0, 0.5)))
e("f2", "chsel", "bpf", port((1, 0.5), (0, 0.5)))
e("f3", "bpf", "epoch", port((1, 0.5), (0, 0.5)))

cov_x = pre_x0 + 2 * 72 + 42 + 40
title("t-cov", "Covariance matrix", cov_x - 20, mid - 86, 150, size=13, align="center")
img("cov", "2_covariance", cov_x, mid - 55, 110, 110)
e("f4", "epoch", "cov", port((1, 0.5), (0, 0.5)))
v("cov-eq", r"\(C = (1-\alpha)\,S + \alpha\,\tfrac{\operatorname{tr} S}{8}\,I\)"
            r"<br>\(S\): sample covariance, \(\alpha\): OAS shrinkage"
            r"<br>\(C \in \mathbb{S}^{8}_{++}\)",
  TEXT + "fontSize=12;align=left;", cov_x + 122, mid - 34, 250, 68)
cov_bottom = (cov_x + 55, mid + 55)

# Side note: the rest of the benchmark, on the same protocol.
v("others",
  "<b>Also evaluated under the same protocol</b> (not drawn)<br>"
  "<b>Classical baselines:</b> CSP&nbsp;+&nbsp;LDA, tangent space&nbsp;+&nbsp;LR, "
  "tangent space&nbsp;+&nbsp;RBF SVM, MDM, log-variance&nbsp;+&nbsp;LDA, FBCSP<br>"
  "<b>Dimension-matched controls:</b> linear and RBF SVMs on the same four PCA "
  "features the circuits receive",
  NOTE, 940, 40, 370, 96)

# ================================================================ lanes
LX, LW = 60, 860          # lane containers
BUS_X = 30                # the fork bus, left of the lanes
FORK_Y = 300              # the trunk turns here, under the front end

# ---- lane 1: quantum density-matrix kernels ------------------------------
l1y, l1h = 322, 238
v("lane1", "", LANE, LX, l1y, LW, l1h)
title("t-lane1", "Quantum density-matrix kernels", LX + 12, l1y + 6, 400)

v("frame", r"<b>Reference frame</b><br>\(W = M^{-1/2}\)<br>\(\tilde C = W C W\)",
  FRAME, 80, l1y + 56, 124, 76)
v("frame-note", r"\(M\): Fréchet mean of the training covariances<br>"
                r"sensor frame: \(W = I\)",
  TEXT + f"fontSize=9;fontColor={MUTED};", 70, l1y + 136, 144, 44)

title("t-rho", "Density matrix", 246, l1y + 30, 120, size=12, align="center")
img("rho", "3_density_matrix", 256, l1y + 52, 100, 100)
v("rho-eq", r"\(\rho = \tilde C / \operatorname{tr}\tilde C\)<br>"
            r"\(\rho \succ 0,\ \operatorname{tr}\rho = 1\)",
  TEXT + "fontSize=12;", 226, l1y + 156, 160, 44)
e("k1", "frame", "rho", port((1, 0.5), (0, 0.5)))

g1x, g1y, g1w = 410, l1y + 34, 200
v("kd-box", "", GROUP, g1x, g1y, g1w, 196)
v("kd-title", r"\(k(\rho_i, \rho_j)\), one per run", TEXT + "fontSize=12;",
  g1x, g1y + 4, g1w, 22)
dens = ["Hilbert–Schmidt overlap", "Fidelity", "HS-RBF", "Bures-RBF", "QRE-RBF"]
for i, name in enumerate(dens):
    v(f"kd-{i}", name, ITEM, g1x + 14, g1y + 32 + i * 32, g1w - 28, 26)
e("k2", "rho", "kd-box", port((1, 0.5), (0, 0.44)))

# SWAP test callout on the overlap kernel.
# It stops short of the lane's lower edge so the kernel output can pass under it.
sx, sy, sw_, sh_ = 632, l1y + 44, 278, 150
v("swap-box", "", SWAPBOX, sx, sy, sw_, sh_)
v("swap-title", "<b>SWAP test</b> for the overlap",
  TEXT + "fontSize=12;", sx, sy + 4, sw_, 20)
swap_h = 112
swap_w = swap_h * 1988 / 2243
img("swap", "7_swap_test", sx + 8, sy + 28, swap_w, swap_h)
v("swap-eq", r"\(P(a{=}0) = \tfrac{1}{2}\,(1 + \operatorname{tr}\rho\sigma)\)"
             r"<br><span style='font-size:10px;color:#546e7a'>how the overlap would be "
             r"measured on hardware; simulated in the finite-shot analysis</span>",
  TEXT + "fontSize=11;", sx + swap_w + 14, sy + 34, sw_ - swap_w - 20, 104)
e("callout", "kd-0", "swap-box", port((1, 0.5), (0, 0.3)),
  points=[(618, g1y + 32 + 13), (618, sy + 0.3 * sh_)], style=CALLOUT)

# ---- lane 2: quantum circuit kernels ------------------------------------
l2y, l2h = 580, 172
v("lane2", "", LANE, LX, l2y, LW, l2h)
title("t-lane2", "Quantum circuit kernels", LX + 12, l2y + 6, 400)
v("t-lane2-sub", "4 qubits; IQP, ring CNOT, and the entanglement ablation",
  TEXT + f"fontSize=10;fontColor={MUTED};align=left;", LX + 190, l2y + 7, 400, 18)

bx = 80
for i, (cid, label) in enumerate([("ts", "Tangent space"),
                                  ("pca", "Standardise, PCA to 4"),
                                  ("ang", r"Angles \(x \in [0,\pi]^4\)")]):
    v(cid, label, PRE + "fontSize=11;", bx, l2y + 38 + i * 42, 140, 26)
e("c1", "ts", "pca", port((0.5, 1), (0.5, 0)))
e("c2", "pca", "ang", port((0.5, 1), (0.5, 0)))

cw = 290
ch_ = cw * 1213 / 3283
img("circ", "6_circuit_iqp", 250, l2y + 30, cw, ch_)
v("circ-note", r"feature map \(U(x)\), one of two layers shown; "
               r"\(z_{i,i+1} = (\pi - x_i)(\pi - x_{i+1})\)",
  TEXT + f"fontSize=10;fontColor={MUTED};", 240, l2y + 32 + ch_, 310, 30)
e("c3", "ang", "circ", port((1, 0.5), (0, 0.5)),
  points=[(236, l2y + 38 + 2 * 42 + 13), (236, l2y + 30 + ch_ / 2)])

g2x, g2y, g2w = 580, l2y + 30, 220
v("kc-box", "", GROUP, g2x, g2y, g2w, 130)
v("kc-title", r"\(k(x_i, x_j) = |\langle \psi(x_j) \mid \psi(x_i) \rangle|^2\)",
  TEXT + "fontSize=12;", g2x, g2y + 4, g2w, 24)
for i, name in enumerate(["IQP feature map", "Ring CNOT",
                          "No entanglers (ablation)"]):
    v(f"kc-{i}", name, ITEM, g2x + 14, g2y + 34 + i * 31, g2w - 28, 25)
e("c4", "circ", "kc-box", port((1, 0.5), (0, 0.5)))

# ---- lane 3: the classical twin ------------------------------------------
l3y, l3h = 772, 112
v("lane3", "", TWIN_LANE, LX, l3y, LW, l3h)
title("t-lane3", "Classical twin (control)", LX + 12, l3y + 6, 220)
v("t-lane3-sub", "same covariances, same frame, same SVM and tuning budget: "
                 "only the metric differs",
  TEXT + f"fontSize=10;fontColor={MUTED};align=left;", LX + 180, l3y + 7, 460, 18)
v("twin-frame", r"Same reference frame<br>\(\tilde C = W C W\)",
  FRAME + "strokeColor=#e0b4a0;fontSize=11;", 80, l3y + 42, 140, 48)
g3x, g3y, g3w = 410, l3y + 34, 300
v("kt-box", "", GROUP + "fillColor=#fffaf8;strokeColor=#ecc9b8;", g3x, g3y, g3w, 70)
v("kt-0", r"Riemannian: \(k(C_i, C_j) = \operatorname{tr}[\log\tilde C_i \,\log\tilde C_j]\)",
  TWIN_ITEM, g3x + 12, g3y + 8, g3w - 24, 26)
v("kt-1", "log-Euclidean, reported alongside", TWIN_ITEM + "fontColor=#6d4c41;",
  g3x + 12, g3y + 38, g3w - 24, 24)
e("t1", "twin-frame", "kt-box", port((1, 0.5), (0, 0.5)))

# ---- fork: covariance to the three lanes ---------------------------------
for cid, tgt, ty in (("fork1", "frame", l1y + 56 + 38),
                     ("fork2", "ts", l2y + 38 + 13),
                     ("fork3", "twin-frame", l3y + 42 + 24)):
    e(cid, "cov", tgt, port((0.5, 1), (0, 0.5)),
      points=[(cov_bottom[0], FORK_Y), (BUS_X, FORK_Y), (BUS_X, ty)])
v("fork-label", r"\(C\)", TEXT + f"fontSize=13;fontColor={LINK};align=left;",
  cov_bottom[0] + 6, cov_bottom[1] + 16, 24, 20)

# ================================================================ merge
kx, ky, ks = 985, 468, 128
title("t-K", "Kernel matrix", kx - 10, ky - 26, ks + 20, size=13, align="center")
img("K", "4_kernel_matrix", kx, ky, ks, ks)
v("K-eq", r"\(K_{ij} = k(\cdot_i, \cdot_j),\ K = K^{\top}\)<br>"
          r"projected to \(K \succeq 0\) when needed",
  TEXT + "fontSize=10;", kx - 18, ky + ks + 4, ks + 36, 40)
MERGE_X = 948
# Lane 1 leaves its kernel box low (exitY 0.9) and runs under the SWAP callout.
for cid, src, ey, sy_ in (("m1", "kd-box", 0.9, g1y + 0.9 * 196),
                          ("m2", "kc-box", 0.5, g2y + 65),
                          ("m3", "kt-box", 0.5, g3y + 35)):
    e(cid, src, "K", port((1, ey), (0, 0.5)),
      points=[(MERGE_X, sy_), (MERGE_X, ky + ks / 2)])

svm_x, svm_y, svm_w, svm_h = 1150, 430, 170, 204
v("svm", "", SVMBOX, svm_x, svm_y, svm_w, svm_h)
v("svm-title", "<b>SVM</b> (precomputed kernel)", TEXT + "fontSize=12;",
  svm_x, svm_y + 6, svm_w, 20)
svm_img_w = svm_w - 24
img("svm-img", "5_svm", svm_x + 12, svm_y + 34, svm_img_w, svm_img_w * 1132 / 1203)
v("svm-note", "same regularisation grid, same inner search for every kernel",
  TEXT + f"fontSize=9;fontColor={MUTED};", svm_x + 6, svm_y + svm_h - 34, svm_w - 12, 28)
e("s1", "K", "svm", port((1, 0.5), (0, 0.5)))

# The class legend lives inside the box, so the arrow below it crosses nothing.
v("pred",
  f"<b>Predicted class</b><br>"
  f"<span style='font-size:11px'><font color='{BLUE}'>&#9679;</font> left hand &nbsp; "
  f"<font color='{ORANGE}'>&#9675;</font> right hand</span><br>"
  f"<span style='font-size:9px;color:{MUTED}'>four-class IV-2a adds feet, tongue</span>",
  PRED, svm_x - 5, 668, svm_w + 10, 66)
e("s2", "svm", "pred", port((0.5, 1), (0.5, 0)))

v("eval",
  "<b>Evaluation</b><br>"
  "nested CV: 5 folds × 3 repeats outer,<br>4-fold inner search<br>"
  "paired Wilcoxon, Holm within families<br>"
  "TOST equivalence against the twin",
  NOTE + "fontSize=10;", svm_x - 10, 770, svm_w + 20, 100)
e("s3", "pred", "eval", port((0.5, 1), (0.5, 0)))

ET.indent(root_el, space="  ")
ET.ElementTree(root_el).write(OUT, encoding="utf-8", xml_declaration=False)
print("wrote", OUT)
