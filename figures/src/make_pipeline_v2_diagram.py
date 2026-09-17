"""Make quantum_kernel_eeg_pipeline_v2.drawio from the author's diagram.

Corrections: aligned EEG traces, OAS covariance, reference-frame note, page
fitted to the drawing. Addition: a bottom row with the circuit-kernel branch
and the SWAP test, both as verified Qiskit renders.
"""
import base64
import copy
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "experiments" / "architecture_parts"
SRC = D / "quantum_kernel_eeg_pipeline.drawio (1).xml"
DST = D / "quantum_kernel_eeg_pipeline_v2.drawio"

DX, DY = -2393.0, -65.0          # move the drawing onto the page
PAGE_W, PAGE_H = 1010, 850

tree = ET.parse(SRC)
root = tree.getroot()
model = root.find(".//mxGraphModel")
top = model.find("root")
cells = {c.get("id"): c for c in top.iter("mxCell")}


def png(name):
    return "data:image/png," + base64.b64encode((D / f"{name}.png").read_bytes()).decode()


def geo(cell):
    return cell.find("mxGeometry")


# ---- 1  traces aligned with the channel labels -----------------------------
# Label centres inside the group: FC3 at 11.62, CP4 at 179.39. The trace image
# puts trace i at (m + i) / (7 + 2m) of its height, m = 0.6.
m = 0.6
first, last = 4.982608695652175 + 13.286956521739132 / 2, 172.74704347826088 + 13.286956521739132 / 2
H = (last - first) * (7 + 2 * m) / 7
Y = first - H * m / (7 + 2 * m)
tr = cells["57"]
g = geo(tr)
g.set("y", f"{Y:.3f}")
g.set("height", f"{H:.3f}")
style = tr.get("style")
head, _, rest = style.partition("image=")
tr.set("style", head + "image=" + png("1_eeg_traces") + ";")

# ---- 2  OAS covariance -----------------------------------------------------
cov = cells["11"]
cov.set("value", r"\(C = (1-\alpha)\,S + \alpha\,\tfrac{\operatorname{tr} S}{8}\,I\)"
                 r"<br>\(S\): sample covariance"
                 r"<br>\(\alpha\): OAS shrinkage"
                 r"<br>\(C \in \mathbb{S}^{8}_{++}\)")
g = geo(cov)
# Narrow, so the circuit branch can leave the covariance matrix down its left.
g.set("x", "2798"); g.set("width", "186"); g.set("y", "233"); g.set("height", "90")
cov.set("style", cov.get("style").replace("fontSize=14", "fontSize=12"))

# ---- geometry shift and page ------------------------------------------------
for c in top.iter("mxCell"):
    if c.get("parent") != "1":
        continue
    g = geo(c)
    if g is None:
        continue
    if c.get("vertex"):
        g.set("x", f"{float(g.get('x', 0)) + DX:.3f}")
        g.set("y", f"{float(g.get('y', 0)) + DY:.3f}")
    for p in g.iter("mxPoint"):
        if p.get("as") == "offset":
            continue
        if p.get("x") is not None:
            p.set("x", f"{float(p.get('x')) + DX:.3f}")
        if p.get("y") is not None:
            p.set("y", f"{float(p.get('y')) + DY:.3f}")
# Kernel-matrix equations a little lower, so the circuit branch can reach the
# matrix from below; the density-to-kernel edge a little lower, so the circuit
# branch's lane above it stays clear of it.
geo(cells["36"]).set("y", "478")
for p in geo(cells["14"]).iter("mxPoint"):
    if p.get("y") is not None and abs(float(p.get("y")) - 270.67) < 0.1:
        p.set("y", "282")
model.set("pageWidth", str(PAGE_W))
model.set("pageHeight", str(PAGE_H))
model.set("dx", "0")
model.set("dy", "0")

# ---- new cells (coordinates below are already on the page) -----------------
TEXT = ("text;html=1;whiteSpace=wrap;align=center;verticalAlign=middle;"
        "fontFamily=Helvetica;fontColor=#263238;fontStyle=0;")
BOX = ("rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#f5fbff;"
       "strokeColor=#9cc8ea;fontFamily=Helvetica;fontColor=#263238;")
CONTAINER = ("rounded=1;arcSize=4;whiteSpace=wrap;html=1;fillColor=none;"
             "strokeColor=#9fa8da;dashed=1;")
IMAGE = ("shape=image;verticalLabelPosition=bottom;verticalAlign=top;"
         "imageAspect=0;aspect=fixed;")
EDGE = ("edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;html=1;"
        "endArrow=block;endFill=1;strokeColor=#263238;strokeWidth=1.3;")
new_ids = []


def vertex(cid, value, style, x, y, w, h):
    c = ET.SubElement(top, "mxCell", id=cid, value=value, style=style,
                      vertex="1", parent="1")
    ET.SubElement(c, "mxGeometry", x=f"{x}", y=f"{y}", width=f"{w}",
                  height=f"{h}", **{"as": "geometry"})
    new_ids.append(cid)


def edge(cid, src, tgt):
    c = ET.SubElement(top, "mxCell", id=cid, value="", style=EDGE, edge="1",
                      parent="1", source=src, target=tgt)
    ET.SubElement(c, "mxGeometry", relative="1", **{"as": "geometry"})


def routed(cid, src, tgt, ports, points):
    """An edge with fixed exit and entry ports and explicit waypoints."""
    c = ET.SubElement(top, "mxCell", id=cid, value="", style=EDGE + ports,
                      edge="1", parent="1", source=src, target=tgt)
    g = ET.SubElement(c, "mxGeometry", relative="1", **{"as": "geometry"})
    arr = ET.SubElement(g, "Array", **{"as": "points"})
    for x, y in points:
        ET.SubElement(arr, "mxPoint", x=f"{x}", y=f"{y}")


# Reference-frame note, under the Reference frame box (now x 627.97, y 68.33).
vertex("qk-refnote",
       r"\(M\): Riemannian mean of training covariances<br>sensor frame: \(W = I\)",
       TEXT + "fontSize=9;fontColor=#546e7a;", 580, 30, 180, 34)

# Row 3a: circuit-kernel branch.
x0, y0 = 20, 575
vertex("qk-circ-box", "", CONTAINER, x0, y0, 590, 250)
vertex("qk-circ-title", "<b>Circuit kernels on reduced features</b>",
       TEXT + "fontSize=13;", x0, y0 + 4, 590, 22)
vertex("qk-circ-sub", "(IQP, ring CNOT, and the ablation with entanglers removed; 4 qubits)",
       TEXT + "fontSize=10;fontColor=#546e7a;", x0, y0 + 24, 590, 16)
vertex("qk-ts", "Tangent space", BOX + "fontSize=11;", x0 + 12, y0 + 62, 128, 28)
vertex("qk-pca", "Standardise, PCA to 4", BOX + "fontSize=11;", x0 + 12, y0 + 106, 128, 28)
vertex("qk-ang", r"Angles \(x \in [0,\pi]^4\)", BOX + "fontSize=11;", x0 + 12, y0 + 150, 128, 28)
edge("qk-e1", "qk-ts", "qk-pca")
edge("qk-e2", "qk-pca", "qk-ang")
cw = 285.0
ch = cw * 1213 / 3283
vertex("qk-circ-img", "", IMAGE + "image=" + png("6_circuit_iqp") + ";",
       x0 + 160, y0 + 52, round(cw, 2), round(ch, 2))
edge("qk-e3", "qk-ang", "qk-circ-img")
vertex("qk-circ-note",
       r"\(U(x)\), one of two layers shown; "
       r"\(z_{i,i+1} = (\pi - x_i)(\pi - x_{i+1})\)"
       r"<br>ring CNOT: a CNOT ring replaces the couplings; ablation: couplings removed",
       TEXT + "fontSize=10;", x0 + 150, y0 + 58 + ch, 432, 34)
vertex("qk-circ-k",
       r"\(k(x,x') =\)<br>\(|\langle \psi(x') \mid \psi(x) \rangle|^2\)",
       TEXT + "fontSize=12;", x0 + 452, y0 + 72, 134, 76)
edge("qk-e4", "qk-circ-img", "qk-circ-k")

# Row 3b: SWAP test.
x1 = 630
vertex("qk-swap-box", "", CONTAINER, x1, y0, 343, 250)
vertex("qk-swap-title", "<b>SWAP test for the overlap kernel</b>",
       TEXT + "fontSize=13;", x1, y0 + 4, 343, 22)
vertex("qk-swap-sub",
       r"(how \(\operatorname{tr}\rho\sigma\) is measured on hardware; simulated in the finite-shot analysis)",
       TEXT + "fontSize=10;fontColor=#546e7a;", x1 + 10, y0 + 24, 323, 28)
sh = 188.0
sw = sh * 1988 / 2243
vertex("qk-swap-img", "", IMAGE + "image=" + png("7_swap_test") + ";",
       x1 + 14, y0 + 56, round(sw, 2), sh)
vertex("qk-swap-eq",
       r"\(P(a{=}0) = \tfrac{1}{2}\,(1 + \operatorname{tr}\rho\sigma)\)"
       r"<br><br>\(S\) shots:<br>\(\operatorname{Var}\hat k = (1-k^2)/S\)",
       TEXT + "fontSize=11;", x1 + 190, y0 + 80, 148, 110)

# ---- where the circuits join the graph --------------------------------------
# Ids of the author's cells, positions after the shift:
#   covariance image  HYGETC2J5IyEfWyGQ5A--68  x 443.26..552.61, y 53.23..162.58
#   density image     HYGETC2J5IyEfWyGQ5A--74  x 827..933.17,    y 49.25..155.42
#   kernel matrix     HYGETC2J5IyEfWyGQ5A--77  x 374..503.22,    y 322..451.22
COV, DEN, KMAT = "HYGETC2J5IyEfWyGQ5A--68", "HYGETC2J5IyEfWyGQ5A--74", "HYGETC2J5IyEfWyGQ5A--77"
LABEL = TEXT + "fontSize=11;fontColor=#1f5fa8;"

# Circuit branch in: the covariance matrix, down its left side, along a lane
# above the density-to-kernel edge, down the page margin, into Tangent space.
# The Epoching box, rotated, occupies x 371..411, y 43..173; the formula text
# under the covariance matrix starts at about x 432. The lane runs at x 420.
routed("qk-in-circ", COV, "qk-ts",
       "exitX=0;exitY=0.976;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;",
       [(420, 160), (420, 256), (8, 256), (8, 651)])
vertex("qk-lab-c", r"\(C\)", LABEL + "align=right;", 390, 188, 26, 18)

# Circuit branch out: the kernel value, up into the kernel matrix from below.
routed("qk-out-circ", "qk-circ-k", KMAT,
       "exitX=0.5;exitY=0;exitDx=0;exitDy=0;entryX=0.9;entryY=1;entryDx=0;entryDy=0;",
       [(539, 466), (490.3, 466)])
vertex("qk-lab-kc", r"\(K_{ij} = k(x_i, x_j)\)", LABEL + "align=right;", 400, 556, 134, 18)

# SWAP test in: two density matrices, from the density matrix down the right
# margin.
routed("qk-in-swap", DEN, "qk-swap-box",
       "exitX=1;exitY=0.1;exitDx=0;exitDy=0;entryX=1;entryY=0.5;entryDx=0;entryDy=0;",
       [(990, 59.9), (990, 700)])
vertex("qk-lab-rho", r"\(\rho_i,\ \rho_j\)", LABEL + "align=right;", 930, 292, 56, 18)

# SWAP test out: the overlap, up into the kernel matrix from its right side.
routed("qk-out-swap", "qk-swap-box", KMAT,
       "exitX=0.204;exitY=0;exitDx=0;exitDy=0;entryX=1;entryY=0.913;entryDx=0;entryDy=0;",
       [(700, 555), (560, 555), (560, 440)])
vertex("qk-lab-ks", r"\(K_{ij} = \operatorname{tr}\rho_i\rho_j\)", LABEL + "align=left;", 566, 536, 130, 18)

ET.indent(tree, space="  ")
tree.write(DST, encoding="utf-8", xml_declaration=False)
print("wrote", DST, "new cells:", len(new_ids))
