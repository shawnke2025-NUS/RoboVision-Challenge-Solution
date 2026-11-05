"""Publication-style flowchart of the v3 solver (vector PDF + 400 dpi PNG)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from PIL import Image

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

BASE = r"C:\Users\shawn\Desktop\robo-vision-challenge-data-hack-3-0"
ROOT = os.path.join(BASE, "Challenge_Dataset")

INK, GREY, LIGHT = "#1a1a1a", "#6f6f6f", "#efefef"
ACC, FALL = "#2f5d8a", "#7a4b8a"

fig = plt.figure(figsize=(7.16, 3.6), dpi=400)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 120); ax.set_ylim(13.0, 60.0); ax.axis("off")


def box(x, y, w, h, text, fc="white", ec=INK, lw=0.7, fs=6.6, bold=False, tc=INK, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.3",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.32, zorder=4)


def group(x, y, w, h, label, color=GREY, pad=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                fc="none", ec=color, lw=0.6, ls=(0, (3, 2)), zorder=1))
    ax.text(x + pad, y + h - 1.5, label, ha="left", va="center", fontsize=6.7,
            color=color, style="italic", zorder=2)


def arrow(p0, p1, color=INK, lw=0.7, rad=0.0, ls="-", ms=5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw,
                                 color=color, ls=ls, shrinkA=1.5, shrinkB=1.5,
                                 connectionstyle=f"arc3,rad={rad}", zorder=2))


# ------------------------------------------------ (a) input
group(1.0, 44.2, 16.0, 15.3, "(a) input")
box(2.4, 45.2, 13.2, 4.6, "camera\n$K,\\,[R\\,|\\,t]$", fs=6.4)
ax.add_patch(FancyBboxPatch((2.4, 51.4), 13.2, 5.2, boxstyle="round,pad=0,rounding_size=0.3",
                            fc="white", ec=INK, lw=0.7, zorder=3))
try:
    tr = pd.read_csv(os.path.join(ROOT, "train.csv"))
    img = np.asarray(Image.open(os.path.join(ROOT, tr.image_path[0].replace("/", os.sep))).convert("RGB"))
    axi = ax.inset_axes([2.45, 51.45, 13.1, 5.1], transform=ax.transData, zorder=4)
    axi.imshow(img, aspect="auto"); axi.set_xticks([]); axi.set_yticks([]); axi.tick_params(length=0)
    for s in axi.spines.values():
        s.set_linewidth(0.4); s.set_color(INK)
except Exception as e:
    print("thumbnail skipped:", e)

# ------------------------------------------------ (b) parsing
BX, BWd = 19.0, 23.0
group(BX, 37.0, BWd, 21.5, "(b) scene parsing")
bw, bh = BWd - 2.4, 3.9
bxs = BX + 1.2
ys = [51.0, 46.6, 42.2, 37.8]
for y, t in zip(ys, ["background mask\n3 flat colours",
                     "connected\ncomponents",
                     "split by colour\nalbedo fit",
                     "filter + merge\n8--10 objects"]):
    box(bxs, y, bw, bh, t)
for i in range(3):
    arrow((bxs + bw / 2, ys[i]), (bxs + bw / 2, ys[i + 1] + bh))
arrow((17.0, 53.8), (bxs, 53.2), rad=-0.05)
arrow((17.0, 47.7), (bxs, 45.0), rad=0.08, ls=(0, (2.2, 1.8)))

# ------------------------------------------------ (c) rule engine
CX, CWd = 44.0, 23.0
group(CX, 37.0, CWd, 21.5, "(c) rule engine")
cxs = CX + 1.2
for y, t in zip(ys, ["pair colours\n$d<0.03$",
                     "block vs bowl\n$h/w$ vs $\\theta$",
                     "order by\nintensity",
                     "assign slots\n$0$--$2$ / $3$--$5$"]):
    box(cxs, y, bw, bh, t)
for i in range(3):
    arrow((cxs + bw / 2, ys[i]), (cxs + bw / 2, ys[i + 1] + bh))
arrow((bxs + bw, ys[0] + bh / 2), (cxs, ys[0] + bh / 2))
arrow((bxs + bw, ys[3] + bh / 2), (cxs, ys[3] + bh / 2))

# ------------------------------------------------ (d) geometry
DX, DWd = 69.0, 23.0
group(DX, 37.0, DWd, 21.5, "(d) geometry")
dxs = DX + 1.2
box(dxs, 50.2, bw, 5.4, "back-project\nthe base point", fs=6.5)
box(dxs, 42.4, bw, 5.4, "calibrated affine\nscale $1.29$", fs=6.5)
box(dxs, 37.8, bw, 3.5, "6 $\\times$ pick $(x,y)$", fc=LIGHT, bold=True, fs=6.8)
arrow((dxs + bw / 2, 50.2), (dxs + bw / 2, 47.8))
arrow((dxs + bw / 2, 42.4), (dxs + bw / 2, 41.3))
arrow((cxs + bw, ys[0] + bh / 2), (dxs, 52.4), rad=-0.04)
arrow((cxs + bw, ys[3] + bh / 2), (dxs, 44.6), rad=-0.06)

# ------------------------------------------------ right column: place + submission
box(94.0, 47.2, 25.0, 6.2, "place lookup (exact)\n$x_{\\mathrm{pl}},y_{\\mathrm{pl}}$ per (task, step)",
    ec=ACC, lw=0.8, fs=6.3)
arrow((dxs + bw, 51.5), (94.0, 51.0), color=ACC, rad=-0.03)

box(94.0, 31.0, 25.0, 9.6,
    "submission\n$\\left(x_{\\mathrm{pick}},y_{\\mathrm{pick}}\\right)$  model\n"
    "$\\left(x_{\\mathrm{place}},y_{\\mathrm{place}}\\right)$  lookup",
    fc=LIGHT, fs=6.2)
arrow((106.5, 47.2), (106.5, 40.6), color=ACC)
arrow((dxs + bw, 39.0), (99.0, 40.6), rad=0.10)

# ------------------------------------------------ (e) fallback
group(19.0, 19.5, 60.0, 13.5, "(e) fallback, $\\sim$5 % of frames", color=FALL)
box(20.6, 21.4, 25.0, 8.4, "EfficientNet-B0\nImageNet weights, BEV\nmirror augmentation", ec=FALL, fs=6.3)
box(50.0, 21.4, 27.6, 8.4, "heat map, stride 2\nsoft-argmax $\\rightarrow$ metres\nfocal $+\\,L_1$", ec=FALL, fs=6.3)
arrow((45.6, 25.6), (49.9, 25.6), color=FALL)
arrow((cxs + bw / 2, ys[3]), (cxs + bw / 2, 33.2), color=FALL, ls=(0, (2.2, 1.8)))
ax.text(cxs + bw / 2 + 1.6, 34.6, "3 pairs not found", fontsize=5.8, color=FALL, style="italic")
arrow((79.0, 25.6), (93.8, 33.0), color=FALL, rad=-0.12)

# ------------------------------------------------ (f) scene structure
FX, FY, FW, FH = 1.0, 15.0, 16.0, 22.5
group(FX, FY, FW, FH, "(f) scene structure")
ax.text(FX + FW / 2, FY + 18.3, "10 objects", ha="center", va="center", fontsize=6.3)
COLS = [4.4, 9.0, 13.6]
for cx in COLS:
    ax.add_patch(Rectangle((FX + cx - 1.05, FY + 13.0), 2.1, 2.1,
                           fc="#dfe7f1", ec=ACC, lw=0.7, zorder=3))
    ax.add_patch(Circle((FX + cx, FY + 10.0), 1.05, fc="#f7f7f7", ec=ACC, lw=0.7, zorder=3))
ax.text(FX + FW / 2, FY + 16.4, "blocks $\\rightarrow$ 3--5", ha="center", va="center",
        fontsize=5.4, color=ACC)
ax.text(FX + FW / 2, FY + 7.5, "3 colour pairs", ha="center", va="center", fontsize=5.8, color=ACC)
ax.text(FX + FW / 2, FY + 6.2, "= targets", ha="center", va="center", fontsize=5.8, color=ACC)
for cx in COLS:
    ax.add_patch(Circle((FX + cx, FY + 2.8), 0.95, fc="white", ec=GREY, lw=0.6,
                        ls=(0, (1.4, 1.4)), zorder=3))
ax.text(FX + FW / 2, FY + 1.1, "4 unique = distractors", ha="center", va="center",
        fontsize=5.8, color=GREY)

# ------------------------------------------------ caption
ax.text(60, 14.6,
        "Fig. 1.  Analytic scene solver with a learned fallback.  (b)--(d) are deterministic, $\\sim$0.14 s per frame; the network in (e) is\n"
        "consulted only when a colour pair is missing.  Actions 0--2 are bowls, 3--5 blocks.  Public leaderboard: 0.877 (1st of 4).",
        ha="center", va="center", fontsize=5.8, color=GREY, style="italic", linespacing=1.5)

out_pdf = os.path.join(BASE, "algorithm_flowchart.pdf")
out_png = os.path.join(BASE, "algorithm_flowchart.png")
fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
fig.savefig(out_png, dpi=400, bbox_inches="tight", pad_inches=0.02)
print("wrote", out_pdf, "and", out_png)
