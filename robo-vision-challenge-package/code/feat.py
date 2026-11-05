"""Shared feature extraction for the RoboVision challenge (numpy only)."""
import numpy as np, os, json, sys
from PIL import Image

ROOT = None
BG = np.array([[47, 47, 47], [162, 186, 224], [239, 239, 239]], np.int16)
REF = [
    ("navy",   0.098, 0.098, 0.439), ("blue",  0.306, 0.475, 0.655),
    ("brown",  0.612, 0.459, 0.373), ("green", 0.349, 0.663, 0.310),
    ("red",    1.000, 0.341, 0.349), ("purple", 0.690, 0.478, 0.631),
    ("orange", 0.949, 0.557, 0.169), ("cyan",  0.463, 0.718, 0.698),
    ("gray",   0.729, 0.690, 0.675), ("pink",  1.000, 0.616, 0.655),
    ("yellow", 0.929, 0.788, 0.282),
]
NAMES = [r[0] for r in REF]
RGB = np.array([r[1:] for r in REF], float)
INTEN = RGB @ np.array([0.299, 0.587, 0.114])
CHROMA = RGB / RGB.sum(1, keepdims=True)
LUMW = np.array([0.299, 0.587, 0.114])


def cc_label(mask):
    """Run-length two-pass labeling (4-connectivity). Returns (labels, n)."""
    h, w = mask.shape
    parent = [0]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    runs_all = []
    prev = []
    for y in range(h):
        row = mask[y]
        if not row.any():
            prev = []
            runs_all.append([])
            continue
        d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        runs = []
        for s, e in zip(starts, ends):
            lab = 0
            k = 0
            while k < len(prev) and prev[k][1] <= s:
                k += 1
            kk = k
            while kk < len(prev) and prev[kk][0] < e:
                r = find(prev[kk][2])
                if lab == 0:
                    lab = r
                elif r != lab:
                    a_, b_ = find(lab), r
                    if a_ != b_:
                        parent[max(a_, b_)] = min(a_, b_)
                        lab = min(a_, b_)
                kk += 1
            if lab == 0:
                lab = len(parent)
                parent.append(lab)
            runs.append((s, e, lab))
        prev = runs
        runs_all.append(runs)

    mapping = {}
    out = np.zeros((h, w), np.int32)
    nid = 0
    for y, runs in enumerate(runs_all):
        for s, e, lab in runs:
            r = find(lab)
            mo = mapping.get(r)
            if mo is None:
                nid += 1
                mapping[r] = mo = nid
            out[y, s:e] = mo
    return out, nid


def object_mask(a, thr=40):
    f = a.reshape(-1, 3).astype(np.int16)
    d = np.abs(f[None] - BG[:, None, :]).sum(2).min(0)
    return (d > thr).reshape(a.shape[:2])


def detect(a, min_area=180, merge_px=13.0):
    """Detect tabletop objects. Returns list of dicts."""
    m = object_mask(a)
    lab, n = cc_label(m)
    if n == 0:
        return []
    cnt = np.bincount(lab.ravel(), minlength=n + 1)
    objs = []
    H, W = a.shape[:2]
    for i in range(1, n + 1):
        if cnt[i] < min_area:
            continue
        ys, xs = np.where(lab == i)
        if ys.min() <= 1 or xs.min() <= 1 or ys.max() >= H - 2 or xs.max() >= W - 2:
            continue
        pix = a[ys, xs].astype(np.float32)
        lum = pix @ LUMW
        top = pix[lum >= np.percentile(lum, 80)].mean(0) + 1e-6
        ch = top / top.sum()
        k = int(np.argmin(((ch[None] - CHROMA) ** 2).sum(1)))
        objs.append(dict(cx=float(xs.mean()), cy=float(ys.mean()), area=float(cnt[i]),
                         color=k, inten=float(INTEN[k]), cdist=float(((ch[None] - CHROMA) ** 2).sum(1).min()),
                         lum_lo=float(np.percentile(lum, 5) / (np.median(lum) + 1e-6)),
                         lum_hi=float(np.percentile(lum, 95) / (np.median(lum) + 1e-6)),
                         x0=int(xs.min()), y0=int(ys.min()), x1=int(xs.max()), y1=int(ys.max()),
                         w=int(xs.max() - xs.min() + 1), h=int(ys.max() - ys.min() + 1)))
    changed = True
    while changed:
        changed = False
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                if np.hypot(objs[i]["cx"] - objs[j]["cx"], objs[i]["cy"] - objs[j]["cy"]) < merge_px:
                    A_, B_ = objs[i], objs[j]
                    t = A_["area"] + B_["area"]
                    big, small = (A_, B_) if A_["area"] >= B_["area"] else (B_, A_)
                    new = dict(big)
                    new.update(cx=(A_["cx"] * A_["area"] + B_["cx"] * B_["area"]) / t,
                               cy=(A_["cy"] * A_["area"] + B_["cy"] * B_["area"]) / t, area=t,
                               x0=min(A_["x0"], B_["x0"]), y0=min(A_["y0"], B_["y0"]),
                               x1=max(A_["x1"], B_["x1"]), y1=max(A_["y1"], B_["y1"]))
                    new["w"] = new["x1"] - new["x0"] + 1
                    new["h"] = new["y1"] - new["y0"] + 1
                    objs = [o for k2, o in enumerate(objs) if k2 not in (i, j)] + [new]
                    changed = True
                    break
            if changed:
                break
    return objs
