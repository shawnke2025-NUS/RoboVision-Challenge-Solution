"""Scene parser v4: colour-split connected components (handles touching objects)."""
import numpy as np, sys, os, json
sys.path.insert(0, r"C:\Users\shawn\Desktop\robo-vision-challenge-data-hack-3-0\work")
import solution_core as S
from feat import cc_label, NAMES, INTEN, RGB, CHROMA

ROOT = S.find_root()
C = np.array([0.0, -0.5])
BG = np.array([[47, 47, 47], [162, 186, 224], [239, 239, 239]], np.float32)
ALB = RGB * 255.0
ALB_N2 = (ALB ** 2).sum(1)


def masks(img):
    f = img.reshape(-1, 3).astype(np.float32)
    d = np.abs(f[:, None, :] - BG[None]).sum(2).min(1)
    mx, mn = f.max(1), f.min(1)
    return ((d > 42) & ~((mx - mn) < 12)).reshape(img.shape[:2])


def norm_rgb(rgb):
    r = np.asarray(rgb, np.float32) + 1e-6
    return r / r.sum()


def pixel_colors(pix):
    """nearest reference albedo (up to shading) for a (N,3) float array."""
    p = np.asarray(pix, np.float32)
    proj = p @ ALB.T
    sc = np.clip(proj / ALB_N2[None], 0.05, None)
    resid = ((p[:, None, :] - sc[:, :, None] * ALB[None]) ** 2).sum(-1) ** .5
    return resid.argmin(1), resid.min(1)


def make_obj(ys, xs, img):
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    pix = img[ys, xs].astype(np.float32)
    lum = pix @ np.array([0.299, 0.587, 0.114])
    sel = lum >= np.percentile(lum, 78)
    top = pix[sel]
    return dict(cx=float(xs.mean()), cy=float(ys.mean()), area=float(len(xs)),
                h=h, w=w, fill=len(xs) / (h * w), hw=h / max(w, 1),
                tall=h / np.sqrt(len(xs)), y1=int(ys.max()), y0=int(ys.min()),
                x1=int(xs.max()), x0=int(xs.min()),
                tx=float(xs[sel].mean()), ty=float(ys[sel].mean()),
                bw=float(min(w, h)), lum_p90=float(np.percentile(lum, 90)),
                rgb=top.mean(0), chroma=norm_rgb(top.mean(0)))


def comps(img, min_area=70, share=0.16, fill_min=0.24):
    m = masks(img)
    L, n = cc_label(m)
    if n == 0:
        return []
    cnt = np.bincount(L.ravel(), minlength=n + 1)
    H, W = img.shape[:2]
    out = []
    for i in range(1, n + 1):
        if cnt[i] < min_area:
            continue
        ys, xs = np.where(L == i)
        if ys.max() >= H - 2 or xs.max() >= W - 2 or ys.min() <= 1 or xs.min() <= 1:
            if cnt[i] > 2600:
                continue
        # split by reference colour so touching objects separate
        k, res = pixel_colors(img[ys, xs])
        uniq, inv, counts = np.unique(k, return_inverse=True, return_counts=True)
        big = [j for j in range(len(uniq)) if counts[j] >= max(60, share * cnt[i])]
        if len(big) <= 1:
            parts = [np.arange(len(k))]
        else:
            parts = [np.where(inv == j)[0] for j in big]
        for part in parts:
            if len(part) < min_area:
                continue
            py, px_ = ys[part], xs[part]
            o = make_obj(py, px_, img)
            if o["h"] < 6 or o["w"] < 6 or o["fill"] < 0.24 or max(o["h"] / o["w"], o["w"] / o["h"]) > 3.0:
                continue
            o["border"] = (o["y0"] <= 1 or o["x0"] <= 1 or o["y1"] >= H - 2 or o["x1"] >= W - 2)
            out.append(o)
    # merge only small same-colour fragments
    if out:
        med = float(np.median([o["area"] for o in out]))
        changed = True
        while changed:
            changed = False
            for a in range(len(out)):
                for b in range(a + 1, len(out)):
                    A_, B_ = out[a], out[b]
                    dc = float(np.linalg.norm(A_["chroma"] - B_["chroma"]))
                    dist = np.hypot(A_["cx"] - B_["cx"], A_["cy"] - B_["cy"])
                    if dc < 0.012 and dist < 0.5 * (A_["w"] + B_["w"]) and min(A_["area"], B_["area"]) < 0.5 * med:
                        t = A_["area"] + B_["area"]
                        new = dict(A_)
                        new.update(cx=(A_["cx"] * A_["area"] + B_["cx"] * B_["area"]) / t,
                                   cy=(A_["cy"] * A_["area"] + B_["cy"] * B_["area"]) / t,
                                   area=t, h=max(A_["h"], B_["h"]),
                                   w=max(A_["w"], B_["w"]))
                        new["hw"] = new["h"] / max(new["w"], 1)
                        out = [o for q, o in enumerate(out) if q not in (a, b)] + [new]
                        changed = True
                        break
                if changed:
                    break
    return out


def parse(img, cam, min_area=70, share=0.16, fill_min=0.24):
    objs = comps(img, min_area, share, fill_min)
    if len(objs) < 6:
        return None
    uv = np.array([[o["cx"], o["y1"] - 0.5 * o["h"] * 0.32] for o in objs])
    P = cam.backproject(uv, 0.0)[:, :2]
    for o, q in zip(objs, P):
        o["wx"], o["wy"] = float(q[0]), float(q[1])
    return objs
