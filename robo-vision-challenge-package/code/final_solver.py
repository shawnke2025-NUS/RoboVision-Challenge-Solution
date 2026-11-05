"""Final solver: calibrated parameters + test submission generation."""
import numpy as np, pandas as pd, sys, os, json, pickle, time
sys.path.insert(0, r"C:\Users\shawn\Desktop\robo-vision-challenge-data-hack-3-0\work")
import scene3, solution_core as S
from feat import INTEN, CHROMA
from PIL import Image

ROOT = scene3.ROOT
C = np.array([0.0, -0.5])

# ---- calibrated constants (fitted on train, see fit_pos3.py) ----
FRAC = 0.39          # base-point = bbox bottom - FRAC * height
H_CUBE = 0.010       # unprojection plane for blocks
H_BOWL = 0.000
SCALE = 1.290
AFF = {
    "bowl": np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
    "cube": np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
}
F_CUBE = lambda t: 0.483 * np.cos(t) + 0.773 * np.sin(t) + 0.269
F_BOWL = lambda t: 0.565 * np.cos(t) + 1.291 * np.sin(t) - 0.445


def elevation(cam, wx, wy):
    return float(np.arctan2(cam.t[2], np.hypot(cam.t[0] - wx, cam.t[1] - wy)))


def shape_of(o, cam):
    t = elevation(cam, o["wx"], o["wy"])
    zc = abs(o["hw"] - F_CUBE(t)) / 0.093
    zb = abs(o["hw"] - F_BOWL(t)) / 0.046
    return 1 if zc < zb else 0


def ref_intensity(ch):
    d = ((CHROMA - ch) ** 2).sum(1) ** .5
    return float(INTEN[int(d.argmin())]), float(d.min())


def solve(objs, cam, task, dmax=0.030):
    """-> (6x2 pick predictions in metres, info)."""
    if not objs or len(objs) < 6:
        return None, "few"
    n = len(objs)
    cand = []
    for a in range(n):
        for b in range(a + 1, n):
            cand.append((float(np.linalg.norm(objs[a]["chroma"] - objs[b]["chroma"])), a, b))
    cand.sort()
    used, pairs = set(), []
    for d, a, b in cand:
        if d > dmax:
            break
        if a in used or b in used:
            continue
        used.add(a); used.add(b); pairs.append((a, b))
    if len(pairs) not in (2, 3):
        return None, "pairs=%d" % len(pairs)
    partial = (len(pairs) == 2)
    rows = []
    for a, b in pairs:
        oa, ob = objs[a], objs[b]
        sa, sb = shape_of(oa, cam), shape_of(ob, cam)
        if sa == sb:
            sa, sb = (1, 0) if oa["hw"] >= ob["hw"] else (0, 1)
        cube, bowl = (oa, ob) if sa == 1 else (ob, oa)
        ch = scene3.norm_rgb(0.5 * (oa["rgb"] + ob["rgb"]))
        inten, cd = ref_intensity(ch)
        if cd > 0.02:
            return None, "colour"
        rows.append((inten, bowl, cube))
    rows.sort(key=lambda r: r[0], reverse=(task == "task2"))
    out = np.full((6, 2), np.nan)
    for s, (_, bowl, cube) in enumerate(rows):
        for slot, o, key in ((s, bowl, "bowl"), (s + 3, cube, "cube")):
            h = H_BOWL if key == "bowl" else H_CUBE
            p = cam.backproject(np.array([[o["cx"], o["y1"] - FRAC * o["h"]]]), h)[0][:2]
            q = C + SCALE * (p - C)
            m = AFF[key]
            out[slot] = [m[0, 0] * q[0] + m[0, 1] * q[1] + m[2, 0],
                         m[1, 0] * q[0] + m[1, 1] * q[1] + m[2, 1]]
    return out, ("partial" if partial else "ok")


PLACE = None


def place_table(df):
    """exact (x_place, y_place) per (task_type, step)"""
    A = S.parse_actions(df)
    tab = {}
    for t in df.task_type.unique():
        m = (df.task_type == t).values
        for s in range(6):
            tab[(t, s)] = tuple(np.round(A[m, s, 2:].mean(0), 4))
    return tab


if __name__ == "__main__":
    tr = pd.read_csv(os.path.join(ROOT, "train.csv"))
    te = pd.read_csv(os.path.join(ROOT, "test.csv"))
    PLACE = place_table(tr)
    Atr = S.parse_actions(tr)
    MEAN_PICK = {}
    for t in tr.task_type.unique():
        m = (tr.task_type == t).values
        for s in range(6):
            MEAN_PICK[(t, s)] = tuple(np.round(Atr[m, s, :2].mean(0), 4))

    for name, df in (("train", tr.head(400)), ("test", te)):
        t0 = time.time()
        pred = np.full((len(df), 6, 2), np.nan)
        n_ok = n_part = n_fail = 0
        categories = []
        for i in range(len(df)):
            img = np.asarray(Image.open(os.path.join(ROOT, df.image_path[i].replace("/", os.sep))).convert("RGB"))
            cam = S.Cam(json.load(open(os.path.join(ROOT, df.camera_params_path[i].replace("/", os.sep)))))
            objs = scene3.parse(img, cam)
            p, info = solve(objs, cam, df.task_type[i])
            if p is None or info != "ok":
                # second chance: looser segmentation (recovers small / attached objects)
                for kw in (dict(min_area=45, share=0.10, fill_min=0.18),
                           dict(min_area=35, share=0.08, fill_min=0.15)):
                    objs2 = scene3.parse(img, cam, **kw)
                    p2, info2 = solve(objs2, cam, df.task_type[i])
                    if p2 is not None and info2 == "ok":
                        p, info = p2, "ok(2nd)"
                        break
            if p is None:
                n_fail += 1
                categories.append("fail")
                for s in range(6):
                    pred[i, s] = MEAN_PICK[(df.task_type[i], s)]
            else:
                if info == "partial":
                    n_part += 1
                    categories.append("partial")
                else:
                    n_ok += 1
                    categories.append("full")
                for s in range(6):
                    if np.isnan(p[s, 0]):
                        pred[i, s] = MEAN_PICK[(df.task_type[i], s)]
                    else:
                        pred[i, s] = p[s]
        print("%s: full %d  partial %d  fail %d  (%.0fs)" % (name, n_ok, n_part, n_fail, time.time() - t0))
        if name == "train":
            gt = Atr[:len(df), :, :2]
            d = np.linalg.norm(pred - gt, axis=-1)
            print("   train metric on these rows: %.4f  hit3 %.3f  hit5 %.3f" % (
                np.exp(-(d / 0.05) ** 2).mean(), (d < .03).mean(), (d < .05).mean()))
            cat = np.array(categories)
            for c in ("full", "partial", "fail"):
                m = cat == c
                if m.sum():
                    print("     %-8s n=%3d  metric %.4f  hit3 %.3f" % (
                        c, m.sum(), np.exp(-(d[m] / 0.05) ** 2).mean(), (d[m] < .03).mean()))
        if name == "test":
            rows = []
            for i, (_, r) in enumerate(df.iterrows()):
                o = {"image_id": r.image_id}
                for s in range(6):
                    x, y = pred[i, s]
                    xp, yp = PLACE[(r.task_type, s)]
                    o[f"action_{s}"] = f"{x:.4f},{y:.4f},{xp:.4f},{yp:.4f}"
                rows.append(o)
            sub = pd.DataFrame(rows)
            sub.to_csv(r"C:\Users\shawn\Desktop\robo-vision-challenge-data-hack-3-0\submission_solver.csv", index=False)
            print(sub.head(3).to_string())
            print("rows", len(sub))
