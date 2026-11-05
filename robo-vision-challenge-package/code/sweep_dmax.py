import numpy as np, pandas as pd, sys, os, pickle
sys.path.insert(0, r"C:\Users\shawn\Desktop\robo-vision-challenge-data-hack-3-0\work")
import scene3, solver_final as SF, solution_core as S

ROOT = scene3.ROOT
tr = pd.read_csv(os.path.join(ROOT, "train.csv"))
A = S.parse_actions(tr)
data = pickle.load(open(r"C:\Users\shawn\Desktop\robo-vision-challenge-data-hack-3-0\work\objs400.pkl", "rb"))
C = SF.C
MEAN = {}
for t in tr.task_type.unique():
    m = (tr.task_type == t).values
    for s in range(6):
        MEAN[(t, s)] = A[m, s, :2].mean(0)


def run(dmax, require=3):
    pred = np.zeros((len(data), 6, 2))
    stats = {"full": 0, "partial": 0, "fail": 0}
    for i, (objs, cam) in enumerate(data):
        p, info = SF.solve(objs, cam, tr.task_type[i], dmax=dmax) if False else (None, None)
        # inline solve with configurable dmax
        if objs is None or len(objs) < 6:
            p, info = None, "few"
        else:
            n = len(objs)
            cand = sorted((float(np.linalg.norm(objs[a]["chroma"] - objs[b]["chroma"])), a, b)
                          for a in range(n) for b in range(a + 1, n))
            used, pairs = set(), []
            for d, a, b in cand:
                if d > dmax:
                    break
                if a in used or b in used:
                    continue
                used.add(a); used.add(b); pairs.append((a, b))
            if len(pairs) != require:
                p, info = None, "pairs=%d" % len(pairs)
            else:
                rows = []
                bad = False
                for a, b in pairs:
                    oa, ob = objs[a], objs[b]
                    sa, sb = SF.shape_of(oa, cam), SF.shape_of(ob, cam)
                    if sa == sb:
                        sa, sb = (1, 0) if oa["hw"] >= ob["hw"] else (0, 1)
                    cube, bowl = (oa, ob) if sa == 1 else (ob, oa)
                    inten, cd = SF.ref_intensity(scene3.norm_rgb(0.5 * (oa["rgb"] + ob["rgb"])))
                    if cd > 0.02:
                        bad = True
                    rows.append((inten, bowl, cube))
                if bad:
                    p, info = None, "colour"
                else:
                    rows.sort(key=lambda r: r[0], reverse=(tr.task_type[i] == "task2"))
                    out = np.zeros((6, 2))
                    for s, (_, bowl, cube) in enumerate(rows):
                        for slot, o, h in ((s, bowl, 0.0), (s + 3, cube, 0.010)):
                            q = cam.backproject(np.array([[o["cx"], o["y1"] - 0.39 * o["h"]]]), h)[0][:2]
                            out[slot] = C + 1.290 * (q - C)
                    p, info = out, "ok"
        if p is None:
            stats["fail"] += 1
            pred[i] = np.array([MEAN[(tr.task_type[i], s)] for s in range(6)])
        else:
            stats["full"] += 1
            pred[i] = p
    gt = A[:len(data), :, :2]
    d = np.linalg.norm(pred - gt, axis=-1)
    return stats, float(np.exp(-(d / 0.05) ** 2).mean()), float((d < .03).mean())


for dmax in [0.014, 0.02, 0.03, 0.05, 0.08]:
    st, m, h3 = run(dmax)
    print("dmax=%.3f  full %3d  fail %3d  metric %.4f  hit3 %.3f" % (dmax, st["full"], st["fail"], m, h3))

