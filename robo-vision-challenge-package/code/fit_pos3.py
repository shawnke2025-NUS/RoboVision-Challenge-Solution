import numpy as np, pandas as pd, sys, os, pickle
sys.path.insert(0, r"C:\Users\shawn\Desktop\robo-vision-challenge-data-hack-3-0\work")
import scene3, solver_final as SF, solution_core as S
from learn_correct import solve_details

ROOT = scene3.ROOT
tr = pd.read_csv(os.path.join(ROOT, "train.csv"))
A = S.parse_actions(tr)
data = pickle.load(open("objs400.pkl", "rb"))
C = SF.C
slots_all = []
for i, (objs, cam) in enumerate(data):
    if objs is None:
        continue
    sl = solve_details(objs, cam, tr.task_type[i])
    if sl is None:
        continue
    slots_all.append((i, cam, sl))
gt = np.array([A[i, :, :2] for i, _, _ in slots_all])
print("images", len(slots_all))


def pos(frac, hc, hb):
    out = np.zeros((len(slots_all), 6, 2))
    for k, (i, cam, sl) in enumerate(slots_all):
        for s, (o, is_cube) in enumerate(sl):
            uv = np.array([[o["cx"], o["y1"] - frac * o["h"]]])
            out[k, s] = cam.backproject(uv, hc if is_cube else hb)[0][:2]
    return out


def score(P):
    d = np.linalg.norm(P - gt, axis=-1)
    return np.exp(-(d / 0.05) ** 2).mean()


best = None
for frac in np.arange(0.10, 0.61, 0.05):
    for hc in [0.0, 0.01, 0.02, 0.03]:
        for sc in [1.15, 1.20, 1.24, 1.26, 1.30]:
            P = C + sc * (pos(frac, hc, 0.0) - C)
            s = score(P)
            if best is None or s > best[0]:
                best = (s, frac, hc, sc)
print("best: score %.4f frac=%.2f hc=%.2f scale=%.2f" % best)

s0, frac, hc, sc = best
best2 = None
for f2 in np.arange(frac - 0.05, frac + 0.051, 0.02):
    for h2 in np.arange(max(0, hc - 0.01), hc + 0.012, 0.005):
        for s2 in np.arange(sc - 0.03, sc + 0.031, 0.01):
            s = score(C + s2 * (pos(f2, h2, 0.0) - C))
            if best2 is None or s > best2[0]:
                best2 = (s, f2, h2, s2)
print("fine: score %.4f frac=%.2f hc=%.3f scale=%.3f" % best2)

# per-object height from projected size instead of a constant
_, frac, hc, sc = best2
P = C + sc * (pos(frac, hc, 0.0) - C)
d = np.linalg.norm(P - gt, axis=-1)
print("hit3 %.3f hit5 %.3f  p99 err %.1f cm" % ((d < .03).mean(), (d < .05).mean(), np.percentile(d, 99) * 100))
# how much of the error is in the tail?
bad = d > 0.05
print("frac actions >5cm: %.3f" % bad.mean())
