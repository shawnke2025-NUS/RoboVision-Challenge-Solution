"""
RoboVision Challenge - DataHack 3.0 : core solution.

Key insights
------------
1. action_k = [x, y, z, grip].  For a given (task_type, step) the (z, grip) pair is a
   *constant*: there are only two possible sequences in train and they are reproduced
   exactly in test.  -> 12 of the 24 submitted numbers are free.
2. (x, y) are tabletop coordinates in metres and every scene is rendered from a random
   camera pose.  Warping each frame into a canonical bird's-eye view (BEV) of the table
   plane with the supplied intrinsics/extrinsics removes nearly all nuisance variation.
3. A small U-Net then predicts one localisation heat-map per step directly in BEV pixel
   space; soft-argmax converts them back to world metres.  Sub-pixel soft-argmax beats
   direct coordinate regression.
"""
import os, json, math, time, glob
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------
X0, X1 = -0.45, 0.45
Y0, Y1 = -0.90, -0.10
PX = 320.0
BEV_W = int(round((X1 - X0) * PX))
BEV_H = int(round((Y1 - Y0) * PX))
NSTEP = 6
LUM = np.array([0.299, 0.587, 0.114])


def find_root():
    """Locate the Challenge_Dataset folder (Kaggle input or local copy)."""
    cands = []
    for pat in ["/kaggle/input/*/Challenge_Dataset", "/kaggle/input/Challenge_Dataset",
                "/kaggle/input/*/*/Challenge_Dataset"]:
        cands += glob.glob(pat)
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    cands += [here, os.path.join(here, "Challenge_Dataset"), os.path.join(here, "..", "Challenge_Dataset")]
    for c in cands:
        if os.path.exists(os.path.join(c, "train.csv")):
            return os.path.abspath(c)
    raise FileNotFoundError("Challenge_Dataset not found")


class Cam:
    """Camera: OpenCV convention, extrinsics = camera-to-world."""

    def __init__(self, d):
        self.K = np.array(d["intrinsics"], float)
        self.E = np.array(d["extrinsics"], float)
        self.R = self.E[:3, :3]
        self.t = self.E[:3, 3]

    def project(self, P):
        Xc = (np.asarray(P, float) - self.t) @ self.R
        z = Xc[..., 2]
        uv = Xc[..., :2] / z[..., None]
        return uv @ self.K[:2, :2].T + self.K[:2, 2], z

    def backproject(self, uv, z0=0.0):
        uv = np.asarray(uv, float)
        d = (uv - self.K[:2, 2]) @ np.linalg.inv(self.K[:2, :2]).T
        d = np.concatenate([d, np.ones(d.shape[:-1] + (1,))], -1)
        dw = d @ self.R.T
        s = (z0 - self.t[2]) / dw[..., 2]
        return self.t + s[..., None] * dw


_MESH = None


def _mesh():
    global _MESH
    if _MESH is None:
        xs = X0 + (np.arange(BEV_W) + 0.5) / PX
        ys = Y0 + (np.arange(BEV_H) + 0.5) / PX
        XX, YY = np.meshgrid(xs, ys)
        _MESH = np.stack([XX, YY, np.zeros_like(XX)], -1).reshape(-1, 3)
    return _MESH


def birdseye(img, cam):
    """Warp a 640x480 frame into the canonical top-down BEV."""
    P = _mesh()
    uv, z = cam.project(P)
    u, v = uv[:, 0], uv[:, 1]
    h, w = img.shape[:2]
    ok = (z > 0.05) & (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
    u0 = np.clip(u.astype(np.int32), 0, w - 2)
    v0 = np.clip(v.astype(np.int32), 0, h - 2)
    fu = (u - u0).astype(np.float32)[:, None]
    fv = (v - v0).astype(np.float32)[:, None]
    f = img.astype(np.float32)
    c = (f[v0, u0] * (1 - fu) * (1 - fv) + f[v0, u0 + 1] * fu * (1 - fv)
         + f[v0 + 1, u0] * (1 - fu) * fv + f[v0 + 1, u0 + 1] * fu * fv)
    out = np.zeros((BEV_H * BEV_W, 3), np.float32)
    out[ok] = c[ok]
    return out.reshape(BEV_H, BEV_W, 3).astype(np.uint8)


def w2b(x, y):
    """world metres -> BEV pixels (works for numpy and torch)."""
    return (x - X0) * PX, (y - Y0) * PX


def b2w(bx, by):
    """BEV pixels -> world metres (works for numpy and torch)."""
    return bx / PX + X0, by / PX + Y0


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------
def parse_actions(df):
    """-> (N, 6, 4) float array"""
    return np.stack([df[f"action_{i}"].str.split(",", expand=True).astype(float).values
                     for i in range(NSTEP)], 1)


def load_split(root, name, cache=True):
    """Load images -> BEV stack (uint8 N,H,W,3) plus camera metadata and targets."""
    df = pd.read_csv(os.path.join(root, f"{name}.csv"))
    cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"bev_{name}.npy") \
        if "__file__" in globals() else f"bev_{name}.npy"
    if cache and os.path.exists(cache_path):
        arr = np.load(cache_path, mmap_mode="r")
        if len(arr) == len(df):
            return df, arr
    arr = np.zeros((len(df), BEV_H, BEV_W, 3), np.uint8)
    for i in range(len(df)):
        img = np.asarray(Image.open(os.path.join(root, df.image_path.iloc[i].replace("/", os.sep)))
                         .convert("RGB"))
        cam = Cam(json.load(open(os.path.join(root, df.camera_params_path.iloc[i].replace("/", os.sep)))))
        arr[i] = birdseye(img, cam)
        if i % 500 == 0:
            print(f"  {name} bev {i}/{len(df)}", flush=True)
    if cache:
        np.save(cache_path, arr)
    return df, arr


class BEVDataset(torch.utils.data.Dataset):
    def __init__(self, bev, xy, task, augment=False):
        self.bev = bev            # (N,H,W,3) uint8, memory mapped
        self.xy = xy              # (N,6,2) world metres
        self.task = task          # (N,) 0/1
        self.aug = augment
        self.bxy = np.stack(w2b(xy[..., 0], xy[..., 1]), -1).astype(np.float32)  # (N,6,2) px

    def __len__(self):
        return len(self.bev)

    def __getitem__(self, i):
        img = np.asarray(self.bev[i], np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        t = torch.from_numpy(np.ascontiguousarray(img))
        t = torch.cat([t, torch.full((1, BEV_H, BEV_W), float(self.task[i]))], 0)
        bxy = torch.from_numpy(self.bxy[i])
        xy = torch.from_numpy(self.xy[i].astype(np.float32))
        if self.aug:
            # photometric + small translation (targets move with the image)
            t[:3] = (t[:3] * (0.85 + 0.3 * torch.rand(1))).clamp(0, 1)
            t[:3] = (t[:3] + 0.03 * (torch.rand(1) - 0.5)).clamp(0, 1)
        return t, bxy, xy


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------
def conv_bn(cin, cout, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, bias=False), nn.BatchNorm2d(cout), nn.SiLU(True))


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = conv_bn(c, c); self.c2 = nn.Conv2d(c, c, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)

    def forward(self, x):
        return F.silu(x + self.bn(self.c2(self.c1(x))), True)


class Net(nn.Module):
    """Small U-Net: BEV -> 6 localisation heat-maps (stride 2) + global xy head."""

    def __init__(self, cin=4, width=32, nstep=NSTEP):
        super().__init__()
        w = width
        self.s1 = nn.Sequential(conv_bn(cin, w, 3, 2), Res(w))          # /2
        self.s2 = nn.Sequential(conv_bn(w, w * 2, 3, 2), Res(w * 2))    # /4
        self.s3 = nn.Sequential(conv_bn(w * 2, w * 4, 3, 2), Res(w * 4))  # /8
        self.s4 = nn.Sequential(conv_bn(w * 4, w * 8, 3, 2), Res(w * 8))  # /16
        self.u3 = conv_bn(w * 8 + w * 4, w * 4)
        self.u2 = conv_bn(w * 4 + w * 2, w * 2)
        self.u1 = conv_bn(w * 2 + w, w)
        self.head = nn.Conv2d(w, nstep, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reg = nn.Sequential(nn.Linear(w * 8, 256), nn.SiLU(True), nn.Linear(256, nstep * 2))
        nn.init.zeros_(self.head.weight); nn.init.constant_(self.head.bias, -2.0)
        self.nstep = nstep

    def forward(self, x):
        f1 = self.s1(x)
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        f4 = self.s4(f3)
        g = self.pool(f4).flatten(1)
        reg = self.reg(g).view(-1, self.nstep, 2)
        y = F.interpolate(f4, scale_factor=2, mode="nearest")
        y = self.u3(torch.cat([y, f3], 1))
        y = F.interpolate(y, scale_factor=2, mode="nearest")
        y = self.u2(torch.cat([y, f2], 1))
        y = F.interpolate(y, scale_factor=2, mode="nearest")
        y = self.u1(torch.cat([y, f1], 1))
        return self.head(y), reg


def heatmap_targets(bxy, h, w, sigma=4.0, scale=2.0):
    """bxy (B,6,2) in *full resolution* BEV pixels -> (B,6,h,w) gaussians.

    The heat-map has stride `scale` relative to the BEV, so pixel (i,j) of the heat-map
    sits at ((j+.5)*scale, (i+.5)*scale) in BEV pixels.
    """
    B, S_, _ = bxy.shape
    ys = (torch.arange(h, device=bxy.device, dtype=torch.float32) + .5) * scale
    xs = (torch.arange(w, device=bxy.device, dtype=torch.float32) + .5) * scale
    cx = bxy[..., 0][..., None, None]
    cy = bxy[..., 1][..., None, None]
    d2 = ((xs[None, None, None, :] - cx) ** 2 + (ys[None, None, :, None] - cy) ** 2)
    return torch.exp(-d2 / (2 * sigma ** 2))


def soft_argmax(hm, scale=2.0, beta=6.0):
    B, S_, h, w = hm.shape
    p = torch.softmax(hm.reshape(B, S_, -1) * beta, -1).reshape(B, S_, h, w)
    ys = (torch.arange(h, device=hm.device, dtype=torch.float32) + .5) * scale
    xs = (torch.arange(w, device=hm.device, dtype=torch.float32) + .5) * scale
    px = (p.sum(2) * xs).sum(-1)
    py = (p.sum(3) * ys).sum(-1)
    return torch.stack([px, py], -1)


# --------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------
def focal_heatmap_loss(logits, tgt, eps=1e-4):
    """CenterNet-style penalty-reduced focal loss (normalised by #targets)."""
    p = torch.sigmoid(logits).clamp(eps, 1 - eps)
    pos = tgt > 0.9
    npos = pos.sum().clamp(min=1)
    pos_loss = -((1 - p) ** 2 * torch.log(p) * tgt).sum()
    neg_loss = -((1 - tgt) ** 4 * p ** 2 * torch.log(1 - p)).sum()
    return (pos_loss + neg_loss) / npos


def run_epoch(model, dl, opt, scaler, train=True, device="cuda", l1w=8.0, regw=0.15):
    model.train(train)
    tot, n = 0.0, 0
    for x, bxy, xy in dl:
        x = x.to(device, non_blocking=True)
        bxy = bxy.to(device); xy = xy.to(device)
        with torch.autocast("cuda", enabled=(device == "cuda")):
            logits, reg = model(x)
            tgt_hm = heatmap_targets(bxy, logits.shape[2], logits.shape[3])
            lh = focal_heatmap_loss(logits, tgt_hm)
            sa = soft_argmax(logits)
            wx, wy = b2w(sa[..., 0], sa[..., 1])
            l1 = (torch.stack([wx, wy], -1) - xy).abs().mean()
            wrx, wry = b2w(reg[..., 0], reg[..., 1])
            lreg = (torch.stack([wrx, wry], -1) - xy).abs().mean()
            loss = lh + l1w * l1 + regw * lreg
        if train:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
        tot += float(loss.detach()) * len(x); n += len(x)
    return tot / max(n, 1)


@torch.no_grad()
def predict(model, dl, device="cuda", blend=0.0):
    model.eval()
    out = []
    for x, bxy, xy in dl:
        x = x.to(device)
        with torch.autocast("cuda", enabled=(device == "cuda")):
            logits, reg = model(x)
            sa = soft_argmax(logits)
            bxy_p = (1 - blend) * sa + blend * reg
            wx, wy = b2w(bxy_p[..., 0], bxy_p[..., 1])
        out.append(torch.stack([wx, wy], -1).float().cpu().numpy())
    return np.concatenate(out)
