import numpy as np, pandas as pd, sys, pickle
sys.path.insert(0,r"C:\Users\shawn\Desktop\robo-vision-challenge-data-hack-3-0\work")
import scene3, solution_core as S
from feat import NAMES, INTEN, CHROMA
rows=pickle.load(open("objs200_v4.pkl","rb"))
tr=pd.read_csv(scene3.ROOT+r"\train.csv")
import os, json
# rebuild theta for each object (needs camera) - approximate from stored data is impossible,
# so recompute for 120 images
from PIL import Image
import solution_core as S2
th=[]; hw=[]; lab=[]
for i in range(120):
    objs=rows[i]
    if not objs: continue
    cam=S.Cam(json.load(open(os.path.join(scene3.ROOT,tr.camera_params_path[i].replace("/",os.sep)))))
    # pairs by chroma
    n=len(objs); cand=[]
    for a in range(n):
        for b in range(a+1,n):
            cand.append((float(np.linalg.norm(objs[a]["chroma"]-objs[b]["chroma"])),a,b))
    cand.sort(); used=set(); pairs=[]
    for d,a,b in cand:
        if d>0.014: break
        if a in used or b in used: continue
        used.add(a); used.add(b); pairs.append((a,b))
    if len(pairs)!=3: continue
    for a,b in pairs:
        gap=objs[a]["hw"]-objs[b]["hw"]
        if abs(gap)<0.10: continue     # only clear cases
        hi,lo=(a,b) if gap>0 else (b,a)
        for idx,shape in ((hi,1),(lo,0)):
            o=objs[idx]
            dx=cam.t[0]-o["wx"]; dy=cam.t[1]-o["wy"]
            theta=np.arctan2(cam.t[2], np.hypot(dx,dy))
            th.append(theta); hw.append(o["hw"]); lab.append(shape)
th=np.array(th); hw=np.array(hw); lab=np.array(lab)
print("clear cases:",len(th)," cubes",int(lab.sum())," bowls",int((1-lab).sum()))
for shape,name in ((1,"cube"),(0,"bowl")):
    m=lab==shape
    X=np.stack([np.cos(th[m]),np.sin(th[m]),np.ones(m.sum())],1)
    sol,res,_,_=np.linalg.lstsq(X,hw[m],rcond=None)
    pred=X@sol
    print("%s: hw ~ %.3f*cos + %.3f*sin + %.3f  resid std %.3f"%(name,sol[0],sol[1],sol[2],(hw[m]-pred).std()))
print("elevation range deg:",np.round(np.degrees(np.percentile(th,[5,50,95])),1))
