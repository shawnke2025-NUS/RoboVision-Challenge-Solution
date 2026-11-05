"""Fetch the finished Kaggle kernel output and sync it back into the project folder."""
import os
import shutil
import subprocess
import sys

BASE = r"C:\Users\shawn\Desktop\robo-vision-challenge-data-hack-3-0"
REF = "shawnjuyangke/robovision-3-0-bev-heatmap-u-net-t4x2"
PY = r"C:\Users\shawn\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
TOKEN = os.environ.get("KAGGLE_API_TOKEN", "")


def run(*args):
    env = dict(os.environ)
    if TOKEN:
        env["KAGGLE_API_TOKEN"] = TOKEN
    env["PYTHONIOENCODING"] = "utf-8"
    p = subprocess.run([PY, "-m", "kaggle", *args], capture_output=True, text=True, env=env)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def main():
    out = os.path.join(BASE, "kaggle_output")
    os.makedirs(out, exist_ok=True)
    rc, txt = run("kernels", "output", REF, "-p", out)
    print(txt.strip()[-2000:])
    if rc != 0:
        print("output download failed")
        return 1
    for name in os.listdir(out):
        print(" -", name)
    src = os.path.join(out, "submission.csv")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(BASE, "submission.csv"))
        print("copied submission.csv ->", BASE)
        import pandas as pd
        df = pd.read_csv(src)
        print(df.shape)
        print(df.head(3).to_string())
    else:
        print("submission.csv not found in kernel output")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
