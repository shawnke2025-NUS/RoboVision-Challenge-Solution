import json, sys, os, numpy as np
os.chdir(r"C:\Users\shawn\Desktop\robo-vision-challenge-data-hack-3-0")
NB = r"C:\Users\shawn\Desktop\robo-vision-challenge-data-hack-3-0\robo_vision_solver_v3.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
g = {"__name__": "__main__"}
# run the analytical part only: setup -> parse/label -> camera/cc -> detect -> rule engine -> validation
for k in range(6):
    src = codes[k]
    if k == 5:                       # shrink the validation loop
        src = src.replace("NVAL = 300", "NVAL = 60")
    exec(compile(src, f"cell{k}", "exec"), g)
print("\nNOTEBOOK SOLVER CELLS OK")
