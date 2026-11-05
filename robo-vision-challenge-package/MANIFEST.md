# RoboVision Challenge - DataHack 3.0  |  solution package

Assembled: 2026-11-05 20:49

## Result

| stage | public LB |
|---|---|
| per-(task,step) mean baseline | 0.00189 |
| v1 pure CNN, 252 epochs | 0.44662 |
| solver v1 (3 colour pairs, intensity order) | 0.86025 |
| + EfficientNet-B0 fallback | 0.87254 |
| **+ relaxed pairing + two-chance detection** | **0.87702 (rank 1 of 4)** |

## Contents

| file | bytes | sha256[:16] | note |
|---|---|---|---|
| `README.md` | 10915 | `02118098011a2961` | main write-up |
| `code/draw_flowchart.py` | 7673 | `c6ae110d148c779c` | renders Fig. 1 (vector PDF + PNG) |
| `code/feat.py` | 5078 | `60ca985868f34177` | reference colours + run-length connected components |
| `code/final_solver.py` | 6734 | `944c3367a86b3ec0` | rule engine + calibrated constants + submission writer |
| `code/finalize.py` | 1469 | `38664dceb0ba0b4c` | downloads kernel output from Kaggle |
| `code/fit_pos3.py` | 2253 | `597fcbe9cd53028f` | position calibration sweep (frac / plane / scale) |
| `code/fit_shape.py` | 2002 | `c0c694ee760beb68` | block-vs-bowl elevation model fit |
| `code/nb_builder3.py` | 31632 | `d5b037c52e35d0ae` | builds the main notebook from tested blocks |
| `code/nb_dryrun.py` | 697 | `dbd019e98bd38324` | executes the notebook's analytic cells end-to-end |
| `code/scene3.py` | 4888 | `2b936e11377957e8` | scene parser: mask / colour split / components |
| `code/solution_core.py` | 11950 | `c6bc4ff5b85929e0` | camera model (OpenCV convention) + IO helpers |
| `code/sweep_dmax.py` | 3214 | `b4f03ed37f914cca` | colour-pairing threshold sweep |
| `figure/algorithm_flowchart.pdf` | 102100 | `f42c07f35800d2e4` | Fig. 1, vector 7.20x3.64 in |
| `figure/algorithm_flowchart.png` | 428510 | `2ab9c52ead629f57` | Fig. 1, 400 dpi raster |
| `kaggle_run/robovision-3-0-deterministic-solver.log` | 6867 | `acd81316ff0c56f1` | full run log: 2x T4, EfficientNet weights, solver stats |
| `kaggle_upload/dataset-metadata.json` | 655 | `4444500b95d5ebdb` | dataset metadata |
| `kaggle_upload/kernel-metadata.json` | 458 | `ff6a3c9587470937` | notebook push metadata (T4x2) |
| `notebooks/robo_vision_solution.ipynb` | 27787 | `b8e615165dd628b0` | v1 pure-CNN baseline (LB 0.44662) |
| `notebooks/robo_vision_solver_v3.ipynb` | 38445 | `b5abd9977df9ea01` | MAIN: deterministic solver + EfficientNet fallback |
| `submissions/submission.csv` | 209061 | `74d0634dc9b0217b` | BEST - public LB 0.87702 |
| `submissions/submission_baseline.csv` | 209044 | `930618c36d0aa8a1` | per-(task,step) mean baseline (LB 0.00189) |
| `submissions/submission_solver.csv` | 210033 | `fbdf3c06e7d690a9` | solver v2 (LB 0.87702) |
| `submissions/submission_v3_final.csv` | 209061 | `74d0634dc9b0217b` | latest notebook output (train-est 0.864) |

## How to run

1. Kaggle: open `notebooks/robo_vision_solver_v3.ipynb`, set Accelerator = **GPU T4 x 2**,
   Internet = on (for the ImageNet weights; it degrades gracefully if off), Run All.
2. Local: `pip install numpy pandas pillow torch torchvision` then run the same notebook;
   it locates `Challenge_Dataset/` next to itself or under `/kaggle/input/`.
3. Regenerate Fig. 1: `cd code && python draw_flowchart.py`.

## Not included

* `Challenge_Dataset/` - the official competition data (2.6 GB), download from Kaggle.
* intermediate scratch scripts and `.pkl` detection caches - superseded by the files above.

## Notes

* Actions are `x_pick,y_pick,x_place,y_place`; `(x_place,y_place)` is one value per
  `(task_type, step)` and is written from a lookup table (exact).
* The scene parser is analytic (no training). The EfficientNet-B0 branch is trained on
  T4x2 with mirror augmentation and only handles the ~5 % of frames where a colour pair breaks.
* Source links: <https://www.kaggle.com/code/shawnjuyangke/robovision-3-0-deterministic-solver>
  and <https://www.kaggle.com/datasets/shawnjuyangke/robo-vision-solution-package>
