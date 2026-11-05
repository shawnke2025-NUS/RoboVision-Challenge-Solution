# RoboVision Challenge — DataHack 3.0 · 方案与复现说明

> 赛题：<https://www.kaggle.com/competitions/robo-vision-challenge-data-hack-3-0>
> 训练环境：Kaggle Notebook，**T4 × 2**
> **Public LB：0.87702（榜上第 1）** — 相比第一版 CNN 方案的 0.44662 提升 96%
> （notebook 最新一版再加两轮检测兜底，训练集估计 0.8641，见第 5 节）

---

## 1. 任务与数据

| 项目 | 内容 |
|---|---|
| 数据量 | train 5000 张，test 1000 张 |
| 图像 | 640 × 480 光线追踪渲染的桌面场景（彩色方块 + 圆碗，深色桌面） |
| 相机 | 每张图一个 `intrinsics` (3×3) 与 `extrinsics` (4×4)，另附 `camera_position` / `target_position` / `up_vector` |
| 提交 | 6 个动作，每个动作是 **`x_pick,y_pick,x_place,y_place`**（米） |
| 指标 | `score = exp(−(d_pick + d_place)² / 0.05²)`，逐动作平均，满分 1.0 |
| 干扰项 | `complexity_score`、`centroid_x` 等 15 个衍生列与标签零相关（\|r\| < 0.05），已忽略 |

### 指标的关键含义

赛题使用的是一个**高斯加权的命中率**（容差 5 cm）。这个形式是**从数据反推、并用线上分数验证**的，
不是猜的：

* `train.csv` 里第 3/4 列的取值范围落在 `[-0.3, 0.3]` / `[-0.8, -0.2]`，且每 `(task_type, step)`
  只有一个取值 —— 与"place 坐标是对角线/三角形固定槽位"的图完全吻合，说明 4 个数是
  `x_pick, y_pick, x_place, y_place`；
* 用三次线上提交反解：逐 (task,step) 均值 → 0.00189，近常数预测 → 0.02944，
  有结构但精度差的预测 → 0.44662。这三个点只有"绝对距离 + 5 cm 高斯核"能同时解释
  （详见下面的换算表），RMSE 型或相关性型指标都对不上。

```
distance = √((x_pick_pred − x_pick_true)² + (y_pick_pred − y_pick_true)²)
         + √((x_place_pred − x_place_true)² + (y_place_pred − y_place_true)²)
score    = exp(−(distance / 0.05)²)
```

换算关系：误差 1 cm → 0.96，2 cm → 0.85，3 cm → 0.70，**5 cm → 0.37，10 cm → 0.018**。
**这个指标对绝对误差极其敏感，没有任何尺度/偏移不变性**，所以：
* 不是"回归得差不多"就行，而是**要么命中、要么归零**；
* 决策重点是**选对物体**（选错 = 误差 15–30 cm = 0 分），而不是微调精度。

另外：`action_0-2` 是**圆碗**（权重 60%，**顺序敏感**，顺序错扣 30%/35%），
`action_3-5` 是**方块**（权重 40%，Hungarian 匹配，**顺序无关**）。

---

## 2. 两个决定性发现

### 发现 1：24 个数字里有 12 个是免费的

`(x_place, y_place)` 在 `(task_type, step)` 的每种组合下**只有一个取值**（5000 行训练数据零反例）：

| task_type | step → (x_place, y_place) |
|---|---|
| task1 | `(-0.2,-0.3) (0.0,-0.5) (0.2,-0.7)` 循环两次 —— "对角线" |
| task2 | `(0.0,-0.5) (0.18,-0.32) (-0.18,-0.32)` 循环两次 —— "三角形" |

直接查表即精确命中，且 place 项在指标里是**第二项平方根**，命中即得满分项。

### 发现 2：场景结构完全可解（100% 在训练集上验证）

每个场景固定 **10 个物体 = 3 组"同色物体对" + 4 个独立颜色**：

* **3 组同色对才是目标**（共 6 个物体），另外 4 个是干扰项；
* 每组同色对里一个是**方块**、一个是**圆碗**；
* 三组按颜色亮度排序（`color_intensity_reference.csv` 的 Rec.601 亮度），
  `task1` 升序、`task2` 降序；
* 圆碗 → `action_0-2`，方块 → `action_3-5`。

这条规则把"训练 CNN"变成了"写确定性解析器"，误差从 ~13 cm 直接降到 **1.3–2 cm**。

---

## 3. 方案

![算法流程图](algorithm_flowchart.png)

*图 1：v3 流水线。(b)–(d) 是确定性解析，单帧约 0.14 s；(e) 的神经网络只在同色对配对失败时启用。
矢量版见 [`algorithm_flowchart.pdf`](algorithm_flowchart.pdf)（7.20 × 3.64 in，可直接用于双栏论文的 `figure*`）。*

```
640×480 图像 ──► 背景剔除（3 个纯色背景）──► 连通域
             ──► 按参考反照率把粘连的物体拆开
             ──► 按色度距离 < 0.03 配对           → 恰好 3 组
             ──► 方块/圆碗：高宽比 vs 相机仰角模型（97% 准确）
             ──► 组排序：颜色亮度（升/降序）
             ──► pick 坐标：底边点回投到桌面平面（再经标定仿射）
             ──► place 坐标：查表（精确）
```

**方块/圆碗判别**用物理模型而不是阈值——高宽比随相机仰角变化：

```
h/w ≈ 0.483·cosθ + 0.773·sinθ + 0.269     (方块)
h/w ≈ 0.565·cosθ + 1.291·sinθ − 0.445     (圆碗)
```

θ 由相机位置与物体位置算出，取残差更小的模型。在 290 组"确定无疑"的样本上**逐物体准确率 97.1%**。

**位置标定**：`pick = backproject(bbox_bottom − 0.39·height, plane z=0.01)`，再绕桌面中心 `(0,−0.5)` 缩放 1.29
（三个常数都在训练集上拟合，见 `work/fit_pos3.py`）。

**EfficientNet-B0 兜底**：约 9% 的帧因为漏检一个物体导致同色对被破坏，解析器无解。
此时用 ImageNet 预训练的 EfficientNet-B0（BEV 正视图 + **镜像增广**，T4×2 训练）给出预测。
镜像增广是标签保持的：BEV 水平翻转 + 世界坐标 `x' = X0 + X1 − x` 同步镜像，
因为规则只依赖颜色与形状，与左右无关。

**为什么不用 YOLO？** 场景是纯合成渲染，物体颜色就是已知的 11 个参考反照率，
"背景剔除 + 反照率拟合"比通用检测器更准也更稳（无预训练域差异）；YOLO 若要可用，
得先用本解析器批量生成伪标签再训（见第 8 节）。

---

## 4. 训练配置

| 项 | 取值 |
|---|---|
| 解析器 | 纯 numpy，无训练，1000 张测试图约 4 分钟（CPU） |
| 兜底 backbone | `torchvision` EfficientNet-B0，ImageNet 预训练 |
| 输入 | BEV 4×256×288（RGB + task 通道），由 `K`/`[R\|t]` 反投影得到 |
| Head | stride-2 的 6 通道热力图 + 全局回归头 |
| Loss | focal heat-map + 8·L1(soft-argmax 世界坐标) + 0.15·L1(回归头) |
| 增广 | 亮度/通道抖动 + **镜像增广**（BEV 翻转 + 标签镜像） |
| 训练 | 25 epoch，AdamW + OneCycle，AMP，`DataParallel` 双 T4 |
| 耗时 | BEV 预处理 ~4 min，CNN ~3 min，解析器 ~5 min，总计 ~15 min |

---

## 5. 结果（Kaggle 实测）

| 提交 | 内容 | Public LB |
|---|---|---|
| **解析器 v2（宽松配对 + 二次检测）** | 纯解析器，未解析帧回退均值 | **0.87702（第 1）** |
| notebook 输出 v1（解析器 + EfficientNet 兜底） | 两段式 | 0.87254 |
| 解析器 v1 | 纯解析器 | 0.86025 |
| 纯 CNN，252 epoch | 第一版（v1） | 0.44662 |
| 纯 CNN，35 epoch 早停 | 第一版（v2） | 0.02944 |
| 逐 (task,step) 均值 | baseline | 0.00189 |

> 每一步的组成与增益都清晰：解析器把分数从 0.447 拉到 0.860，放宽配对 + 二次检测再加到 0.877，
> EfficientNet 兜底在 8.6% 的解析失败帧上再加 +0.012。
> notebook 最新一版同时包含这三项，训练集估计 0.8641（对应线上约 0.89），
> 其产出文件见 `submission_v3_final.csv`。

### 分项验证（训练集前 400 张）

| 类别 | 占比 | 指标 | 3 cm 内命中率 |
|---|---|---|---|
| 解析器完整解出（3 组同色对） | 94.5% | **0.8876** | 95.1% |
| 只解出 2 组（部分预测） | 5.3% | 0.383 | 41.3% |
| 完全失败（回退均值） | 0.3% | 0.001 | 0% |
| **合计（v3：两轮检测 + dmax 0.03）** | | **0.8589** | 92.0% |
| 合计（v2：单轮检测 + dmax 0.014） | 90.5% | 0.8386 | 89.8% |

线上验证：v2 → 0.86025，v2+宽松配对+二次检测 → **0.87702**，与训练集估计的差值稳定。

### 解析率是怎么提上去的

| 改动 | 完整解析 | 训练集指标 |
|---|---|---|
| 初版（dmax=0.014，单轮检测） | 362/400 = 90.5% | 0.8048 |
| 配对阈值放宽到 0.03（不同颜色色度距离 ≥0.15，不会误配） | 373/400 = 93.3% | 0.8296 |
| + 失败帧再用两档更宽松的分割重试 | **378/400 = 94.5%** | **0.8589** |
| 再放到测试集 | **948/1000 = 94.8%** | 0.8641 |

---

## 6. 复现

### 6.1 Kaggle（推荐）

* Notebook：<https://www.kaggle.com/code/shawnjuyangke/robovision-3-0-deterministic-solver>
* 设置：Accelerator = **GPU T4 × 2**，Internet 需开启（下载 EfficientNet 预训练权重；关掉也能跑，会自动退回从头训练）
* Run All → `/kaggle/working/submission.csv` → `kaggle competitions submit -c robo-vision-challenge-data-hack-3-0 -f submission.csv -m "solver"`

### 6.2 本地

```bash
python -m pip install numpy pandas pillow torch torchvision
jupyter notebook robo_vision_solver_v3.ipynb
```

Notebook 自动定位数据目录（`/kaggle/input/*/Challenge_Dataset` 或同级 `Challenge_Dataset/`）。

---

## 7. 文件清单

| 文件 | 说明 |
|---|---|
| `robo_vision_solver_v3.ipynb` | **主 notebook**：解析器 + 规则 + EfficientNet 兜底 + 提交 |
| `robo_vision_solution.ipynb` | 第一版纯 CNN 方案（保留作为对照，LB 0.44662） |
| `README.md` | 本文件 |
| `submission.csv` | 线上 0.87702 的提交文件 |
| `submission_v3_final.csv` | 最新 notebook 输出（训练集估计 0.8641） |
| `submission_solver.csv` | 解析器 v2 提交（0.87702） |
| `submission_baseline.csv` | 均值基线（0.00189） |
| `kaggle_output_v3/` | Kaggle 运行产物 + 训练日志 |
| `work/scene3.py` | 解析器（背景剔除 / 颜色拆分 / 连通域） |
| `work/final_solver.py` | 规则引擎 + 标定常数 + 提交生成 |
| `work/fit_pos3.py`, `work/fit_shape.py` | 位置与形状模型的拟合/验证脚本 |
| `work/draw_flowchart.py` | 生成算法流程图的脚本（matplotlib，矢量输出） |
| `work/nb_builder3.py` | 由源码块生成 notebook 的脚本 |

---

## 8. 还能更好的地方

1. **把解析率从 91% 提到 98%+（收益最大，预计 0.86 → 0.94）**。
   失败都是同一个原因：10 个物体里漏检 1 个，同色对被破坏。
   可以做的：用本解析器在 5000 张训练图上生成**伪标签**，训练一个 YOLO/RT-DETR 检测器，
   推理时与解析器交叉验证；或用第二个阈值的检测结果做增量补漏。
2. **部分预测更聪明**：只解出 2 组时，若能推断缺失颜色在亮度序中的位置，可把 4 个已知目标的
   slot 放对，指标可从 0.368 提到 ~0.6。
3. **剩余 3% 的尾部误差**（>5 cm）主要来自"同色对选错"，可加一层几何一致性校验
   （同色对的两物体应分别落在 block/bowl 的仰角模型里）。
4. 兜底 CNN 目前只贡献 ~0.9% 的分数，若能把 60/40 的加权结构显式建模（碗单独一个头），
   还能再提一点。
