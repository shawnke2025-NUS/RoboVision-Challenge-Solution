---
layout: post
title: "RoboVision Kaggle 复盘：规则场景解析、相机几何与 EfficientNet 兜底"
date: 2026-11-05
categories: [Kaggle, Computer-Vision, Robotics]
tags: [RoboVision, Geometry, Scene-Parsing, EfficientNet]
---

> [竞赛](https://www.kaggle.com/competitions/robo-vision-challenge-data-hack-3-0) · [Kaggle Notebook](https://www.kaggle.com/code/shawnjuyangke/robovision-3-0-deterministic-solver) · [复现指南]({{ '/KAGGLE_REPRODUCE' | relative_url }})  
> 本文基于原始工程包与运行日志复盘；**0.87702 是上传记录中的历史 Public LB，而非实时排行榜验证结果**。

## TL;DR

本题输入合成桌面图像和相机参数，为六个目标生成抓取/放置动作。最初采用纯 CNN 回归，上传记录中的 Public LB 为 **0.44662**；利用 **背景分割 → 连通域 → 颜色配对 → 方块/碗判别 → 顺序规则 → 相机反投影 → 标定** 后，规则解达到 **0.86025**，改进版本达到 **0.87702**。另一条「规则 + EfficientNet-B0 兜底」历史版本记录为 **0.87254**。注意这不是连续叠加的消融结果，不能声称网络进一步把 0.87702 提高了。

![算法结构图]({{ '/assets/solver_pipeline.svg' | relative_url }})

**关键洞察**：这道题并非完全自由的 image-to-action 回归，而是有强场景生成先验的结构化问题。先解离散的物体配对与槽位分配，再解连续的世界坐标预测，比直接回归 24 个数更容易控制误差。

## 1. 任务与数据结构

每帧有十个物体：**三组同颜色的方块/圆碗目标对，共六个目标；四个颜色独特的干扰物**。六步动作中，**action_0～2 为 bowl，action_3～5 为 block**。三对目标按参考颜色亮度排序，排序方向依赖 task_type（task1 暗到亮、task2 亮到暗）。亮度近似公式：

\[
Y = 0.299 R + 0.587 G + 0.114 B.
\]

训练集记录有 5000 个场景、测试集 1000 个场景，图像大小 640×480；相机内参为 3×3 矩阵，外参为 4×4 矩阵。每条提交是六个动作，各含 `(x_pick, y_pick, x_place, y_place)` 四个值（单位米）。

通过逐组检查训练标签，发现放置位置可由 **(task_type, step)** 查表确定，因此重点转向六个抓取坐标。**这是数据集经验规律，不是一般机器人任务都成立的前提**。

## 2. 背景分割和实例恢复

从图像中移除少量固定背景色，例如 RGB 近似 `(47,47,47)`、`(162,186,224)`、`(239,239,239)`，利用 RGB 距离、灰度约束和连通域得到物体区域。对于靠得很近的两个物体，单纯连通域可能合并，需要按颜色再分离。

假设参考颜色反照率为 `a_k`，观察像素为 `p`，先拟合亮度缩放：

\[
\alpha_k=\max\left(0.05,\frac{p^\top a_k}{a_k^\top a_k}\right),\qquad
k^*=\arg\min_k\|p-\alpha_k a_k\|_2.
\]

这样可降低阴影对颜色分类的干扰；之后按类别切开粘连区域、过滤碎片并合并邻近小块。每个候选物体记录中心、边界框、面积、宽高比、参考颜色及归一化色度向量。

## 3. 颜色配对和动作槽位

把物体平均色归一化为色度：

\[
c=(R,G,B)/(R+G+B+\varepsilon).
\]

枚举两两色度距离，按距离从小到大贪心挑选不共享物体的配对，典型阈值 **d < 0.03**。每对必须恰有一个 bowl 和一个 block。按颜色亮度为三组分配槽位，前三步取 bowls，后三步取 blocks。

```python
candidate_pairs = sorted(
    (np.linalg.norm(obj[i]["chroma"] - obj[j]["chroma"]), i, j)
    for i in range(len(obj)) for j in range(i + 1, len(obj))
)
used, pairs = set(), []
for dist, i, j in candidate_pairs:
    if dist > 0.03:
        break
    if i not in used and j not in used:
        pairs.append((i, j))
        used.update((i, j))
```

漏检时会以较宽松的面积和填充率阈值重试。日志记录在 1000 张测试图中：**948 张找到完整三对，48 张找到两对，4 张仅找到一对**。这是 *规则解析完整率 94.8%*，**不是 94.8% 抓取准确率**。

## 4. 方块 vs 圆碗：使用相机视角

同一个物体在不同拍摄仰角下的框宽高比会改变。相比固定 `h/w` 阈值，代码依据相机与物体的相对位置估计俯仰角 `θ`，然后与经验拟合曲线比较：

\[
f_{block}(\theta)=0.483\cos\theta+0.773\sin\theta+0.269,
\]
\[
f_{bowl}(\theta)=0.565\cos\theta+1.291\sin\theta-0.445.
\]

标准化误差分别除以约 `0.093` 和 `0.046` 后选较小者；若同色配对两端判断成相同类型，还会按相对宽高比强制拆分为 bowl/block。这是利用**组合约束修正局部感知不确定性**。

## 5. 像素到米：逆投影和标定

对于像素 `(u,v)`，构造光线：

\[
d_c=K^{-1}[u,v,1]^\top,\quad
d_w=R^\top d_c,\quad
P=t+\lambda d_w,\quad
\lambda=\frac{z_0-t_z}{(d_w)_z}.
\]

外参的 `R,t` 使用方式取决于其是 world-to-camera 还是 camera-to-world；须与代码中的相机约定保持一致。实现用 **bbox 下缘附近的基点** 而非中心点：`u=cx`，`v=y1-0.39h`，并分别设置 bowl/block 的投影平面高度。随后对桌面中心 `C=(0,-0.5)` 做经验尺度校准：

\[
p_{pick}=C+1.29(p-C).
\]

`0.39`、`1.29` 是训练数据拟合得到的常数，不应视为普适的物理标定。原工程保留了仿射校正结构，但当前起实际作用的是中心缩放，并非已学习一套完整仿射矩阵。

## 6. Learned fallback：仅处理规则找不到的目标

解析缺失颜色对或出现遮挡时，使用预训练 **EfficientNet-B0** 的 BEV 回归网络补全：利用相机参数将桌面映射到鸟瞰图，经过编码器和 U-Net 式解码器，输出六通道热力图；使用 **soft-argmax** 把热力图映射为米坐标，辅以全局回归头。

\[
\mathcal L=\mathcal L_{focal}+8\mathcal L_{1,softargmax}+0.15\mathcal L_{1,regression}.
\]

原始训练配置为 AdamW、OneCycleLR、AMP、25 epochs、batch 48、双 T4；包含镜像与亮度扰动，镜像时同步调整世界坐标。**一个需注意的代码问题**：虽然数据集构造了 RGB + task_type 四通道输入，模型前向却使用 `x[:, :3]`，第四通道实际未进入主干网络。想做任务条件化需要修改首层并重新训练验证。

这条网络仅做异常场景的补偿。现有工程没有提供在完全相同评估条件下足以证明「最终融合必定优于纯规则 0.87702」的证据。

## 7. 项目记录中的性能

| 版本 | Public LB | 说明 |
| --- | ---: | --- |
| 仅任务步骤均值 | 0.00189 | 不看视觉 |
| 纯 CNN baseline | 0.44662 | 252 epochs 版本 |
| 规则解析 v1 | 0.86025 | 几何 + 场景先验 |
| 规则 + CNN 历史版本 | 0.87254 | 对残缺解析做回退 |
| **改进规则解析器** | **0.87702** | 配对/检测重试优化 |

**如何解读**：主要增益来自发现结构先验，而不是模型规模。日志还有训练子集解析分数约 0.8641、3 cm 内命中率 0.924 和中位定位误差 1.3 cm；这些统计口径与线上 Public LB 不同，不能混用。

## 8. Kaggle 开箱复现

详见 **[完整 Kaggle 使用说明]({{ '/KAGGLE_REPRODUCE' | relative_url }})**。基本步骤是加入竞赛、克隆 [原 Kaggle Notebook](https://www.kaggle.com/code/shawnjuyangke/robovision-3-0-deterministic-solver)、挂载竞赛数据，在可用的 GPU 环境下 Run All，检查 `/kaggle/working/submission.csv`，然后提交竞赛。赛题数据、GPU、联网和比赛截止日期都需要在 Kaggle 页面再次确认。

## 9. 局限与下一步

1. 把简单贪心配对换成 **颜色一致性 + 形状互补 + 任务排序** 的全局最小代价匹配。
2. 为解析器引入**置信度**，对低信心目标触发 CNN，而不是只判断是否找到三对。
3. 让网络真正使用 task_type，并做固定 split 下的 **rule-only / CNN-only / fusion** 消融。
4. 几何参数在合成分布上拟合，改变物体尺寸、相机姿态或真实环境时需要重新标定。
5. 部分历史辅助脚本依赖未打包的缓存和本地绝对路径；建议以原 Kaggle 主 Notebook 为复现入口。

## 总结

**最值得复用的是分析范式**：先分析输出规律和场景约束，把能解析的部分交给确定性几何与规则，神经网络只处理剩余不确定性。对于生成机制稳定的合成视觉任务，这种 hybrid solver 往往比盲目的端到端回归更可解释、更省推理成本。

---

**链接**：[Kaggle 竞赛](https://www.kaggle.com/competitions/robo-vision-challenge-data-hack-3-0) · [原 Notebook](https://www.kaggle.com/code/shawnjuyangke/robovision-3-0-deterministic-solver) · [本仓库](https://github.com/shawnke2025-NUS/RoboVision-Challenge-Solution)
