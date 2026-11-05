# Kaggle 开箱即用：RoboVision Challenge

**入口**：[竞赛主页](https://www.kaggle.com/competitions/robo-vision-challenge-data-hack-3-0) · [现有 Notebook](https://www.kaggle.com/code/shawnjuyangke/robovision-3-0-deterministic-solver)

## 在线运行

1. 登录 Kaggle，加入 RoboVision Challenge 并接受规则（若仍提供数据访问）。
2. 打开上面的 Notebook，点击 **Copy & Edit**。若 Notebook 权限无法访问，请从项目原始压缩包导入 `robo_vision_solver_v3.ipynb`。
3. 在 **Add Input** 挂载竞赛数据，核对 `Challenge_Dataset` 下是否存在 `train.csv`、`test.csv`、`sample_submission.csv` 和场景图像/相机参数。
4. **Settings → Accelerator** 选择可用 GPU（原始训练使用 2×T4；实际选项依账号配额而异）。ImageNet 预训练权重可能需要 Internet，禁网时须先挂载权重 Dataset。
5. 点击 **Run All**。训练回退网络可能耗时；如只复现规则解析，应根据 Notebook 的实际单元选择对应执行路径，勿误以为所有路径都无需训练。
6. 完成后检查 `/kaggle/working/submission.csv`，确认文件存在且格式正确，再回到比赛提交页面提交。

## CSV 格式

CSV 需要 `image_id,action_0,...,action_5` 共 7 列；每个动作单元是四个连续坐标值构成的字符串。

```python
import pandas as pd
sub = pd.read_csv("/kaggle/working/submission.csv")
assert sub.columns.tolist() == ["image_id"] + [f"action_{i}" for i in range(6)]
assert sub["image_id"].notna().all()
print("rows:", len(sub))
print(sub.head(2).to_string(index=False))
```

原测试集记录为 1000 条，但不同版本数据需以当前官方 `sample_submission.csv` 为准。

## Kaggle CLI（可选）

```bash
python -m pip install kaggle
# 先按照官方文档配置 Kaggle API 凭证，勿提交 token 到 GitHub
kaggle kernels status YOUR_USERNAME/YOUR_NOTEBOOK_SLUG
kaggle kernels output YOUR_USERNAME/YOUR_NOTEBOOK_SLUG -p kaggle_output
kaggle competitions submit \
  -c robo-vision-challenge-data-hack-3-0 \
  -f kaggle_output/submission.csv \
  -m "hybrid scene solver"
```

Notebook 推送需要 `kernel-metadata.json` 中 `id` 与 `code_file` 对应你的 Kaggle 帐号和文件。双 T4 / Internet 权限应在 UI 中核对。比赛截止后提交按钮可能不可用。

## 运行注意

- 规则 + 几何依赖官方相机矩阵和数据集生成规律；**不是通用真实机械臂抓取代码**。
- 训练脚本中的辅助缓存、Windows 路径可能要手动改成 `/kaggle/input`。
- 数据集、预训练模型权重和 GPU 环境未随本仓库打包；本文没有重新在线训练。
- **历史 Public LB 0.87702 不能当成当前复现实验的保证分数**。
