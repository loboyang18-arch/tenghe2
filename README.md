# 腾河：外生预测与实时电价预测

本项目包含两类最新版脚本，用于在云服务器或本机复现实验：

1. **外生序列预测**：对 5 条关键外生量（系统负荷、光伏、联络线、上/下旋备用）做未来 6 小时轨迹预测与回测。
2. **实时电价预测**：PatchTST 序列模型，预测未来 24 步（6 小时）的实时出清电价；提供 baseline 与「仅用 5 条外生」的改进版，便于对比。

---

## 环境要求

- **Python**：3.9 或 3.10 推荐（3.12 亦可）。
- **依赖**：见 `requirements.txt`。主要包含 `pandas`、`numpy`、`scikit-learn`、`matplotlib`、`openpyxl`、`torch`。

### 设备：本地 CPU / 云上 GPU

- **外生预测**（`run_best_exog_suite.py`、`exog_forecaster_v0_4_1_clean.py`）仅用 sklearn/ pandas，**不使用 GPU**。
- **实时电价**（`rt_patchtst_baseline_v0.py`、`rt_patchtst_exog5_v1.py`）使用 PyTorch，**已按环境自动选择设备**：
  - **本机无 GPU**：自动使用 `cpu`，无需改代码。
  - **云上有 GPU**：自动使用 `cuda`（需在云上安装带 CUDA 的 PyTorch，见下）。
  - 也可通过参数强制指定：`--device cuda` 或 `--device cpu`。

### 本机安装（无 GPU，CPU 即可）

```bash
pip install -r requirements.txt
# 或 conda：
conda install pytorch cpuonly -c pytorch -y
pip install -r requirements.txt
```

### 云上安装（使用 GPU）

在带 NVIDIA GPU 的云服务器上，安装 **带 CUDA 的 PyTorch**，其余依赖同 `requirements.txt`。例如（按你云环境的 CUDA 版本选一个）：

```bash
# Conda（推荐，以 CUDA 11.8 为例）
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt

# 或 pip（以 CUDA 12.1 为例，见 https://pytorch.org/get-started/locally/）
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

运行时会打印 `Device: cuda` 即表示在使用 GPU。

---

## 数据准备

- 将 Excel 数据文件放在项目根目录（或任意路径，通过命令行参数指定）。
- 表格需包含列（名称需一致或通过参数指定）：
  - 时间列：如 `datetime`
  - 实时出清电价：如 `实时出清电价`
  - 可选：日前出清电价、系统负荷实际值、光伏实际值、联络线实际值、上旋备用实际值、下旋备用实际值，以及对应的「XX预测值」列（若做外生预测）
- **不要将含敏感信息的原始数据提交到 Git**；本仓库已通过 `.gitignore` 排除 `*.xlsx` 等。

---

## 仓库中的脚本（仅最新版）

| 脚本 | 用途 |
|------|------|
| `run_best_exog_suite.py` | 一键运行「当前最佳外生预测方案」：5 个目标各自用选定最佳方法（hgbt_residual / use_pred / hgbt_pure），带回测与绘图。 |
| `exog_forecaster_v0_4_1_clean.py` | 外生预测底层实现（被 `run_best_exog_suite.py` 调用）：多种方法、无泄露划分、metrics 与图表输出。 |
| `rt_patchtst_baseline_v0.py` | 实时电价 PatchTST 基准模型：多外生 + 持久化先验 + 残差学习，输出 test 指标与 last7d 图。 |
| `rt_patchtst_exog5_v1.py` | 实时电价改进版：仅使用 5 条关键外生（系统负荷、光伏、联络线、上/下旋备用），其余设定与 baseline 一致，便于对比。 |

历史版本脚本已通过 `.gitignore` 排除，不会随仓库上传。

---

## 运行方式

### 1. 外生预测（最佳方案回测 + 绘图）

```bash
python -u run_best_exog_suite.py \
  --excel "山东-全年-带时间点.xlsx" \
  --time_col "datetime" \
  --step_minutes 15 --lookback 192 --H 24 \
  --val_days 5 --test_days 6 \
  --out_dir "outputs" --tag "exog_best_current"
```

输出目录：`outputs/exog_best_current/`，内含各目标的 metrics、per-horizon MAE 图、last7d 图及预测长表。

### 2. 实时电价 baseline

```bash
python -u rt_patchtst_baseline_v0.py \
  --file "山东-全年-带时间点.xlsx" \
  --outdir "rt_patchtst_v0_1_outputs"
```

默认按时间划分 train/val/test；输出 `rt_patchtst_v0_1_outputs/summary.json`、按 lead 的 CSV、last7d 图及模型权重。

### 3. 实时电价改进版（仅 5 条外生）

```bash
python -u rt_patchtst_exog5_v1.py \
  --file "山东-全年-带时间点.xlsx" \
  --outdir "rt_patchtst_exog5_v1_outputs"
```

输出结构同 baseline，便于直接对比 `summary.json` 中的 test MAE/RMSE、lead1/lead24、高价段等指标。

---

## 云服务器上长时间运行

训练耗时较长时，建议使用 `nohup` 或 `screen`/`tmux`。云上有 GPU 时脚本会自动用 `cuda`，也可显式加上 `--device cuda`：

```bash
nohup python -u rt_patchtst_exog5_v1.py \
  --file "山东-全年-带时间点.xlsx" \
  --outdir "rt_patchtst_exog5_v1_outputs" \
  --device cuda \
  > rt_exog5.log 2>&1 &
```

---

## 输出与结果

- **外生**：`outputs/<tag>/` 下按目标分子目录，每目录含 `metrics.json`、`test_horizon_mae.png`、`test_last7d_h24.png`、`test_predictions_long.csv`；根目录有汇总 CSV。
- **RT 电价**：`<outdir>/summary.json` 含完整配置与 test/val 指标；另有 `test_by_lead_metrics_model.csv`、last7d 图、模型 `.pt` 文件。

上述输出目录已在 `.gitignore` 中排除，不会提交到仓库。
