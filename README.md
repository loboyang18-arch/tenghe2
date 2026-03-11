# 腾河：外生预测与实时电价预测

本项目包含两类最新版脚本，用于在云服务器或本机复现实验：

1. **外生序列预测**：对 5 条关键外生量（系统负荷、光伏、联络线、上/下旋备用）做未来 6 小时轨迹预测与回测。
2. **实时电价预测**：PatchTST 序列模型，预测未来 24 步（6 小时）的实时出清电价；提供 baseline 与「仅用 5 条外生」的改进版，便于对比。

---

## 环境要求

- **Python**：3.9 或 3.10 推荐（3.12 亦可）。
- **依赖**：见 `requirements.txt`。主要包含 `pandas`、`numpy`、`scikit-learn`、`matplotlib`、`openpyxl`、`torch`。

### 设备：本地 CPU / 云上 GPU

- **外生预测**（`scripts/run_exog_suite.py`、`scripts/run_exog_full.py`）仅用 sklearn/pandas，**不使用 GPU**。
- **实时电价**（`scripts/run_rt_baseline.py`、`scripts/run_rt_exog5.py`）使用 PyTorch，**已按环境自动选择设备**：
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

## 数据准备（P3 数据与仓库解耦）

- 仓库**不随代码提交大体积数据**；数据请放在 **`data/`** 目录，或通过 `--file` / `--excel` 指定路径（支持本机路径或 DSW 挂载的 OSS 路径）。
- **列约定**：见 `data/README.md` 与 `data/schema.json`。表格需包含（名称需一致或通过参数指定）：
  - **时间列**：如 `datetime`
  - **实时出清电价**：如 `实时出清电价`
  - **可选**：日前出清电价、系统负荷实际值、光伏实际值、联络线实际值、上旋备用实际值、下旋备用实际值，以及对应的「XX预测值」列（若做外生预测）
- **推荐**：将 Excel/CSV 放入 `data/` 后运行，例如：
  `--file data/山东-全年-带时间点.xlsx`、`--excel data/山东-全年-带时间点.xlsx`

---

## 目录与入口（P2 拆分后）

- **scripts/**：推荐入口，薄脚本，解析参数后调用 `src` 与（可选）`configs`。
  - `run_rt_baseline.py` — 实时电价 baseline（多外生 + 持久化先验 + 残差学习）
  - `run_rt_exog5.py` — 实时电价仅 5 条外生
  - `run_exog_suite.py` — 一键运行当前最佳外生预测方案
  - `run_exog_full.py` — 外生预测完整 CLI（多目标、多模型）
  - `run_rt_future_exog.py` — **ROADMAP 第一步**：双分支（past + 未来 24 步 exog5 预测）
- **src/**：可复用逻辑（数据、特征、划分、数据集、模型、训练、外生方法）。
  - `split.py`（**P4 统一**）：RT 用 `SplitConfig` + `build_samples_seq2seq`；外生用 `ExogSplitConfig` + `split_by_target_max_date`，默认天数 `DEFAULT_EXOG_VAL_DAYS`/`TEST_DAYS`。
  - `features_config.py`（**P5**）：`load_config`、`get_preferred_exog_order`、`get_exog_key5`、`get_exog_best_methods`，可从 YAML/JSON 或内置默认读特征列表。
  - `dataio.py`、`feature_engineering.py`、`datasets.py`
  - `models/patchtst.py`、`train/rt_train.py`、`train/eval.py`
  - `exog/` — 外生预测（Cfg、eval_one_target、read_excel_any 等）
- **configs/**：实验配置（**P5 特征配置化**）：`rt_baseline.yaml`（preferred_exog_order）、`rt_exog5.yaml`（exog_columns）、`exog_key5.yaml`（target_cols、best_methods）。脚本支持 `--config <path>` 从 YAML 读列与 best_methods，避免硬编码。

根目录保留兼容入口（转发到 scripts）：`rt_patchtst_baseline_v0.py`、`rt_patchtst_exog5_v1.py`、`run_best_exog_suite.py`。  
历史实现脚本 `exog_forecaster_v0_4_1_clean.py` 仍可单独运行；等价完整 CLI 为 `scripts/run_exog_full.py`。

---

## 运行方式

**推荐从仓库根目录执行**（以下均可在根目录运行）：

### 1. 外生预测（最佳方案回测 + 绘图）

数据默认使用 `data/山东-全年-带时间点.xlsx`，可不写 `--excel`：

```bash
python -u scripts/run_exog_suite.py \
  --time_col "datetime" \
  --step_minutes 15 --lookback 192 --H 24 \
  --val_days 5 --test_days 6 \
  --out_dir "outputs" --tag "exog_best_current"
```

或指定：`--excel "data/山东-全年-带时间点.xlsx"`。根目录兼容：`python -u run_best_exog_suite.py`。  
输出目录：`outputs/exog_best_current/`，内含各目标的 metrics、per-horizon MAE 图、last7d 图及预测长表。

### 2. 实时电价 baseline

默认 `--file data/山东-全年-带时间点.xlsx`，可直接：

```bash
python -u scripts/run_rt_baseline.py --outdir "rt_patchtst_v0_1_outputs"
```

或根目录：`python -u rt_patchtst_baseline_v0.py`。  
默认按时间划分 train/val/test；输出 `summary.json`、按 lead 的 CSV、last7d 图及模型权重。

### 3. 实时电价改进版（仅 5 条外生）

```bash
python -u scripts/run_rt_exog5.py --outdir "rt_patchtst_exog5_v1_outputs"
```

或根目录：`python -u rt_patchtst_exog5_v1.py`。  
输出结构同 baseline，便于直接对比 `summary.json` 中的 test MAE/RMSE、lead1/lead24、高价段等指标。

### 4. 外生预测完整 CLI（多目标、多模型）

```bash
python -u scripts/run_exog_full.py \
  --target_cols "系统负荷实际值,光伏实际值,联络线实际值,上旋备用实际值,下旋备用实际值" \
  --models "seasonal_median,persist,use_pred,ridge_pure,ridge_residual,hgbt_pure,hgbt_residual" \
  --tag "exog_v0_4_1_clean_key5"
```
（默认 `--excel data/山东-全年-带时间点.xlsx`，可省略。）

### 使用配置文件（P5）

特征列与 best_methods 可从 YAML 读，避免改代码：

```bash
python scripts/run_rt_baseline.py --config configs/rt_baseline.yaml
python scripts/run_rt_exog5.py --config configs/rt_exog5.yaml
python scripts/run_exog_suite.py --config configs/exog_key5.yaml
python scripts/run_exog_full.py --config configs/exog_key5.yaml
```

修改 `configs/*.yaml` 中的 `preferred_exog_order`、`exog_columns`、`target_cols`、`best_methods` 即可换列或换方法。

---

## 云服务器上长时间运行

训练耗时较长时，建议使用 `nohup` 或 `screen`/`tmux`。云上有 GPU 时脚本会自动用 `cuda`，也可显式加上 `--device cuda`：

```bash
nohup python -u scripts/run_rt_exog5.py \
  --outdir "rt_patchtst_exog5_v1_outputs" \
  --device cuda \
  > rt_exog5.log 2>&1 &
```

---

## 代码格式（协作与 review）

仓库内核心脚本已用 **black**（行宽 88）+ **isort** 统一格式，保证 Git diff 可读、便于 DSW 热修复与代码审阅。配置见 `pyproject.toml`。

提交前建议执行：

```bash
pip install black isort -q
black scripts/ src/ run_best_exog_suite.py rt_patchtst_baseline_v0.py rt_patchtst_exog5_v1.py
isort scripts/ src/ run_best_exog_suite.py rt_patchtst_baseline_v0.py rt_patchtst_exog5_v1.py
```

或项目根目录下：`black . && isort .`（会跳过 `rt-env` 等）。

---

## 输出与结果

- **外生**：`outputs/<tag>/` 下按目标分子目录，每目录含 `metrics.json`、`test_horizon_mae.png`、`test_last7d_h24.png`、`test_predictions_long.csv`；根目录有汇总 CSV。
- **RT 电价**：`<outdir>/summary.json` 含完整配置与 test/val 指标；另有 `test_by_lead_metrics_model.csv`、last7d 图、模型 `.pt` 文件。

上述输出目录已在 `.gitignore` 中排除，不会提交到仓库。
