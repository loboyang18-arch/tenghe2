# RT 预测改进路线图

基于当前实验结论（baseline 稳但不如 prior、exog5(actual) 无效且偏置大），按三阶段推进，**优先把信息流改对**。

---

## 一、当前结论摘要

| 版本 | 现象 | 判断 |
|------|------|------|
| **prior** | TEST MAE ≈ 49.30，强基线 | 短时惯性大，作为业务参照 |
| **baseline** | 能训练，TEST MAE ≈ 72.27，高价段严重低估 | 稳但抓不住 spike |
| **exog5(actual)** | TEST MAE ≈ 81.34，Bias +60，Lead24 更偏 | 失败方向，不继续投入 |

**本质**：不是模型训坏了，而是**信息流设计**有问题——模型预测未来 24 步，但只看到过去/当前 actual，没有“未来 6 小时驱动信息”。

---

## 二、第一阶段：把信息流改对（最优先）

### 方案 1：接入未来 24 步 exog5 预测轨迹（主线）

- **目标**：用 exog_forecaster 输出的未来关键外生序列作为 RT 模型的**未来分支**输入。
- **正确链路**：exog 预测 5 条（系统负荷、光伏、联络线、上旋备用、下旋备用）未来 24 步 → 构造 RT 样本时 **past branch**（历史价格+历史外生）+ **future branch**（未来 24 步 exog 预测）→ 模型输出未来 24 步 RT 价格。
- **实现**：`scripts/run_rt_future_exog.py`（入口）、`src/future_exog.py`（未来 exog 预计算与对齐）、`src/models/patchtst_dual.py`（双分支 + concat head）、`src/datasets.RTDualBranchDataset`、`src/train/rt_train` 中 `train_one_epoch_dual` / `eval_model_dual` / `predict_split_dual`。

### 方案 2：折中版特征（baseline + exog5 + 少量代理变量）

- 保留 baseline 特征，加 exog5，再补：竞价空间实际值、风光总加实际值、实际价差方向、实际价差、相似时刻 Top1 等。
- **目的**：区分是“关键外生本身没用”还是“去掉代理后信息不够”。

### 方案 3：双分支建模（与方案 1 一致）

- **Branch A**：历史 RT + DA + 历史 exog → 学习惯性、局部时序。
- **Branch B**：未来 24 步 exog 预测 → 提供未来驱动。
- **融合**：concat 后 MLP head（初期推荐）；后续可试 cross-attention。

---

## 三、第二阶段：增强高价时段能力

- **方案 4**：高价样本加权（P85/P95 权重 1.5/2.0）。
- **方案 5**：Huber / MAE + 高价加权，减少 MSE 的过度平滑。
- **方案 6**：spike 辅助分类头（是否出现 >P85 / >P95），L = L_reg + λ*L_cls。

---

## 四、第三阶段：训练与评估机制

- **方案 7**：early stop 用业务指标（raw-space val MAE、Lead1/Lead24、High>=P85）。
- **方案 8**：分段 head 或 horizon-aware loss（1–8 / 9–16 / 17–24）。
- **方案 9**：严格消融顺序（prior → baseline → +exog5(actual) → +exog5(pred) → +高价加权 → +代理变量 → 双分支等）。

---

## 五、推荐执行顺序

1. **第一步**：baseline + **predicted exog5**（未来 24 步接入）。
2. **第二步**：在该版本上加**高价加权 loss**。
3. **第三步**：对照 baseline + exog5(pred) + **少量代理变量**。
4. **第四步**：若有提升，再考虑双分支细化、分段 head、spike 辅助头。

---

## 六、版本定位

- **prior**：强业务基线，主要参照物。
- **baseline**：最稳学习型基线，后续改进均与之对比。
- **exog5(actual)**：已验证失败方向，不再投入。
- **下一主线**：**future exog 驱动的 RT 模型**（方案 1 + 双分支）。

---

## 七、一句话总结

> 问题不是 PatchTST 不会预测，而是模型还没有拿到真正对应“未来 6 小时价格变化”的有效未来信息。  
> 下一步重点：**未来关键外生轨迹 + 高价敏感训练机制** 接进 RT 主线。
