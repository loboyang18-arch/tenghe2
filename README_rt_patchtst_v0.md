# RT PatchTST V0（山东全年 15min，6h ahead）

## 目标
对每个决策时刻 t（15min），预测 y(t)=实时出清电价(t+6h)。

## 数据
- 文件：山东-全年-带时间点.xlsx
- 默认 sheet：第一个 sheet（脚本自动识别）
- 关键列：
  - datetime（时间）
  - 实时出清电价（RT target）
  - 日前出清电价（可选通道）
  - *…实际值（12列）与 *…预测值（10列）作为外生序列通道

V0 默认 **不启用** “日前电价预测值/实时电价预测值”（缺失较多），可通过参数开启。

## 主要设计
- 输入：多通道序列 [C, L]，L=192（48h）默认
- Patch：patch_len=16（4h），stride=8（2h）
- 模型：patchify -> TransformerEncoder -> mean pooling -> context fusion -> 回归头
- Context：决策时刻与目标时刻的 sin/cos 时间特征 + target_hour
- 损失：Weighted Huber
  - 高价阈值：train 标签 85% 分位
  - 加权规则：仅 seg2（target_hour∈[16,24)）且 y>=阈值的样本权重=1.5，其余=1.0

## 切分与防泄露
- 按天顺序切分：最后 6 天 test、之前 5 天 val、其余 train
- scaler 仅用 train 范围拟合（特征行不越过 train 结束）

## 运行
```bash
# 默认使用当前目录下的 "山东-全年-带时间点.xlsx"
python rt_patchtst_v0.py

# 或指定其他路径
python rt_patchtst_v0.py --excel "山东-全年-带时间点.xlsx"
```

常用调参：
```bash
python rt_patchtst_v0.py --lookback 192 --patch_len 16 --stride 8 --epochs 50 --batch 256 --lr 1e-3
python rt_patchtst_v0.py --include_price_forecast
```

## 输出
- checkpoints/rt_patchtst_v0_best.pt
- outputs/rt_patchtst_v0_metrics.json
- outputs/rt_patchtst_v0_test_predictions.csv
