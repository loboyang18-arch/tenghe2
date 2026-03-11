#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_best_exog_suite.py

一键运行“当前最佳外生预测方案”，并自动做回测与绘图。

依赖现有脚本 exog_forecaster_v0_4_1_clean.py 中的实现，不重复造轮子：
- 使用同样的样本构造、时间切分（按 target_max_date）、回测逻辑
- 自动输出：
  - 每个 target / method 的 metrics.json
  - 每个 target / method 的 test_horizon_mae.png / test_last7d_h24.png
  - 每个 target / method 的 test_predictions_long.csv
  - 汇总表 best_suite_summary.csv

当前“最佳方案”映射（根据现有 results 总结）：
- 系统负荷实际值: hgbt_residual
- 光伏实际值    : hgbt_residual
- 联络线实际值  : use_pred
- 上旋备用实际值: hgbt_pure
- 下旋备用实际值: hgbt_pure
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import pandas as pd

from exog_forecaster_v0_4_1_clean import (  # type: ignore
    Cfg,
    eval_one_target,
    ensure_dir,
    read_excel_any,
    setup_matplotlib_fonts,
)


# 固定的“当前最佳模型”映射表
BEST_METHODS: Dict[str, str] = {
    "系统负荷实际值": "hgbt_residual",
    "光伏实际值": "hgbt_residual",
    "联络线实际值": "use_pred",
    "上旋备用实际值": "hgbt_pure",
    "下旋备用实际值": "hgbt_pure",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="一键运行当前最佳外生预测方案（基于 exog_forecaster_v0_4_1_clean）",
    )
    p.add_argument("--excel", type=str, required=True, help="源 Excel 文件路径，例如: 山东-全年-带时间点.xlsx")
    p.add_argument("--sheet", type=str, default=None, help="Sheet 名称或索引，默认使用第一个 sheet")
    p.add_argument("--time_col", type=str, default="datetime")
    p.add_argument(
        "--step_minutes",
        type=int,
        default=15,
        help="时间粒度（分钟），默认 15",
    )
    p.add_argument(
        "--lookback",
        type=int,
        default=192,
        help="历史窗口长度（步数），默认 192（即 48h@15min）",
    )
    p.add_argument(
        "--H",
        type=int,
        default=24,
        help="预测步数（horizon），默认 24（即 6h@15min）",
    )
    p.add_argument("--val_days", type=int, default=5, help="验证天数，默认 5")
    p.add_argument("--test_days", type=int, default=6, help="测试天数，默认 6")
    p.add_argument(
        "--feat_nan_strategy",
        type=str,
        default="ffill_bfill",
        choices=["none", "ffill", "ffill_bfill"],
        help="特征缺失值处理策略（仅作用在特征列，不动标签），默认 ffill_bfill",
    )
    p.add_argument(
        "--ridge_alpha",
        type=float,
        default=10.0,
        help="Ridge 正则系数（仅当用到 ridge_* 时有效）",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="outputs",
        help="输出根目录，默认 outputs",
    )
    p.add_argument(
        "--tag",
        type=str,
        default="exog_best_current",
        help="子目录 tag，结果会写到 outputs/<tag>/ 下",
    )
    return p.parse_args()


def build_cfg(args: argparse.Namespace, models: List[str]) -> Cfg:
    """构造 exog_forecaster_v0_4_1_clean.Cfg，模型集合由调用方决定。"""
    return Cfg(
        excel=args.excel,
        sheet=args.sheet,
        time_col=args.time_col,
        target_cols=list(BEST_METHODS.keys()),
        step_minutes=int(args.step_minutes),
        lookback=int(args.lookback),
        H=int(args.H),
        val_days=int(args.val_days),
        test_days=int(args.test_days),
        feat_nan_strategy=str(args.feat_nan_strategy),
        ridge_alpha=float(args.ridge_alpha),
        models=models,
        out_dir=str(args.out_dir),
        tag=str(args.tag),
    )


def main() -> None:
    setup_matplotlib_fonts()
    args = parse_args()

    print("[BestSuite] 运行当前最佳外生预测方案")
    print(f"[Config] excel={args.excel} sheet={args.sheet} time_col={args.time_col}")
    print(
        f"[Config] lookback={args.lookback} H={args.H} step={args.step_minutes}min "
        f"val_days={args.val_days} test_days={args.test_days}",
    )
    print(
        f"[Config] feat_nan_strategy={args.feat_nan_strategy} ridge_alpha={args.ridge_alpha}",
    )
    print(f"[Out] {os.path.join(args.out_dir, args.tag)}")
    print(f"[BestMethods] {BEST_METHODS}")

    # 读取数据
    df = read_excel_any(args.excel, args.sheet)
    print(f"[Excel] loaded rows={len(df)} cols={len(df.columns)}")
    if args.time_col not in df.columns:
        raise ValueError(f"time_col not found: {args.time_col}")

    ensure_dir(os.path.join(args.out_dir, args.tag))

    # 为了复用 eval_one_target，我们分 target 依次调用，并每次只给对应的“最佳模型”
    all_rows: List[Dict] = []
    for target_col, best_method in BEST_METHODS.items():
        if target_col not in df.columns:
            print(f"[Skip] 目标列不存在，跳过: {target_col}")
            continue

        print(
            f"[Run] target={target_col}  best_method={best_method} "
            f"(单目标回测+绘图)",
        )
        # 构造只包含当前 best_method 的 Cfg
        cfg = build_cfg(args, models=[best_method])
        rows = eval_one_target(cfg, df, target_col)
        all_rows.extend(rows)

    # 保存整体汇总
    if all_rows:
        summary = pd.DataFrame(all_rows)
        summary_path = os.path.join(args.out_dir, args.tag, "best_suite_summary.csv")
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"[Saved] best suite summary: {summary_path}")
    else:
        print("[Done] 没有任何结果产生（可能是目标列缺失或样本不足）。")


if __name__ == "__main__":
    main()

