#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键运行“当前最佳外生预测方案”，调用 src.exog。
推荐从仓库根目录运行: python scripts/run_exog_suite.py --excel <path>
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.exog import (
    Cfg,
    ensure_dir,
    eval_one_target,
    read_excel_any,
    setup_matplotlib_fonts,
)
from src.features_config import get_exog_best_methods, get_exog_key5, load_config
from src.split import DEFAULT_EXOG_TEST_DAYS, DEFAULT_EXOG_VAL_DAYS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="一键运行当前最佳外生预测方案（基于 src.exog）",
    )
    p.add_argument(
        "--excel",
        type=str,
        default="data/山东-全年-带时间点.xlsx",
        help="源 Excel 文件路径，默认 data/ 下",
    )
    p.add_argument("--sheet", type=str, default=None)
    p.add_argument("--time_col", type=str, default="datetime")
    p.add_argument("--step_minutes", type=int, default=15)
    p.add_argument("--lookback", type=int, default=192)
    p.add_argument("--H", type=int, default=24)
    p.add_argument(
        "--val_days",
        type=int,
        default=DEFAULT_EXOG_VAL_DAYS,
        help="外生划分 val 天数（默认见 src.split）",
    )
    p.add_argument(
        "--test_days",
        type=int,
        default=DEFAULT_EXOG_TEST_DAYS,
        help="外生划分 test 天数（默认见 src.split）",
    )
    p.add_argument(
        "--feat_nan_strategy",
        type=str,
        default="ffill_bfill",
        choices=["none", "ffill", "ffill_bfill"],
    )
    p.add_argument("--ridge_alpha", type=float, default=10.0)
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--tag", type=str, default="exog_best_current")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML/JSON 配置路径，可选；用于 target_cols 与 best_methods",
    )
    return p.parse_args()


def build_cfg(
    args: argparse.Namespace,
    models: List[str],
    target_cols: List[str],
) -> Cfg:
    return Cfg(
        excel=args.excel,
        sheet=args.sheet,
        time_col=args.time_col,
        target_cols=target_cols,
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

    cfg_dict = load_config(args.config) if args.config else {}
    target_cols = get_exog_key5(cfg_dict) if args.config else get_exog_key5(None)
    best_methods = (
        get_exog_best_methods(cfg_dict) if args.config else get_exog_best_methods(None)
    )

    print("[BestSuite] 运行当前最佳外生预测方案")
    print(f"[Config] excel={args.excel} sheet={args.sheet} time_col={args.time_col}")
    print(
        f"[Config] lookback={args.lookback} H={args.H} step={args.step_minutes}min "
        f"val_days={args.val_days} test_days={args.test_days}"
    )
    print(f"[Out] {os.path.join(args.out_dir, args.tag)}")
    print(f"[BestMethods] {best_methods}")

    df = read_excel_any(args.excel, args.sheet)
    print(f"[Excel] loaded rows={len(df)} cols={len(df.columns)}")
    if args.time_col not in df.columns:
        raise ValueError(f"time_col not found: {args.time_col}")

    ensure_dir(os.path.join(args.out_dir, args.tag))

    all_rows: List[Dict] = []
    for target_col, best_method in best_methods.items():
        if target_col not in df.columns:
            print(f"[Skip] 目标列不存在，跳过: {target_col}")
            continue
        print(f"[Run] target={target_col}  best_method={best_method}")
        cfg = build_cfg(args, models=[best_method], target_cols=[target_col])
        rows = eval_one_target(cfg, df, target_col)
        all_rows.extend(rows)

    if all_rows:
        summary = pd.DataFrame(all_rows)
        summary_path = os.path.join(args.out_dir, args.tag, "best_suite_summary.csv")
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"[Saved] best suite summary: {summary_path}")
    else:
        print("[Done] 没有任何结果产生（可能是目标列缺失或样本不足）。")


if __name__ == "__main__":
    main()
