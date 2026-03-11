#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
外生预测完整 CLI：多目标、多模型，对应原 exog_forecaster_v0_4_1_clean。
推荐从仓库根目录运行: python scripts/run_exog_full.py --excel <path> --target_cols "col1,col2" --models "seasonal_median,persist,..."
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
from src.features_config import get_exog_key5, load_config
from src.split import DEFAULT_EXOG_TEST_DAYS, DEFAULT_EXOG_VAL_DAYS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="外生预测完整回测（多目标、多模型）",
    )
    p.add_argument(
        "--excel",
        type=str,
        default="data/山东-全年-带时间点.xlsx",
        help="Excel 路径，默认 data/ 下",
    )
    p.add_argument("--sheet", type=str, default=None)
    p.add_argument("--time_col", type=str, default="datetime")
    p.add_argument(
        "--target_cols",
        type=str,
        default=None,
        help="Comma-separated target columns (actual)；也可用 --config 从 YAML 读",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML/JSON 配置路径，可选；含 target_cols 时覆盖 --target_cols",
    )
    p.add_argument("--step_minutes", type=int, default=15)
    p.add_argument("--lookback", type=int, default=192)
    p.add_argument("--H", type=int, default=24)
    p.add_argument("--val_days", type=int, default=DEFAULT_EXOG_VAL_DAYS)
    p.add_argument("--test_days", type=int, default=DEFAULT_EXOG_TEST_DAYS)
    p.add_argument(
        "--feat_nan_strategy",
        type=str,
        default="ffill_bfill",
        choices=["none", "ffill", "ffill_bfill"],
    )
    p.add_argument("--ridge_alpha", type=float, default=10.0)
    p.add_argument(
        "--models",
        type=str,
        default="seasonal_median,persist,use_pred,ridge_pure,ridge_residual",
        help="Comma-separated: seasonal_median,persist,use_pred,ridge_pure,ridge_residual,hgbt_pure,hgbt_residual",
    )
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--tag", type=str, default="exog_v0_4_1_clean")
    return p.parse_args()


def main() -> None:
    setup_matplotlib_fonts()
    args = parse_args()

    cfg_dict = load_config(args.config) if args.config else {}
    if args.config and cfg_dict and get_exog_key5(cfg_dict):
        target_cols = get_exog_key5(cfg_dict)
    elif args.target_cols:
        target_cols = [c.strip() for c in args.target_cols.split(",") if c.strip()]
    else:
        target_cols = get_exog_key5(None)

    cfg = Cfg(
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
        models=[m.strip() for m in args.models.split(",") if m.strip()],
        out_dir=str(args.out_dir),
        tag=str(args.tag),
    )

    print("[Start] exog full suite")
    print(f"[Config] excel={cfg.excel} time_col={cfg.time_col}")
    print(f"[Config] lookback={cfg.lookback} H={cfg.H} val_days={cfg.val_days} test_days={cfg.test_days}")
    print(f"[Config] models={cfg.models}")
    print(f"[Out] {os.path.join(cfg.out_dir, cfg.tag)}")

    df = read_excel_any(cfg.excel, cfg.sheet)
    print(f"[Excel] loaded rows={len(df)} cols={len(df.columns)}")
    if cfg.time_col not in df.columns:
        raise ValueError(f"time_col not found: {cfg.time_col}")

    ensure_dir(os.path.join(cfg.out_dir, cfg.tag))

    all_rows = []
    for tc in cfg.target_cols:
        all_rows.extend(eval_one_target(cfg, df, tc))

    if all_rows:
        summary = pd.DataFrame(all_rows)
        summary_path = os.path.join(cfg.out_dir, cfg.tag, "suite_summary_clean.csv")
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"[Saved] suite summary: {summary_path}")
    else:
        print("[Done] No results produced.")


if __name__ == "__main__":
    main()
