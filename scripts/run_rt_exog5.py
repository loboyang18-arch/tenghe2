#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RT PatchTST exog5 薄入口：仅用五大外生，其余同 baseline。
推荐从仓库根目录运行: python scripts/run_rt_exog5.py --file <excel>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataio import (
    build_feature_matrix,
    detect_da_clearing_col,
    detect_rt_label_col,
    detect_time_col,
    select_exog_actual_cols_exog5,
)
from src.features_config import get_exog_key5, load_config
from src.datasets import RTSeq2SeqDataset
from src.models.patchtst import RT_PatchTST
from src.split import SplitConfig, build_samples_seq2seq
from src.train.eval import (
    compute_by_lead,
    compute_metrics,
    high_metrics,
    plot_last7d,
)
from src.train.rt_train import (
    apply_normalizers,
    eval_model,
    fit_normalizers,
    predict_split,
    train_one_epoch,
)
from src.utils import safe_makedirs, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RT PatchTST exog5 (5 exog cols only, residual learning)"
    )
    parser.add_argument(
        "--file",
        type=str,
        default="data/山东-全年-带时间点.xlsx",
        help="Excel/CSV 路径，默认 data/ 下",
    )
    parser.add_argument("--sheet", type=str, default=None)
    parser.add_argument("--time_col", type=str, default=None)
    parser.add_argument("--rt_label_col", type=str, default=None)
    parser.add_argument("--da_col", type=str, default=None)
    parser.add_argument("--train_start", type=str, default="2025-01-01")
    parser.add_argument("--val_start", type=str, default="2025-11-01")
    parser.add_argument("--test_start", type=str, default="2025-12-01")
    parser.add_argument("--end", type=str, default="2025-12-30")
    parser.add_argument("--L", type=int, default=672)
    parser.add_argument("--H", type=int, default=24)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dim_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="rt_patchtst_exog5_v1_outputs")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML/JSON 配置路径，可选；用于 exog_columns",
    )
    args = parser.parse_args()

    cfg_dict = load_config(args.config) if args.config else {}
    exog_key5_list = get_exog_key5(cfg_dict) if args.config else None

    set_seed(args.seed)

    device = (args.device or ("cuda" if torch.cuda.is_available() else "cpu")).lower()
    if device not in ("cuda", "cpu"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA not available. Use --device cpu")
    print(f"Device: {device}")

    if args.file.lower().endswith(".csv"):
        df = pd.read_csv(args.file)
    else:
        xls = pd.ExcelFile(args.file)
        sheet = args.sheet if args.sheet is not None else xls.sheet_names[0]
        df = pd.read_excel(args.file, sheet_name=sheet)

    time_col = args.time_col or detect_time_col(df)
    rt_label_col = args.rt_label_col or detect_rt_label_col(df)
    da_col = args.da_col or detect_da_clearing_col(df)
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    exog_actual_cols = select_exog_actual_cols_exog5(
        df, time_col, rt_label_col, da_col, candidate_exog=exog_key5_list
    )

    print("=" * 96)
    print("[Exog5] Only: 系统负荷实际值, 光伏实际值, 联络线实际值, 上旋备用实际值, 下旋备用实际值")
    print(f"Selected exog({len(exog_actual_cols)}): {exog_actual_cols}")
    print("=" * 96)

    split = SplitConfig(
        args.train_start, args.val_start, args.test_start, args.end
    )

    ts, feat_mat, rt_y, feat_names = build_feature_matrix(
        df=df,
        time_col=time_col,
        rt_label_col=rt_label_col,
        da_col=da_col,
        exog_actual_cols=exog_actual_cols,
    )

    samples = build_samples_seq2seq(
        ts=ts, feat_mat=feat_mat, rt_y=rt_y, split=split, L=args.L, H=args.H
    )

    Xtr = samples["train"]["X"]
    Ytr = samples["train"]["Y"]
    Ptr = samples["train"]["prior"]
    Ttr = samples["train"]["t_dec"]
    Xva = samples["val"]["X"]
    Yva = samples["val"]["Y"]
    Pva = samples["val"]["prior"]
    Tva = samples["val"]["t_dec"]
    Xte = samples["test"]["X"]
    Yte = samples["test"]["Y"]
    Pte = samples["test"]["prior"]
    Tte = samples["test"]["t_dec"]

    C = Xtr.shape[-1]
    print(f"Features C={C} | L={args.L} | H={args.H}")

    norm = fit_normalizers(Xtr, Ytr)
    Xtr_n, Ytr_res_n, Ptr_n = apply_normalizers(Xtr, Ytr, Ptr, norm)
    Xva_n, Yva_res_n, Pva_n = apply_normalizers(Xva, Yva, Pva, norm)
    Xte_n, Yte_res_n, Pte_n = apply_normalizers(Xte, Yte, Pte, norm)

    ds_tr = RTSeq2SeqDataset(Xtr_n, Ytr_res_n, Ptr_n, Ttr)
    ds_va = RTSeq2SeqDataset(Xva_n, Yva_res_n, Pva_n, Tva)
    dl_tr = DataLoader(
        ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    dl_va = DataLoader(
        ds_va, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    model = RT_PatchTST(
        in_dim=C,
        L=args.L,
        H=args.H,
        patch_len=args.patch_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_val = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, dl_tr, opt, device)
        va_mae_res = eval_model(model, dl_va, device)
        print(
            f"Epoch {ep:03d} | train_loss={tr_loss:.4f} | val_MAE(res_norm)={va_mae_res:.4f}"
        )
        if va_mae_res < best_val - 1e-5:
            best_val = va_mae_res
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[EarlyStop] patience={args.patience} reached.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    yhat_val = predict_split(model, Xva_n, Pva_n, norm, device)
    yhat_test = predict_split(model, Xte_n, Pte_n, norm, device)

    val_metrics = compute_metrics(Yva, yhat_val)
    test_metrics = compute_metrics(Yte, yhat_test)
    val_prior_metrics = compute_metrics(Yva, Pva)
    test_prior_metrics = compute_metrics(Yte, Pte)

    df_by_lead_test = compute_by_lead(Yte, yhat_test)
    df_by_lead_test_prior = compute_by_lead(Yte, Pte)

    ytr_flat = Ytr.reshape(-1)
    thr_p85 = float(np.quantile(ytr_flat, 0.85))
    thr_p95 = float(np.quantile(ytr_flat, 0.95))
    test_high85 = high_metrics(Yte, yhat_test, thr_p85)
    test_high95 = high_metrics(Yte, yhat_test, thr_p95)

    lead1 = df_by_lead_test[df_by_lead_test["lead_idx"] == 1].iloc[0].to_dict()
    lead24 = df_by_lead_test[df_by_lead_test["lead_idx"] == 24].iloc[0].to_dict()
    lead1_p = df_by_lead_test_prior[
        df_by_lead_test_prior["lead_idx"] == 1
    ].iloc[0].to_dict()
    lead24_p = df_by_lead_test_prior[
        df_by_lead_test_prior["lead_idx"] == 24
    ].iloc[0].to_dict()

    print("=" * 96)
    print("[VAL ] Overall:", val_metrics)
    print("[TEST] Overall:", test_metrics)
    print("[TEST] Prior  :", test_prior_metrics)
    print("[TEST] Lead1 (15m)  Model:", lead1)
    print("[TEST] Lead24(6h)   Model:", lead24)
    print("[TEST] High>=P85:", test_high85)
    print("=" * 96)

    safe_makedirs(args.outdir)
    df_by_lead_test.to_csv(
        os.path.join(args.outdir, "test_by_lead_metrics_model.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    df_by_lead_test_prior.to_csv(
        os.path.join(args.outdir, "test_by_lead_metrics_prior.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    rows = []
    for i, tdec in enumerate(pd.to_datetime(Tte)):
        rows.append(
            {
                "t_dec": tdec,
                "t_target_15m": tdec + pd.Timedelta(minutes=15),
                "y_true_15m": float(Yte[i, 0]),
                "y_pred_15m": float(yhat_test[i, 0]),
                "prior_15m": float(Pte[i, 0]),
                "t_target_6h": tdec + pd.Timedelta(hours=6),
                "y_true_6h": float(Yte[i, 23]),
                "y_pred_6h": float(yhat_test[i, 23]),
                "prior_6h": float(Pte[i, 23]),
            }
        )
    pd.DataFrame(rows).to_csv(
        os.path.join(args.outdir, "test_point_predictions_lead1_lead24.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    rt_series = pd.Series(rt_y, index=ts, name="rt")
    plot_path = os.path.join(args.outdir, "RT_patchtst_exog5_test_last7d.png")
    plot_last7d(
        ts_all=ts,
        rt_series=rt_series,
        t_dec_test=Tte,
        y_pred_test=yhat_test,
        out_path=plot_path,
        title=f"RT PatchTST exog5 (Test Last 7d)",
    )

    torch.save(
        model.state_dict(),
        os.path.join(args.outdir, "model_rt_patchtst_exog5_v1.pt"),
    )

    summary = {
        "config": {
            "file": args.file,
            "split": {
                "train_start": split.train_start,
                "val_start": split.val_start,
                "test_start": split.test_start,
                "end": split.end,
            },
            "L": args.L,
            "H": args.H,
        },
        "features": {"C": int(C), "exog_actual_cols": exog_actual_cols},
        "metrics": {
            "val_model": val_metrics,
            "test_model": test_metrics,
            "test_prior": test_prior_metrics,
        },
    }
    with open(
        os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved outputs to: {args.outdir}")


if __name__ == "__main__":
    main()
