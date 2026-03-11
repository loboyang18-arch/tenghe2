#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ROADMAP 第一步：baseline + 未来 24 步 exog5 预测轨迹（双分支）。

Past branch: 历史 RT+DA+历史外生；Future branch: 未来 H 步 5 条外生预测。
推荐从仓库根目录运行: python scripts/run_rt_future_exog.py --file data/山东-全年-带时间点.xlsx
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
    select_exog_actual_cols,
)
from src.datasets import RTDualBranchDataset
from src.features_config import get_preferred_exog_order, load_config
from src.future_exog import (
    align_future_exog_to_rt_samples,
    build_future_exog_for_rt,
    fill_future_exog_nan_with_persist,
)
from src.models.patchtst_dual import RT_PatchTST_DualBranch
from src.split import SplitConfig, build_samples_seq2seq
from src.train.eval import compute_by_lead, compute_metrics, high_metrics, plot_last7d
from src.train.rt_train import (
    apply_normalizers,
    eval_model_dual,
    fit_normalizers,
    inv_y,
    predict_split_dual,
    train_one_epoch_dual,
)
from src.utils import safe_makedirs, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RT 双分支：past + future exog5 预测轨迹"
    )
    parser.add_argument("--file", type=str, default="data/山东-全年-带时间点.xlsx")
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
    parser.add_argument("--outdir", type=str, default="rt_future_exog_outputs")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = (args.device or ("cuda" if torch.cuda.is_available() else "cpu")).lower()
    if device not in ("cuda", "cpu"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    print(f"Device: {device}")

    if args.file.lower().endswith(".csv"):
        df = pd.read_csv(args.file)
    else:
        xls = pd.ExcelFile(args.file)
        sheet = args.sheet or xls.sheet_names[0]
        df = pd.read_excel(args.file, sheet_name=sheet)

    time_col = args.time_col or detect_time_col(df)
    rt_label_col = args.rt_label_col or detect_rt_label_col(df)
    da_col = args.da_col or detect_da_clearing_col(df)
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    cfg_dict = load_config(args.config) if args.config else {}
    preferred_exog = get_preferred_exog_order(cfg_dict) if args.config else None
    exog_actual_cols = select_exog_actual_cols(
        df, time_col, rt_label_col, da_col, preferred_exog_order=preferred_exog
    )

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

    print("[Future exog] Building future 24-step exog5 predictions (RT calendar split)...")
    exog_t_dec, exog_pred = build_future_exog_for_rt(
        df=df,
        time_col=time_col,
        lookback=args.L,
        H=args.H,
        step_minutes=15,
        split=split,
        ridge_alpha=10.0,
    )
    print(f"[Future exog] t_dec shape {exog_t_dec.shape}, pred shape {exog_pred.shape}")

    fut_tr = align_future_exog_to_rt_samples(Ttr, exog_t_dec, exog_pred)
    fut_va = align_future_exog_to_rt_samples(Tva, exog_t_dec, exog_pred)
    fut_te = align_future_exog_to_rt_samples(Tte, exog_t_dec, exog_pred)

    def _nan_rows(fut: np.ndarray) -> int:
        return int(np.isnan(fut.reshape(fut.shape[0], -1)).any(axis=1).sum())

    print(
        f"[Align] nan_rows train={_nan_rows(fut_tr)}/{len(fut_tr)} "
        f"val={_nan_rows(fut_va)}/{len(fut_va)} test={_nan_rows(fut_te)}/{len(fut_te)} (before persist fill)"
    )

    # 关键修复：val/test 若因外生 actual 尾部缺失导致对齐为空，用 persist 兜底补齐
    fut_tr = fill_future_exog_nan_with_persist(df, time_col, Ttr, fut_tr)
    fut_va = fill_future_exog_nan_with_persist(df, time_col, Tva, fut_va)
    fut_te = fill_future_exog_nan_with_persist(df, time_col, Tte, fut_te)

    print(
        f"[Align] nan_rows train={_nan_rows(fut_tr)}/{len(fut_tr)} "
        f"val={_nan_rows(fut_va)}/{len(fut_va)} test={_nan_rows(fut_te)}/{len(fut_te)} (after persist fill)"
    )

    def _valid_mask(fut: np.ndarray) -> np.ndarray:
        return ~np.isnan(fut.reshape(fut.shape[0], -1)).any(axis=1)

    vtr = _valid_mask(fut_tr)
    vva = _valid_mask(fut_va)
    vte = _valid_mask(fut_te)
    Xtr, Ytr, Ptr, Ttr, fut_tr = Xtr[vtr], Ytr[vtr], Ptr[vtr], Ttr[vtr], fut_tr[vtr]
    Xva, Yva, Pva, Tva, fut_va = Xva[vva], Yva[vva], Pva[vva], Tva[vva], fut_va[vva]
    Xte, Yte, Pte, Tte, fut_te = Xte[vte], Yte[vte], Pte[vte], Tte[vte], fut_te[vte]
    print(f"[Filter] train {vtr.sum()}/{len(vtr)} val {vva.sum()}/{len(vva)} test {vte.sum()}/{len(vte)} valid")

    if len(Xte) == 0:
        raise RuntimeError(
            "After aligning future exog predictions, TEST split has 0 valid samples. "
            "This usually means t_dec alignment failed (timestamps mismatch) or future exog could not be built. "
            "Check the printed [Filter] ratios; if test is 0/.., first verify time_col parsing and "
            "that exog prediction build uses the same lookback/H/step_minutes."
        )

    C = Xtr.shape[-1]
    norm = fit_normalizers(Xtr, Ytr)
    Xtr_n, Ytr_res_n, Ptr_n = apply_normalizers(Xtr, Ytr, Ptr, norm)
    Xva_n, Yva_res_n, Pva_n = apply_normalizers(Xva, Yva, Pva, norm)
    Xte_n, Yte_res_n, Pte_n = apply_normalizers(Xte, Yte, Pte, norm)

    ds_tr = RTDualBranchDataset(Xtr_n, fut_tr, Ytr_res_n, Ptr_n)
    ds_va = RTDualBranchDataset(Xva_n, fut_va, Yva_res_n, Pva_n)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = RT_PatchTST_DualBranch(
        past_dim=C,
        L=args.L,
        H=args.H,
        patch_len=args.patch_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
    ).to(device)

    print("=" * 96)
    print("RT PatchTST DualBranch (past + future exog5)")
    print("=" * 96)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_val = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch_dual(model, dl_tr, opt, device)
        va_mae = eval_model_dual(model, dl_va, device)
        print(f"Epoch {ep:03d} | train_loss={tr_loss:.4f} | val_MAE(res_norm)={va_mae:.4f}")
        if va_mae < best_val - 1e-5:
            best_val = va_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[EarlyStop] patience={args.patience}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    yhat_val = predict_split_dual(model, Xva_n, fut_va, Pva_n, norm, device)
    yhat_test = predict_split_dual(model, Xte_n, fut_te, Pte_n, norm, device)

    val_metrics = compute_metrics(Yva, yhat_val)
    test_metrics = compute_metrics(Yte, yhat_test)
    test_prior_metrics = compute_metrics(Yte, Pte)

    df_lead = compute_by_lead(Yte, yhat_test)
    ytr_flat = Ytr.reshape(-1)
    thr_p85 = float(np.quantile(ytr_flat, 0.85))
    thr_p95 = float(np.quantile(ytr_flat, 0.95))
    test_high85 = high_metrics(Yte, yhat_test, thr_p85)
    test_high95 = high_metrics(Yte, yhat_test, thr_p95)

    print("=" * 96)
    print("[VAL ]", val_metrics)
    print("[TEST] Model:", test_metrics)
    print("[TEST] Prior:", test_prior_metrics)
    print("[TEST] High>=P85:", test_high85)
    print("=" * 96)

    safe_makedirs(args.outdir)
    torch.save(model.state_dict(), os.path.join(args.outdir, "model_rt_future_exog.pt"))
    df_lead.to_csv(
        os.path.join(args.outdir, "test_by_lead_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "config": {"L": args.L, "H": args.H, "split": args.train_start},
        "metrics": {
            "val_model": val_metrics,
            "test_model": test_metrics,
            "test_prior": test_prior_metrics,
            "test_high_p85": test_high85,
            "test_high_p95": test_high95,
        },
    }
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved to {args.outdir}")


if __name__ == "__main__":
    main()
