# -*- coding: utf-8 -*-
"""
RT PatchTST baseline v0.1 (seq2seq)
- STRICT no-leakage imputation: ffill only, no bfill, no bidirectional interpolate.
- Persistence prior + residual learning:
    prior(h) = rt(t_dec) for all h in [1..H]
    model learns residual: y - prior
    final pred = prior + residual_pred
- Plot last 7 days: Actual vs Pred(15m ahead) vs Pred(6h ahead)
- Evaluate overall + by lead (lead1 & lead24) + persistence baseline.
"""

import os
import json
import math
import argparse
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def bias(a: np.ndarray, b: np.ndarray) -> float:
    # pred - true bias
    return float(np.mean(a - b))


def safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


def detect_time_col(df: pd.DataFrame) -> str:
    candidates = ["datetime", "时间", "日期时间", "时间点", "timestamp", "Timestamp", "DATETIME"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: first datetime-like
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            return c
    raise ValueError("Cannot detect time column. Please specify via --time_col")


def detect_rt_label_col(df: pd.DataFrame) -> str:
    candidates = [
        "实时出清电价", "实时出清价格", "实时电价", "实时价格", "RT出清电价", "RT出清价格"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: contains 关键字
    for c in df.columns:
        if ("实时" in c) and ("出清" in c) and ("价" in c):
            return c
    raise ValueError("Cannot detect RT label column (实时出清电价). Please specify via --rt_label_col")


def detect_da_clearing_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "日前出清电价", "日前出清价格", "日前电价", "日前价格", "DA出清电价", "DA出清价格"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if ("日前" in c) and ("出清" in c) and ("价" in c):
            return c
    # not mandatory
    return None


def is_numeric_series(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        return True
    # attempt coercion
    try:
        pd.to_numeric(s.dropna().iloc[:50], errors="raise")
        return True
    except Exception:
        return False


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def build_time_features(ts: pd.DatetimeIndex) -> np.ndarray:
    # 11 dims,保持你之前 v0 的风格（可解释 + 轻量）
    hour = ts.hour.values.astype(np.float32)
    minute = ts.minute.values.astype(np.float32)
    quarter = (minute // 15).astype(np.float32)  # 0,1,2,3
    weekday = ts.weekday.values.astype(np.float32)  # 0-6
    month = ts.month.values.astype(np.float32)      # 1-12
    is_weekend = (weekday >= 5).astype(np.float32)

    # cyclic encodings
    hour_sin = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)
    w_sin = np.sin(2 * np.pi * weekday / 7.0).astype(np.float32)
    w_cos = np.cos(2 * np.pi * weekday / 7.0).astype(np.float32)
    m_sin = np.sin(2 * np.pi * (month - 1) / 12.0).astype(np.float32)
    m_cos = np.cos(2 * np.pi * (month - 1) / 12.0).astype(np.float32)

    feats = np.stack([
        hour, quarter, weekday, month, is_weekend,
        hour_sin, hour_cos, w_sin, w_cos, m_sin, m_cos
    ], axis=1)
    return feats


@dataclass
class SplitConfig:
    train_start: str = "2025-01-01"
    val_start: str = "2025-11-01"
    test_start: str = "2025-12-01"
    end: str = "2025-12-30"


# -----------------------------
# Dataset
# -----------------------------
class RTSeq2SeqDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,            # [N, L, C]
        Y_res_norm: np.ndarray,   # [N, H] residual in normalized space
        prior_norm: np.ndarray,   # [N, H] prior in normalized space (for reconstruct if needed)
        t_dec: np.ndarray,        # [N] datetime64
    ):
        self.X = X.astype(np.float32)
        self.Y = Y_res_norm.astype(np.float32)
        self.prior = prior_norm.astype(np.float32)
        self.t_dec = t_dec

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.from_numpy(self.Y[idx]),
            torch.from_numpy(self.prior[idx]),
        )


# -----------------------------
# PatchTST (minimal)
# -----------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, in_dim: int, patch_len: int, d_model: int):
        super().__init__()
        self.in_dim = in_dim
        self.patch_len = patch_len
        self.proj = nn.Linear(in_dim * patch_len, d_model)

    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.shape
        P = self.patch_len
        assert L % P == 0, f"L({L}) must be divisible by patch_len({P})"
        n_patches = L // P
        x = x.reshape(B, n_patches, P * C)  # [B, n_patches, P*C]
        x = self.proj(x)                    # [B, n_patches, d_model]
        return x


class RT_PatchTST(nn.Module):
    def __init__(
        self,
        in_dim: int,
        L: int,
        H: int,
        patch_len: int = 16,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.L = L
        self.H = H
        self.patch_len = patch_len
        assert L % patch_len == 0, "For simplicity, require L % patch_len == 0"
        self.n_patches = L // patch_len

        self.patch_embed = PatchEmbedding(in_dim, patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, H),
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: [B, L, C]
        z = self.patch_embed(x) + self.pos_embed
        z = self.encoder(z)
        z = self.norm(z)
        # global average pooling over patches
        pooled = z.mean(dim=1)  # [B, d_model]
        out = self.head(pooled) # [B, H] residual prediction (normalized space)
        return out


# -----------------------------
# Data preparation
# -----------------------------
def build_feature_matrix(
    df: pd.DataFrame,
    time_col: str,
    rt_label_col: str,
    da_col: Optional[str],
    exog_actual_cols: List[str],
) -> Tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
      ts: DatetimeIndex
      feat_mat: [T, C]   (rt, da(optional), exog..., time_feats...)
      rt_y: [T]          (rt label)
      feat_names: list of length C
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    ts = pd.to_datetime(df[time_col])
    df = df.set_index(ts)

    # Coerce numeric
    numeric_cols = [rt_label_col] + ([da_col] if da_col else []) + exog_actual_cols
    df = coerce_numeric(df, [c for c in numeric_cols if c is not None])

    # Strict no-leakage filling: ONLY forward-fill
    # (Do NOT bfill, do NOT bidirectional interpolate)
    df[numeric_cols] = df[numeric_cols].ffill()

    # Build time features
    tfeats = build_time_features(df.index)  # [T, 11]

    # Assemble
    feat_list = []
    feat_names = []

    # rt current
    feat_list.append(df[rt_label_col].values.astype(np.float32).reshape(-1, 1))
    feat_names.append("rt_current")

    # da clearing (optional)
    if da_col:
        feat_list.append(df[da_col].values.astype(np.float32).reshape(-1, 1))
        feat_names.append("da_clearing")

    # exog
    for c in exog_actual_cols:
        feat_list.append(df[c].values.astype(np.float32).reshape(-1, 1))
        feat_names.append(c)

    # time
    feat_list.append(tfeats.astype(np.float32))
    feat_names.extend([
        "hour", "quarter", "weekday", "month", "is_weekend",
        "hour_sin", "hour_cos", "w_sin", "w_cos", "m_sin", "m_cos"
    ])

    feat_mat = np.concatenate(feat_list, axis=1)  # [T, C]
    rt_y = df[rt_label_col].values.astype(np.float32)

    return df.index, feat_mat, rt_y, feat_names


def select_exog_actual_cols(df: pd.DataFrame, time_col: str, rt_label_col: str, da_col: Optional[str]) -> List[str]:
    """
    Policy: keep non-empty numeric columns that are NOT prices and do NOT contain '预测' (strict).
    """
    banned_substr = ["预测"]  # strict for RT v0.1
    banned_exact = set([time_col, rt_label_col] + ([da_col] if da_col else []))

    cols = []
    for c in df.columns:
        if c in banned_exact:
            continue
        if any(b in str(c) for b in banned_substr):
            continue
        s = df[c]
        if s.isna().all():
            continue
        if not is_numeric_series(s):
            continue
        cols.append(c)

    # Optional: prioritize known exog names first (stable ordering)
    preferred = [
        "系统负荷实际值", "风光总加实际值", "竞价空间实际值", "联络线实际值", "地方电厂发电实际值",
        "风电实际值", "光伏实际值", "自备机组实际值", "试验机组实际值", "非市场化核电实际值",
        "上旋备用实际值", "下旋备用实际值"
    ]
    cols_sorted = []
    for p in preferred:
        if p in cols:
            cols_sorted.append(p)
    for c in cols:
        if c not in cols_sorted:
            cols_sorted.append(c)
    return cols_sorted


def build_samples_seq2seq(
    ts: pd.DatetimeIndex,
    feat_mat: np.ndarray,     # [T, C]
    rt_y: np.ndarray,         # [T]
    split: SplitConfig,
    L: int,
    H: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Build samples for train/val/test:
      X: [N, L, C]
      Y: [N, H]  true y (original scale)
      prior: [N, H] persistence prior (original scale)
      t_dec: [N] decision timestamps (datetime64)
    Strict NaN rule: if any NaN in X window or Y horizon => drop sample.
    """
    assert feat_mat.shape[0] == len(ts) == len(rt_y)
    T, C = feat_mat.shape

    # indices by split
    train_start = pd.Timestamp(split.train_start)
    val_start = pd.Timestamp(split.val_start)
    test_start = pd.Timestamp(split.test_start)
    end = pd.Timestamp(split.end)

    def split_name(t: pd.Timestamp) -> Optional[str]:
        if t < train_start or t > end:
            return None
        if t < val_start:
            return "train"
        if t < test_start:
            return "val"
        return "test"

    buckets = {k: {"X": [], "Y": [], "prior": [], "t_dec": []} for k in ["train", "val", "test"]}

    # decision index i corresponds to t_dec = ts[i]
    # X uses [i-L+1 .. i] inclusive => length L
    # Y uses [i+1 .. i+H] => length H
    for i in range(L - 1, T - H):
        t_dec = ts[i]
        name = split_name(pd.Timestamp(t_dec))
        if name is None:
            continue

        Xwin = feat_mat[i - L + 1: i + 1, :]        # [L, C]
        Yhor = rt_y[i + 1: i + 1 + H].reshape(-1)   # [H]
        prior = np.full((H,), rt_y[i], dtype=np.float32)

        # strict NaN check (no leakage)
        if np.isnan(Xwin).any() or np.isnan(Yhor).any() or np.isnan(prior).any():
            continue

        buckets[name]["X"].append(Xwin)
        buckets[name]["Y"].append(Yhor)
        buckets[name]["prior"].append(prior)
        buckets[name]["t_dec"].append(np.datetime64(t_dec))

    # stack
    out = {}
    for k in ["train", "val", "test"]:
        if len(buckets[k]["X"]) == 0:
            raise RuntimeError(f"No samples built for split={k}. Check split ranges and missing data.")
        out[k] = {
            "X": np.stack(buckets[k]["X"], axis=0),
            "Y": np.stack(buckets[k]["Y"], axis=0),
            "prior": np.stack(buckets[k]["prior"], axis=0),
            "t_dec": np.array(buckets[k]["t_dec"]),
        }
    return out


def fit_normalizers(train_X: np.ndarray, train_Y: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Normalize features per-channel, and normalize y globally.
    """
    # X: [N, L, C]
    X_flat = train_X.reshape(-1, train_X.shape[-1])
    x_mean = X_flat.mean(axis=0)
    x_std = X_flat.std(axis=0) + 1e-6

    # y: [N, H]
    y_mean = train_Y.mean()
    y_std = train_Y.std() + 1e-6

    return {"x_mean": x_mean, "x_std": x_std, "y_mean": np.array([y_mean]), "y_std": np.array([y_std])}


def apply_normalizers(X: np.ndarray, y: np.ndarray, prior: np.ndarray, norm: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
      X_norm: [N, L, C]
      y_res_norm: [N, H] residual in normalized space
      prior_norm: [N, H] prior in normalized space
    """
    x_mean, x_std = norm["x_mean"], norm["x_std"]
    y_mean, y_std = float(norm["y_mean"][0]), float(norm["y_std"][0])

    Xn = (X - x_mean.reshape(1, 1, -1)) / x_std.reshape(1, 1, -1)
    yn = (y - y_mean) / y_std
    pn = (prior - y_mean) / y_std

    # residual in normalized space: (y - prior) / y_std  == yn - pn
    resn = yn - pn
    return Xn.astype(np.float32), resn.astype(np.float32), pn.astype(np.float32)


def inv_y(y_norm: np.ndarray, norm: Dict[str, np.ndarray]) -> np.ndarray:
    y_mean, y_std = float(norm["y_mean"][0]), float(norm["y_std"][0])
    return y_norm * y_std + y_mean


# -----------------------------
# Training / Evaluation
# -----------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    n = 0
    loss_fn = nn.MSELoss()
    for X, y_res, _prior in loader:
        X = X.to(device)
        y_res = y_res.to(device)

        optimizer.zero_grad()
        pred_res = model(X)
        loss = loss_fn(pred_res, y_res)
        loss.backward()
        optimizer.step()

        total += float(loss.item()) * X.size(0)
        n += X.size(0)
    return total / max(n, 1)


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    # return MAE in normalized residual space (for early stop reference)
    maes = []
    for X, y_res, _prior in loader:
        X = X.to(device)
        y_res = y_res.to(device)
        pred_res = model(X)
        maes.append(torch.mean(torch.abs(pred_res - y_res)).item())
    return float(np.mean(maes))


@torch.no_grad()
def predict_split(model, Xn: np.ndarray, prior_norm: np.ndarray, norm: Dict[str, np.ndarray], device) -> np.ndarray:
    """
    Return predictions in original scale: [N, H]
    """
    model.eval()
    bs = 256
    preds = []
    for i in range(0, Xn.shape[0], bs):
        xb = torch.from_numpy(Xn[i:i+bs]).to(device)
        pr = model(xb).cpu().numpy()  # residual norm
        pn = prior_norm[i:i+bs]       # prior norm
        yhat_norm = pn + pr
        preds.append(inv_y(yhat_norm, norm))
    return np.concatenate(preds, axis=0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    return {
        "MAE": mae(yp, yt),
        "RMSE": rmse(yp, yt),
        "Bias": bias(yp, yt),
        "n": int(len(yt)),
    }


def compute_by_lead(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    H = y_true.shape[1]
    rows = []
    for h in range(H):
        yt = y_true[:, h]
        yp = y_pred[:, h]
        rows.append({
            "lead_idx": h + 1,
            "lead_minutes": 15 * (h + 1),
            "MAE": mae(yp, yt),
            "RMSE": rmse(yp, yt),
            "Bias": bias(yp, yt),
            "n": int(len(yt)),
        })
    return pd.DataFrame(rows)


def plot_last7d(
    ts_all: pd.DatetimeIndex,
    rt_series: pd.Series,
    t_dec_test: np.ndarray,
    y_pred_test: np.ndarray,   # [N, H] original scale
    out_path: str,
    title: str,
):
    """
    Build aligned series:
      lead1 predictions align at t_dec+1 step timestamp
      lead24 predictions align at t_dec+24 step timestamp
    Plot last 7 days based on actual series timestamps.
    """
    # map timestamp -> pred value (unique per timestamp in this construction)
    pred1 = {}
    pred24 = {}
    for i, tdec in enumerate(pd.to_datetime(t_dec_test)):
        t1 = tdec + pd.Timedelta(minutes=15)
        t24 = tdec + pd.Timedelta(hours=6)
        pred1[t1] = float(y_pred_test[i, 0])
        pred24[t24] = float(y_pred_test[i, 23])

    s_pred1 = pd.Series(pred1)
    s_pred24 = pd.Series(pred24)

    # last 7 days window by actual timestamps
    end = rt_series.index.max()
    start = end - pd.Timedelta(days=7)
    rt_last = rt_series.loc[(rt_series.index >= start) & (rt_series.index <= end)]
    p1_last = s_pred1.loc[(s_pred1.index >= start) & (s_pred1.index <= end)]
    p24_last = s_pred24.loc[(s_pred24.index >= start) & (s_pred24.index <= end)]

    plt.figure(figsize=(16, 5))
    plt.plot(rt_last.index, rt_last.values, label="RT Clearing (Actual)")
    if len(p1_last) > 0:
        plt.plot(p1_last.index, p1_last.values, label="Pred (15min ahead)")
    if len(p24_last) > 0:
        plt.plot(p24_last.index, p24_last.values, label="Pred (6h ahead)")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("RT Price")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="山东-全年-带时间点.xlsx", help="Excel file path")
    parser.add_argument("--sheet", type=str, default=None, help="Sheet name (default: first sheet)")
    parser.add_argument("--time_col", type=str, default=None, help="Time column name (optional)")
    parser.add_argument("--rt_label_col", type=str, default=None, help="RT label column name (optional)")
    parser.add_argument("--da_col", type=str, default=None, help="DA clearing column name (optional)")
    parser.add_argument("--train_start", type=str, default="2025-01-01")
    parser.add_argument("--val_start", type=str, default="2025-11-01")
    parser.add_argument("--test_start", type=str, default="2025-12-01")
    parser.add_argument("--end", type=str, default="2025-12-30")

    parser.add_argument("--L", type=int, default=672, help="lookback length (steps)")
    parser.add_argument("--H", type=int, default=24, help="horizon length (steps)")
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=16, help="kept for log compatibility (not used in this minimal impl)")
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

    parser.add_argument("--outdir", type=str, default="rt_patchtst_v0_1_outputs")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda, cpu, or omit for auto (cuda if available else cpu). Use cuda on cloud with GPU.")
    args = parser.parse_args()

    set_seed(args.seed)

    device = (args.device or ("cuda" if torch.cuda.is_available() else "cpu")).lower()
    if device not in ("cuda", "cpu"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False. Install CUDA-enabled PyTorch or use --device cpu")
    print(f"Device: {device}")

    # Read
    if args.file.lower().endswith(".csv"):
        df = pd.read_csv(args.file)
    else:
        xls = pd.ExcelFile(args.file)
        sheet = args.sheet if args.sheet is not None else xls.sheet_names[0]
        df = pd.read_excel(args.file, sheet_name=sheet)

    # Detect columns
    time_col = args.time_col or detect_time_col(df)
    rt_label_col = args.rt_label_col or detect_rt_label_col(df)
    da_col = args.da_col or detect_da_clearing_col(df)

    # Parse time
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    exog_actual_cols = select_exog_actual_cols(df, time_col, rt_label_col, da_col)

    print("=" * 96)
    print("[Detected Columns]")
    print(f"time_col: {time_col}")
    print(f"rt_label_col: {rt_label_col}")
    print(f"da_clearing_col: {da_col}")
    print(f"exog_actual_cols({len(exog_actual_cols)}):")
    for i, c in enumerate(exog_actual_cols, 1):
        print(f"  {i:02d}. {c}")
    print("=" * 96)
    print("[Split]")
    print(f"train_start: {args.train_start} val_start: {args.val_start} test_start: {args.test_start} end: {args.end}")

    split = SplitConfig(args.train_start, args.val_start, args.test_start, args.end)

    # Build feature matrix
    ts, feat_mat, rt_y, feat_names = build_feature_matrix(
        df=df,
        time_col=time_col,
        rt_label_col=rt_label_col,
        da_col=da_col,
        exog_actual_cols=exog_actual_cols,
    )

    # Build samples (strict NaN dropping)
    samples = build_samples_seq2seq(
        ts=ts,
        feat_mat=feat_mat,
        rt_y=rt_y,
        split=split,
        L=args.L,
        H=args.H,
    )

    Xtr, Ytr, Ptr, Ttr = samples["train"]["X"], samples["train"]["Y"], samples["train"]["prior"], samples["train"]["t_dec"]
    Xva, Yva, Pva, Tva = samples["val"]["X"], samples["val"]["Y"], samples["val"]["prior"], samples["val"]["t_dec"]
    Xte, Yte, Pte, Tte = samples["test"]["X"], samples["test"]["Y"], samples["test"]["prior"], samples["test"]["t_dec"]

    print(f"Train: X={Xtr.shape} Y={Ytr.shape}  (t_dec from {pd.to_datetime(Ttr[0])} to {pd.to_datetime(Ttr[-1])})")
    print(f"Val  : X={Xva.shape} Y={Yva.shape}  (t_dec from {pd.to_datetime(Tva[0])} to {pd.to_datetime(Tva[-1])})")
    print(f"Test : X={Xte.shape} Y={Yte.shape}  (t_dec from {pd.to_datetime(Tte[0])} to {pd.to_datetime(Tte[-1])})")

    C = Xtr.shape[-1]
    print(f"Features C = {C} | Lookback L = {args.L} | Horizon H = {args.H}")

    # Fit normalizers on train
    norm = fit_normalizers(Xtr, Ytr)

    # Apply normalizers + residual labels
    Xtr_n, Ytr_res_n, Ptr_n = apply_normalizers(Xtr, Ytr, Ptr, norm)
    Xva_n, Yva_res_n, Pva_n = apply_normalizers(Xva, Yva, Pva, norm)
    Xte_n, Yte_res_n, Pte_n = apply_normalizers(Xte, Yte, Pte, norm)

    # Datasets / Loaders
    ds_tr = RTSeq2SeqDataset(Xtr_n, Ytr_res_n, Ptr_n, Ttr)
    ds_va = RTSeq2SeqDataset(Xva_n, Yva_res_n, Pva_n, Tva)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Model
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

    print("=" * 96)
    print("RT PatchTST baseline v0.1 (seq2seq, residual learning)")
    print(f"Input X: [B, L, C] = [B, {args.L}, {C}] | Output Y: [B, H] = [B, {args.H}]")
    print(f"patch_len={args.patch_len} => n_patches={args.L // args.patch_len}")
    print("STRICT: no columns with '预测' used. Missing: ffill only; samples with NaN are dropped.")
    print("Head: predict residual; final_pred = prior + residual_pred")
    print("=" * 96)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, dl_tr, opt, device)
        va_mae_res = eval_model(model, dl_va, device)  # MAE in normalized residual space
        print(f"Epoch {ep:03d} | train_loss={tr_loss:.4f} | val_MAE(res_norm)={va_mae_res:.4f}")

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

    # Predictions (original scale)
    yhat_val = predict_split(model, Xva_n, Pva_n, norm, device)
    yhat_test = predict_split(model, Xte_n, Pte_n, norm, device)

    # Persistence baseline (original scale): y(t_dec) for all horizons
    prior_val = Pva
    prior_test = Pte

    # Metrics: overall
    val_metrics = compute_metrics(Yva, yhat_val)
    test_metrics = compute_metrics(Yte, yhat_test)

    # Baseline metrics (prior only)
    val_prior_metrics = compute_metrics(Yva, prior_val)
    test_prior_metrics = compute_metrics(Yte, prior_test)

    # By lead
    df_by_lead_test = compute_by_lead(Yte, yhat_test)
    df_by_lead_test_prior = compute_by_lead(Yte, prior_test)

    # High-price thresholds from train labels (original scale) over ALL horizons
    ytr_flat = Ytr.reshape(-1)
    thr_p85 = float(np.quantile(ytr_flat, 0.85))
    thr_p95 = float(np.quantile(ytr_flat, 0.95))

    def high_metrics(y_true, y_pred, thr: float) -> Dict[str, float]:
        mask = (y_true.reshape(-1) >= thr)
        if mask.sum() == 0:
            return {"thr": thr, "n": 0, "MAE": float("nan"), "RMSE": float("nan"), "Bias": float("nan")}
        yt = y_true.reshape(-1)[mask]
        yp = y_pred.reshape(-1)[mask]
        return {"thr": float(thr), "n": int(mask.sum()), "MAE": mae(yp, yt), "RMSE": rmse(yp, yt), "Bias": bias(yp, yt)}

    test_high85 = high_metrics(Yte, yhat_test, thr_p85)
    test_high95 = high_metrics(Yte, yhat_test, thr_p95)

    # Print summary
    print("=" * 96)
    print("[VAL ] Overall:", val_metrics)
    print("[VAL ] Prior  :", val_prior_metrics)
    print("[TEST] Overall:", test_metrics)
    print("[TEST] Prior  :", test_prior_metrics)

    # Lead 1 & Lead 24 quick view
    lead1 = df_by_lead_test[df_by_lead_test["lead_idx"] == 1].iloc[0].to_dict()
    lead24 = df_by_lead_test[df_by_lead_test["lead_idx"] == 24].iloc[0].to_dict()
    lead1_p = df_by_lead_test_prior[df_by_lead_test_prior["lead_idx"] == 1].iloc[0].to_dict()
    lead24_p = df_by_lead_test_prior[df_by_lead_test_prior["lead_idx"] == 24].iloc[0].to_dict()

    print("[TEST] Lead1 (15m)  Model:", lead1)
    print("[TEST] Lead1 (15m)  Prior:", lead1_p)
    print("[TEST] Lead24(6h)   Model:", lead24)
    print("[TEST] Lead24(6h)   Prior:", lead24_p)

    print(f"[Train RT thr] P85={thr_p85:.2f}  P95={thr_p95:.2f}")
    print("[TEST] High>=P85:", test_high85)
    print("[TEST] High>=P95:", test_high95)
    print("=" * 96)

    # Outputs
    safe_makedirs(args.outdir)

    # Save by-lead metrics
    df_by_lead_test.to_csv(os.path.join(args.outdir, "test_by_lead_metrics_model.csv"), index=False, encoding="utf-8-sig")
    df_by_lead_test_prior.to_csv(os.path.join(args.outdir, "test_by_lead_metrics_prior.csv"), index=False, encoding="utf-8-sig")

    # Save point predictions for test (lead1 & lead24 aligned to target timestamps)
    rows = []
    for i, tdec in enumerate(pd.to_datetime(Tte)):
        rows.append({
            "t_dec": tdec,
            "t_target_15m": tdec + pd.Timedelta(minutes=15),
            "y_true_15m": float(Yte[i, 0]),
            "y_pred_15m": float(yhat_test[i, 0]),
            "prior_15m": float(prior_test[i, 0]),
            "t_target_6h": tdec + pd.Timedelta(hours=6),
            "y_true_6h": float(Yte[i, 23]),
            "y_pred_6h": float(yhat_test[i, 23]),
            "prior_6h": float(prior_test[i, 23]),
        })
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "test_point_predictions_lead1_lead24.csv"),
                              index=False, encoding="utf-8-sig")

    # Plot last 7 days (aligned)
    rt_series = pd.Series(rt_y, index=ts, name="rt")
    plot_path = os.path.join(args.outdir, "RT_patchtst_v0_1_test_last7d_1step_6h.png")
    plot_last7d(
        ts_all=ts,
        rt_series=rt_series,
        t_dec_test=Tte,
        y_pred_test=yhat_test,
        out_path=plot_path,
        title=f"RT PatchTST baseline v0.1 (Test Last 7d: {pd.Timestamp(args.test_start).date()}~{pd.Timestamp(args.end).date()})",
    )

    # Save model
    torch.save(model.state_dict(), os.path.join(args.outdir, "model_rt_patchtst_v0_1.pt"))

    # Save summary.json (ensure JSON-serializable)
    summary = {
        "config": {
            "file": args.file,
            "sheet": args.sheet,
            "time_col": time_col,
            "rt_label_col": rt_label_col,
            "da_clearing_col": da_col,
            "split": vars(split),
            "L": args.L,
            "H": args.H,
            "patch_len": args.patch_len,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "layers": args.layers,
            "dim_ff": args.dim_ff,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "patience": args.patience,
            "seed": args.seed,
        },
        "features": {
            "C": int(C),
            "feat_names": feat_names,
            "exog_actual_cols": exog_actual_cols,
            "strict_no_pred_cols": True,
            "missing_policy": "ffill only; drop samples with NaN",
            "residual_learning": True,
            "prior": "persistence: rt(t_dec) replicated for all horizons",
        },
        "samples": {
            "train": {"X": list(Xtr.shape), "Y": list(Ytr.shape)},
            "val": {"X": list(Xva.shape), "Y": list(Yva.shape)},
            "test": {"X": list(Xte.shape), "Y": list(Yte.shape)},
        },
        "thresholds": {"train_p85": float(thr_p85), "train_p95": float(thr_p95)},
        "metrics": {
            "val_model": val_metrics,
            "val_prior": val_prior_metrics,
            "test_model": test_metrics,
            "test_prior": test_prior_metrics,
            "test_high_p85_model": test_high85,
            "test_high_p95_model": test_high95,
            "test_lead1_model": lead1,
            "test_lead1_prior": lead1_p,
            "test_lead24_model": lead24,
            "test_lead24_prior": lead24_p,
        },
        "artifacts": {
            "plot_last7d": os.path.basename(plot_path),
            "model": "model_rt_patchtst_v0_1.pt",
            "test_by_lead_model": "test_by_lead_metrics_model.csv",
            "test_by_lead_prior": "test_by_lead_metrics_prior.csv",
            "test_points": "test_point_predictions_lead1_lead24.csv",
        }
    }
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved outputs to: {args.outdir}")
    print(" - summary.json")
    print(" - test_by_lead_metrics_model.csv / test_by_lead_metrics_prior.csv")
    print(" - test_point_predictions_lead1_lead24.csv")
    print(" - RT_patchtst_v0_1_test_last7d_1step_6h.png")
    print(" - model_rt_patchtst_v0_1.pt")


if __name__ == "__main__":
    main()
