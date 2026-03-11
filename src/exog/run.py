# -*- coding: utf-8 -*-
"""
exog run: Cfg through eval_one_target.
Use split_by_target_max_date from src.split.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

try:
    from sklearn.ensemble import HistGradientBoostingRegressor

    _HAS_HGBT = True
except Exception:
    _HAS_HGBT = False

from src.split import split_by_target_max_date


@dataclass
class Cfg:
    excel: str
    sheet: Optional[str]
    time_col: str
    target_cols: List[str]
    step_minutes: int
    lookback: int
    H: int
    val_days: int
    test_days: int
    feat_nan_strategy: str  # none|ffill|ffill_bfill
    ridge_alpha: float
    models: List[str]
    out_dir: str
    tag: str


def setup_matplotlib_fonts():
    """Best-effort: reduce CJK glyph warnings. Safe if fails."""
    try:
        candidates = [
            "Microsoft YaHei",
            "SimHei",
            "Noto Sans CJK SC",
            "Source Han Sans SC",
            "PingFang SC",
            "WenQuanYi Zen Hei",
            "Arial Unicode MS",
        ]
        for f in candidates:
            matplotlib.rcParams["font.sans-serif"] = [f] + matplotlib.rcParams.get(
                "font.sans-serif", []
            )
            fig = plt.figure()
            plt.text(0.5, 0.5, "测试值", ha="center")
            plt.close(fig)
            break
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def safe_name(s: str) -> str:
    return re.sub(r'[\\/:*?"<>| ]+', "_", str(s)).strip("_")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def rmse_np(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def read_excel_any(excel: str, sheet: Optional[str]) -> pd.DataFrame:
    """
    Robust Excel loader:
    - If sheet is None -> sheet_name=None returns dict; pick first sheet.
    - If sheet is digit -> treat as sheet index.
    """
    if sheet is None:
        df_or = pd.read_excel(excel, sheet_name=None)
        if isinstance(df_or, dict):
            first_name = list(df_or.keys())[0]
            print(f"[Excel] sheet_name=None -> dict. Using first sheet: {first_name}")
            return df_or[first_name]
        return df_or

    try:
        if str(sheet).isdigit():
            sheet_idx = int(sheet)
            xls = pd.ExcelFile(excel)
            sheet_name = xls.sheet_names[sheet_idx]
            return pd.read_excel(excel, sheet_name=sheet_name)
    except Exception:
        pass

    return pd.read_excel(excel, sheet_name=sheet)


def apply_feat_nan_strategy(
    df: pd.DataFrame, cols: List[str], strategy: str
) -> pd.DataFrame:
    """Apply NaN handling ONLY for feature columns (never for label/target)."""
    cols_u, seen = [], set()
    for c in cols:
        if c in df.columns and c not in seen:
            cols_u.append(c)
            seen.add(c)

    if not cols_u or strategy == "none":
        return df

    df2 = df.copy()
    for c in cols_u:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
        if strategy in ("ffill", "ffill_bfill"):
            df2[c] = df2[c].ffill()
        if strategy == "ffill_bfill":
            df2[c] = df2[c].bfill()
    return df2


def find_pred_col(df: pd.DataFrame, target_col: str) -> Optional[str]:
    """Try to locate matching '预测值' column for a given '实际值' target."""
    if "实际值" in target_col:
        cand = target_col.replace("实际值", "预测值")
        if cand in df.columns:
            return cand

    cand2 = (
        target_col.replace("值", "预测值")
        if target_col.endswith("值")
        else (target_col + "预测值")
    )
    if cand2 in df.columns:
        return cand2

    return None


def build_seasonal_median_map(
    train_times: pd.Series, train_values: pd.Series
) -> Dict[Tuple[int, int], float]:
    """Map (weekday, minute_of_day) -> median(actual) computed on TRAIN only."""
    ts = pd.to_datetime(train_times)
    v = pd.to_numeric(train_values, errors="coerce")
    m = pd.DataFrame({"ts": ts, "v": v}).dropna()
    if m.empty:
        return {}
    m["wd"] = m["ts"].dt.weekday.astype(int)
    m["mod"] = (m["ts"].dt.hour * 60 + m["ts"].dt.minute).astype(int)
    g = m.groupby(["wd", "mod"])["v"].median()
    return {(int(wd), int(mod)): float(val) for (wd, mod), val in g.items()}


def seasonal_predict(
    med_map: Dict[Tuple[int, int], float], times: np.ndarray
) -> np.ndarray:
    ts = pd.to_datetime(times)
    wd = ts.weekday.astype(int)
    mod = (ts.hour * 60 + ts.minute).astype(int)
    out = np.empty(len(ts), dtype=float)
    global_med = float(np.median(list(med_map.values()))) if len(med_map) else 0.0
    for i in range(len(ts)):
        out[i] = med_map.get((int(wd[i]), int(mod[i])), global_med)
    return out


def build_samples(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
    pred_col: Optional[str],
    lookback: int,
    H: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Returns:
      X_hist: [N, lookback * n_hist_feats]
      Y:      [N, H]     (future actual)
      decision_times: [N]
      pred_future: [N, H] if pred_col exists else None

    Target actual is NOT imputed.
    Sample is kept only if:
      - history window has no NaN (for all hist feats)
      - future labels (Y row) has no NaN
    """
    ts = pd.to_datetime(df[time_col]).values
    y_raw = pd.to_numeric(df[target_col], errors="coerce").values.astype(float)

    hist_feats = [y_raw]
    pred_raw = None
    if pred_col is not None and pred_col in df.columns:
        pred_raw = pd.to_numeric(df[pred_col], errors="coerce").values.astype(float)
        hist_feats.append(pred_raw)

    T = len(df)
    X_list, Y_list, DT_list = [], [], []
    PF_list = [] if pred_raw is not None else None

    for i in range(lookback - 1, T - H):
        h0 = i - (lookback - 1)

        # history must be complete
        ok = True
        for arr in hist_feats:
            if np.any(np.isnan(arr[h0 : i + 1])):
                ok = False
                break
        if not ok:
            continue

        y_seq = y_raw[i + 1 : i + 1 + H]
        if np.any(np.isnan(y_seq)):
            continue

        x = np.stack([arr[h0 : i + 1] for arr in hist_feats], axis=0).reshape(-1)
        X_list.append(x)
        Y_list.append(y_seq)
        DT_list.append(ts[i])

        if pred_raw is not None:
            PF_list.append(pred_raw[i + 1 : i + 1 + H])

    if not X_list:
        n_hist_feats = len(hist_feats)
        X = np.zeros((0, lookback * n_hist_feats), dtype=float)
        Y = np.zeros((0, H), dtype=float)
        DT = np.array([], dtype="datetime64[ns]")
        PF = None if pred_raw is None else np.zeros((0, H), dtype=float)
        return X, Y, DT, PF

    X = np.stack(X_list).astype(float)
    Y = np.stack(Y_list).astype(float)
    DT = np.array(DT_list, dtype="datetime64[ns]")
    PF = None if pred_raw is None else np.stack(PF_list).astype(float)
    return X, Y, DT, PF


def plot_horizon_mae(mae_by_h: np.ndarray, out_path: str, title: str) -> None:
    plt.figure()
    x = np.arange(1, len(mae_by_h) + 1)
    plt.plot(x, mae_by_h)
    plt.xlabel("Horizon step (1..H)")
    plt.ylabel("MAE")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_last7d_h24(
    decision_times: np.ndarray,
    y_true_h24: np.ndarray,
    y_pred_h24: np.ndarray,
    out_path: str,
    title: str,
) -> None:
    dt = pd.to_datetime(decision_times)
    if len(dt) == 0:
        return
    last_day = dt.max().normalize()
    start = last_day - pd.Timedelta(days=6)
    mask = dt >= start
    dt2 = dt[mask]
    yt = y_true_h24[mask]
    yp = y_pred_h24[mask]
    plt.figure(figsize=(12, 4))
    plt.plot(dt2, yt, label="true")
    plt.plot(dt2, yp, label="pred")
    plt.xlabel("Decision time")
    plt.ylabel("Value (h24)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def export_long_csv(
    out_path: str,
    decision_times: np.ndarray,
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    step_minutes: int,
) -> None:
    dt = pd.to_datetime(decision_times)
    dt_arr = dt.to_pydatetime()
    N, H = Y_true.shape
    rows = []
    for i in range(N):
        base = dt_arr[i]
        for h in range(H):
            rows.append(
                {
                    "decision_time": base,
                    "target_time": base + pd.Timedelta(minutes=step_minutes * (h + 1)),
                    "horizon": h + 1,
                    "y_true": float(Y_true[i, h]),
                    "y_pred": float(Y_pred[i, h]),
                    "abs_err": float(abs(Y_true[i, h] - Y_pred[i, h])),
                }
            )
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    mae_by_h = np.mean(np.abs(y_true - y_pred), axis=0)
    return {
        "all_MAE": float(np.mean(mae_by_h)),
        "all_RMSE": rmse_np(y_true.reshape(-1), y_pred.reshape(-1)),
        "h24_MAE": float(mae_by_h[-1]),
        "h24_RMSE": rmse_np(y_true[:, -1], y_pred[:, -1]),
        "mae_by_h": [float(x) for x in mae_by_h],
        "count": int(y_true.shape[0]),
    }


def eval_one_target(cfg: Cfg, df_raw: pd.DataFrame, target_col: str) -> List[Dict]:
    out_rows = []
    if target_col not in df_raw.columns:
        print(f"[Skip] target not found: {target_col}")
        return out_rows

    df = df_raw.copy()
    df[cfg.time_col] = pd.to_datetime(df[cfg.time_col])
    df = df.sort_values(cfg.time_col).reset_index(drop=True)

    tgt_series = pd.to_numeric(df[target_col], errors="coerce")
    last_valid_idx = tgt_series.last_valid_index()
    if last_valid_idx is None:
        print(f"[Skip] target all NaN: {target_col}")
        return out_rows

    last_valid_time = df.loc[last_valid_idx, cfg.time_col]
    df = df[df[cfg.time_col] <= last_valid_time].reset_index(drop=True)
    print(
        f"[Target] {target_col} last_valid_time={last_valid_time} truncated_rows={len(df)}"
    )

    pred_col = find_pred_col(df, target_col)
    if pred_col is not None:
        print(f"[Target] {target_col} pred_col={pred_col}")
        df = apply_feat_nan_strategy(df, [pred_col], cfg.feat_nan_strategy)
    else:
        print(f"[Target] {target_col} pred_col=None")

    X, Y, DT, PF = build_samples(
        df, cfg.time_col, target_col, pred_col, cfg.lookback, cfg.H
    )
    print(f"[Samples] {target_col} N={len(DT)} (after dropping label-NaN windows)")
    if len(DT) == 0:
        return out_rows

    splits = split_by_target_max_date(
        DT, cfg.H, cfg.step_minutes, cfg.val_days, cfg.test_days
    )
    m_tr, m_te = splits["train"], splits["test"]
    print(
        f"[Split] train/val/test = {splits['train'].sum()}/{splits['val'].sum()}/{splits['test'].sum()}  (by target_max_date)"
    )

    train_end_date = pd.to_datetime(pd.Series(splits["train_dates"]).max())
    train_end_time = (
        pd.Timestamp(train_end_date.date())
        + pd.Timedelta(days=1)
        - pd.Timedelta(seconds=1)
    )
    df_train_for_seasonal = df[df[cfg.time_col] <= train_end_time]
    seasonal_map = build_seasonal_median_map(
        df_train_for_seasonal[cfg.time_col], df_train_for_seasonal[target_col]
    )

    base_out = os.path.join(cfg.out_dir, cfg.tag, safe_name(target_col))
    ensure_dir(base_out)

    # seasonal_median
    if "seasonal_median" in cfg.models:
        dt_te = pd.to_datetime(DT[m_te])
        Yhat = np.zeros((len(dt_te), cfg.H), dtype=float)
        for h in range(cfg.H):
            tgt_times = (
                dt_te + pd.to_timedelta(cfg.step_minutes * (h + 1), unit="m")
            ).values
            Yhat[:, h] = seasonal_predict(seasonal_map, tgt_times)
        metrics = compute_metrics(Y[m_te], Yhat)
        out_sub = os.path.join(base_out, "seasonal_median")
        ensure_dir(out_sub)
        json.dump(
            {"method": "seasonal_median", "target": target_col, **metrics},
            open(os.path.join(out_sub, "metrics.json"), "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=2,
        )
        plot_horizon_mae(
            np.array(metrics["mae_by_h"]),
            os.path.join(out_sub, "test_horizon_mae.png"),
            f"{target_col} seasonal_median horizon MAE",
        )
        plot_last7d_h24(
            DT[m_te],
            Y[m_te, -1],
            Yhat[:, -1],
            os.path.join(out_sub, "test_last7d_h24.png"),
            f"{target_col} seasonal_median h24 last7d",
        )
        export_long_csv(
            os.path.join(out_sub, "test_predictions_long.csv"),
            DT[m_te],
            Y[m_te],
            Yhat,
            cfg.step_minutes,
        )
        out_rows.append({"target": target_col, "method": "seasonal_median", **metrics})

    # persist
    if "persist" in cfg.models:
        last_obs = X[m_te, cfg.lookback - 1]
        Yhat = np.repeat(last_obs.reshape(-1, 1), cfg.H, axis=1)
        metrics = compute_metrics(Y[m_te], Yhat)
        out_sub = os.path.join(base_out, "persist")
        ensure_dir(out_sub)
        json.dump(
            {"method": "persist", "target": target_col, **metrics},
            open(os.path.join(out_sub, "metrics.json"), "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=2,
        )
        plot_horizon_mae(
            np.array(metrics["mae_by_h"]),
            os.path.join(out_sub, "test_horizon_mae.png"),
            f"{target_col} persist horizon MAE",
        )
        plot_last7d_h24(
            DT[m_te],
            Y[m_te, -1],
            Yhat[:, -1],
            os.path.join(out_sub, "test_last7d_h24.png"),
            f"{target_col} persist h24 last7d",
        )
        export_long_csv(
            os.path.join(out_sub, "test_predictions_long.csv"),
            DT[m_te],
            Y[m_te],
            Yhat,
            cfg.step_minutes,
        )
        out_rows.append({"target": target_col, "method": "persist", **metrics})

    # use_pred
    if "use_pred" in cfg.models:
        if PF is None:
            print(f"[Skip] use_pred (no pred col) for {target_col}")
        else:
            Yhat = PF[m_te].copy()
            if np.any(np.isnan(Yhat)):
                dt_te = pd.to_datetime(DT[m_te])
                for h in range(cfg.H):
                    miss = np.isnan(Yhat[:, h])
                    if miss.any():
                        tgt_times = (
                            dt_te
                            + pd.to_timedelta(cfg.step_minutes * (h + 1), unit="m")
                        ).values
                        Yhat[miss, h] = seasonal_predict(seasonal_map, tgt_times)[miss]
            metrics = compute_metrics(Y[m_te], Yhat)
            out_sub = os.path.join(base_out, "use_pred")
            ensure_dir(out_sub)
            json.dump(
                {
                    "method": "use_pred",
                    "target": target_col,
                    "pred_col": pred_col,
                    **metrics,
                },
                open(os.path.join(out_sub, "metrics.json"), "w", encoding="utf-8"),
                ensure_ascii=False,
                indent=2,
            )
            plot_horizon_mae(
                np.array(metrics["mae_by_h"]),
                os.path.join(out_sub, "test_horizon_mae.png"),
                f"{target_col} use_pred horizon MAE",
            )
            plot_last7d_h24(
                DT[m_te],
                Y[m_te, -1],
                Yhat[:, -1],
                os.path.join(out_sub, "test_last7d_h24.png"),
                f"{target_col} use_pred h24 last7d",
            )
            export_long_csv(
                os.path.join(out_sub, "test_predictions_long.csv"),
                DT[m_te],
                Y[m_te],
                Yhat,
                cfg.step_minutes,
            )
            out_rows.append({"target": target_col, "method": "use_pred", **metrics})

    # ridge_pure
    if "ridge_pure" in cfg.models:
        model = Ridge(alpha=cfg.ridge_alpha, random_state=0)
        model.fit(X[m_tr], Y[m_tr])
        Yhat = model.predict(X[m_te])
        metrics = compute_metrics(Y[m_te], Yhat)
        out_sub = os.path.join(base_out, "ridge_pure")
        ensure_dir(out_sub)
        json.dump(
            {
                "method": "ridge_pure",
                "target": target_col,
                "ridge_alpha": cfg.ridge_alpha,
                **metrics,
            },
            open(os.path.join(out_sub, "metrics.json"), "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=2,
        )
        plot_horizon_mae(
            np.array(metrics["mae_by_h"]),
            os.path.join(out_sub, "test_horizon_mae.png"),
            f"{target_col} ridge_pure horizon MAE",
        )
        plot_last7d_h24(
            DT[m_te],
            Y[m_te, -1],
            Yhat[:, -1],
            os.path.join(out_sub, "test_last7d_h24.png"),
            f"{target_col} ridge_pure h24 last7d",
        )
        export_long_csv(
            os.path.join(out_sub, "test_predictions_long.csv"),
            DT[m_te],
            Y[m_te],
            Yhat,
            cfg.step_minutes,
        )
        out_rows.append({"target": target_col, "method": "ridge_pure", **metrics})

    # ridge_residual
    if "ridge_residual" in cfg.models:
        if PF is None:
            print(f"[Skip] ridge_residual (no pred col) for {target_col}")
        else:
            Xr = np.concatenate([X, PF], axis=1)
            Yres = Y - PF
            model = Ridge(alpha=cfg.ridge_alpha, random_state=0)
            model.fit(Xr[m_tr], Yres[m_tr])
            Yhat = PF[m_te] + model.predict(Xr[m_te])
            metrics = compute_metrics(Y[m_te], Yhat)
            out_sub = os.path.join(base_out, "ridge_residual")
            ensure_dir(out_sub)
            json.dump(
                {
                    "method": "ridge_residual",
                    "target": target_col,
                    "pred_col": pred_col,
                    "ridge_alpha": cfg.ridge_alpha,
                    **metrics,
                },
                open(os.path.join(out_sub, "metrics.json"), "w", encoding="utf-8"),
                ensure_ascii=False,
                indent=2,
            )
            plot_horizon_mae(
                np.array(metrics["mae_by_h"]),
                os.path.join(out_sub, "test_horizon_mae.png"),
                f"{target_col} ridge_residual horizon MAE",
            )
            plot_last7d_h24(
                DT[m_te],
                Y[m_te, -1],
                Yhat[:, -1],
                os.path.join(out_sub, "test_last7d_h24.png"),
                f"{target_col} ridge_residual h24 last7d",
            )
            export_long_csv(
                os.path.join(out_sub, "test_predictions_long.csv"),
                DT[m_te],
                Y[m_te],
                Yhat,
                cfg.step_minutes,
            )
            out_rows.append(
                {"target": target_col, "method": "ridge_residual", **metrics}
            )

    # HGBT optional
    def _hgbt_est():
        return HistGradientBoostingRegressor(
            max_depth=6, learning_rate=0.08, max_iter=300, random_state=0
        )

    if "hgbt_pure" in cfg.models:
        if not _HAS_HGBT:
            print("[Skip] hgbt_pure: HistGradientBoostingRegressor not available.")
        else:
            model = MultiOutputRegressor(_hgbt_est(), n_jobs=1)
            model.fit(X[m_tr], Y[m_tr])
            Yhat = model.predict(X[m_te])
            metrics = compute_metrics(Y[m_te], Yhat)
            out_sub = os.path.join(base_out, "hgbt_pure")
            ensure_dir(out_sub)
            json.dump(
                {"method": "hgbt_pure", "target": target_col, **metrics},
                open(os.path.join(out_sub, "metrics.json"), "w", encoding="utf-8"),
                ensure_ascii=False,
                indent=2,
            )
            plot_horizon_mae(
                np.array(metrics["mae_by_h"]),
                os.path.join(out_sub, "test_horizon_mae.png"),
                f"{target_col} hgbt_pure horizon MAE",
            )
            plot_last7d_h24(
                DT[m_te],
                Y[m_te, -1],
                Yhat[:, -1],
                os.path.join(out_sub, "test_last7d_h24.png"),
                f"{target_col} hgbt_pure h24 last7d",
            )
            export_long_csv(
                os.path.join(out_sub, "test_predictions_long.csv"),
                DT[m_te],
                Y[m_te],
                Yhat,
                cfg.step_minutes,
            )
            out_rows.append({"target": target_col, "method": "hgbt_pure", **metrics})

    if "hgbt_residual" in cfg.models:
        if PF is None:
            print(f"[Skip] hgbt_residual (no pred col) for {target_col}")
        elif not _HAS_HGBT:
            print("[Skip] hgbt_residual: HistGradientBoostingRegressor not available.")
        else:
            Xr = np.concatenate([X, PF], axis=1)
            Yres = Y - PF
            model = MultiOutputRegressor(_hgbt_est(), n_jobs=1)
            model.fit(Xr[m_tr], Yres[m_tr])
            Yhat = PF[m_te] + model.predict(Xr[m_te])
            metrics = compute_metrics(Y[m_te], Yhat)
            out_sub = os.path.join(base_out, "hgbt_residual")
            ensure_dir(out_sub)
            json.dump(
                {
                    "method": "hgbt_residual",
                    "target": target_col,
                    "pred_col": pred_col,
                    **metrics,
                },
                open(os.path.join(out_sub, "metrics.json"), "w", encoding="utf-8"),
                ensure_ascii=False,
                indent=2,
            )
            plot_horizon_mae(
                np.array(metrics["mae_by_h"]),
                os.path.join(out_sub, "test_horizon_mae.png"),
                f"{target_col} hgbt_residual horizon MAE",
            )
            plot_last7d_h24(
                DT[m_te],
                Y[m_te, -1],
                Yhat[:, -1],
                os.path.join(out_sub, "test_last7d_h24.png"),
                f"{target_col} hgbt_residual h24 last7d",
            )
            export_long_csv(
                os.path.join(out_sub, "test_predictions_long.csv"),
                DT[m_te],
                Y[m_te],
                Yhat,
                cfg.step_minutes,
            )
            out_rows.append(
                {"target": target_col, "method": "hgbt_residual", **metrics}
            )

    json.dump(
        {
            "target": target_col,
            "last_valid_time": str(last_valid_time),
            "counts": {
                "train": int(m_tr.sum()),
                "val": int(splits["val"].sum()),
                "test": int(m_te.sum()),
            },
            "train_target_max_date_range": [
                str(splits["train_dates"][0]),
                str(splits["train_dates"][-1]),
            ],
            "val_target_max_date_range": [
                str(splits["val_dates"][0]),
                str(splits["val_dates"][-1]),
            ],
            "test_target_max_date_range": [
                str(splits["test_dates"][0]),
                str(splits["test_dates"][-1]),
            ],
        },
        open(os.path.join(base_out, "split_dates.json"), "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )

    return out_rows
