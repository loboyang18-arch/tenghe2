# -*- coding: utf-8 -*-
"""评估与绘图：指标、按 lead、last7d、高电价区间。"""

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import bias, mae, rmse, safe_makedirs


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
        rows.append(
            {
                "lead_idx": h + 1,
                "lead_minutes": 15 * (h + 1),
                "MAE": mae(yp, yt),
                "RMSE": rmse(yp, yt),
                "Bias": bias(yp, yt),
                "n": int(len(yt)),
            }
        )
    return pd.DataFrame(rows)


def plot_last7d(
    ts_all: pd.DatetimeIndex,
    rt_series: pd.Series,
    t_dec_test: np.ndarray,
    y_pred_test: np.ndarray,
    out_path: str,
    title: str,
) -> None:
    pred1 = {}
    pred24 = {}
    for i, tdec in enumerate(pd.to_datetime(t_dec_test)):
        t1 = tdec + pd.Timedelta(minutes=15)
        t24 = tdec + pd.Timedelta(hours=6)
        pred1[t1] = float(y_pred_test[i, 0])
        pred24[t24] = float(y_pred_test[i, 23])

    s_pred1 = pd.Series(pred1)
    s_pred24 = pd.Series(pred24)

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


def high_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, thr: float
) -> Dict[str, float]:
    mask = y_true.reshape(-1) >= thr
    if mask.sum() == 0:
        return {
            "thr": thr,
            "n": 0,
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "Bias": float("nan"),
        }
    yt = y_true.reshape(-1)[mask]
    yp = y_pred.reshape(-1)[mask]
    return {
        "thr": float(thr),
        "n": int(mask.sum()),
        "MAE": mae(yp, yt),
        "RMSE": rmse(yp, yt),
        "Bias": bias(yp, yt),
    }
