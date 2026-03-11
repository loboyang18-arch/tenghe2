# -*- coding: utf-8 -*-
"""
划分逻辑统一入口（P4）。

两种策略：
- RT 电价：按日历日期边界划分，使用 SplitConfig + build_samples_seq2seq。
- 外生预测：按 (decision_time + H*step) 的日期划分，使用 ExogSplitConfig + split_by_target_max_date。
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# 外生划分默认天数（与 configs/scripts 一致）
DEFAULT_EXOG_VAL_DAYS = 5
DEFAULT_EXOG_TEST_DAYS = 6


@dataclass
class SplitConfig:
    """RT 电价：按日历日期的 train/val/test 边界。"""

    train_start: str = "2025-01-01"
    val_start: str = "2025-11-01"
    test_start: str = "2025-12-01"
    end: str = "2025-12-30"


@dataclass
class ExogSplitConfig:
    """外生预测：按 target_max 日期，取最后 val_days 为 val、test_days 为 test。"""

    val_days: int = 5
    test_days: int = 6


def build_samples_seq2seq(
    ts: pd.DatetimeIndex,
    feat_mat: np.ndarray,
    rt_y: np.ndarray,
    split: SplitConfig,
    L: int,
    H: int,
) -> Dict[str, Dict[str, Any]]:
    """
    RT seq2seq 样本：按 split 日期落入 train/val/test。
    Returns: {"train"|"val"|"test": {"X", "Y", "prior", "t_dec"}}
    """
    assert feat_mat.shape[0] == len(ts) == len(rt_y)
    T, _ = feat_mat.shape

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

    buckets = {
        k: {"X": [], "Y": [], "prior": [], "t_dec": []}
        for k in ["train", "val", "test"]
    }

    for i in range(L - 1, T - H):
        t_dec = ts[i]
        name = split_name(pd.Timestamp(t_dec))
        if name is None:
            continue

        Xwin = feat_mat[i - L + 1 : i + 1, :]
        Yhor = rt_y[i + 1 : i + 1 + H].reshape(-1)
        prior = np.full((H,), rt_y[i], dtype=np.float32)

        if np.isnan(Xwin).any() or np.isnan(Yhor).any() or np.isnan(prior).any():
            continue

        buckets[name]["X"].append(Xwin)
        buckets[name]["Y"].append(Yhor)
        buckets[name]["prior"].append(prior)
        buckets[name]["t_dec"].append(np.datetime64(t_dec))

    out: Dict[str, Dict[str, Any]] = {}
    for k in ["train", "val", "test"]:
        if len(buckets[k]["X"]) == 0:
            raise RuntimeError(
                f"No samples built for split={k}. Check split ranges and missing data."
            )
        out[k] = {
            "X": np.stack(buckets[k]["X"], axis=0),
            "Y": np.stack(buckets[k]["Y"], axis=0),
            "prior": np.stack(buckets[k]["prior"], axis=0),
            "t_dec": np.array(buckets[k]["t_dec"]),
        }
    return out


def split_by_target_max_date(
    decision_times: np.ndarray,
    H: int,
    step_minutes: int,
    val_days: int,
    test_days: int,
) -> Dict[str, np.ndarray]:
    """
    外生预测：按 (decision_time + H*step) 的日期划分 train/val/test。
    返回 mask 与日期数组：train, val, test, train_dates, val_dates, test_dates。
    """
    dt = pd.to_datetime(decision_times)
    tgt_max = dt + pd.to_timedelta(step_minutes * H, unit="m")
    dates = pd.Series(tgt_max).dt.date.values
    uniq = np.array(sorted(pd.unique(dates)))
    if len(uniq) <= (val_days + test_days + 2):
        raise ValueError(
            f"Not enough unique target_max days ({len(uniq)}) for split: "
            f"val={val_days}, test={test_days}"
        )
    test_dates = uniq[-test_days:]
    val_dates = uniq[-(test_days + val_days) : -test_days]
    train_dates = uniq[: -(test_days + val_days)]
    m_train = np.isin(dates, train_dates)
    m_val = np.isin(dates, val_dates)
    m_test = np.isin(dates, test_dates)
    return {
        "train": m_train,
        "val": m_val,
        "test": m_test,
        "train_dates": train_dates,
        "val_dates": val_dates,
        "test_dates": test_dates,
    }
