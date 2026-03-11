# -*- coding: utf-8 -*-
"""数据读取、列检测、特征矩阵构造（RT）。"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.feature_engineering import build_time_features


def detect_time_col(df: pd.DataFrame) -> str:
    candidates = [
        "datetime",
        "时间",
        "日期时间",
        "时间点",
        "timestamp",
        "Timestamp",
        "DATETIME",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            return c
    raise ValueError("Cannot detect time column. Please specify via --time_col")


def detect_rt_label_col(df: pd.DataFrame) -> str:
    candidates = [
        "实时出清电价",
        "实时出清价格",
        "实时电价",
        "实时价格",
        "RT出清电价",
        "RT出清价格",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if ("实时" in c) and ("出清" in c) and ("价" in c):
            return c
    raise ValueError(
        "Cannot detect RT label column (实时出清电价). Please specify via --rt_label_col"
    )


def detect_da_clearing_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "日前出清电价",
        "日前出清价格",
        "日前电价",
        "日前价格",
        "DA出清电价",
        "DA出清价格",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if ("日前" in c) and ("出清" in c) and ("价" in c):
            return c
    return None


def is_numeric_series(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        return True
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
      feat_mat: [T, C]
      rt_y: [T]
      feat_names: list length C
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    ts = pd.to_datetime(df[time_col])
    df = df.set_index(ts)

    numeric_cols = [rt_label_col] + ([da_col] if da_col else []) + exog_actual_cols
    df = coerce_numeric(df, [c for c in numeric_cols if c is not None])
    df[numeric_cols] = df[numeric_cols].ffill()

    tfeats = build_time_features(df.index)

    feat_list = []
    feat_names = []

    feat_list.append(df[rt_label_col].values.astype(np.float32).reshape(-1, 1))
    feat_names.append("rt_current")

    if da_col:
        feat_list.append(df[da_col].values.astype(np.float32).reshape(-1, 1))
        feat_names.append("da_clearing")

    for c in exog_actual_cols:
        feat_list.append(df[c].values.astype(np.float32).reshape(-1, 1))
        feat_names.append(c)

    feat_list.append(tfeats.astype(np.float32))
    feat_names.extend(
        [
            "hour",
            "quarter",
            "weekday",
            "month",
            "is_weekend",
            "hour_sin",
            "hour_cos",
            "w_sin",
            "w_cos",
            "m_sin",
            "m_cos",
        ]
    )

    feat_mat = np.concatenate(feat_list, axis=1)
    rt_y = df[rt_label_col].values.astype(np.float32)
    return df.index, feat_mat, rt_y, feat_names


def select_exog_actual_cols(
    df: pd.DataFrame,
    time_col: str,
    rt_label_col: str,
    da_col: Optional[str],
    preferred_exog_order: Optional[List[str]] = None,
) -> List[str]:
    """自动筛选外生列：非空、数值、不含「预测」、非时间/RT/DA。优先顺序可由 preferred_exog_order 配置。"""
    from src.features_config import DEFAULT_PREFERRED_EXOG_ORDER

    banned_substr = ["预测"]
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

    preferred = (
        list(preferred_exog_order)
        if preferred_exog_order
        else list(DEFAULT_PREFERRED_EXOG_ORDER)
    )
    cols_sorted = []
    for p in preferred:
        if p in cols:
            cols_sorted.append(p)
    for c in cols:
        if c not in cols_sorted:
            cols_sorted.append(c)
    return cols_sorted


def select_exog_actual_cols_exog5(
    df: pd.DataFrame,
    time_col: str,
    rt_label_col: str,
    da_col: Optional[str],
    candidate_exog: Optional[List[str]] = None,
) -> List[str]:
    """仅保留指定的外生列（默认 key5）；candidate_exog 可由配置提供。"""
    from src.features_config import DEFAULT_EXOG_KEY5

    if candidate_exog:
        list_exog = list(candidate_exog)
    else:
        list_exog = list(DEFAULT_EXOG_KEY5)
    banned = {time_col, rt_label_col}
    if da_col:
        banned.add(da_col)

    chosen: List[str] = []
    for c in list_exog:
        if c not in df.columns or c in banned:
            continue
        s = df[c]
        if s.isna().all() or not is_numeric_series(s):
            continue
        chosen.append(c)
    return chosen
