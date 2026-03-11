# -*- coding: utf-8 -*-
"""
未来 24 步 exog5 预测轨迹：供 RT 双分支模型使用（ROADMAP 第一步）。

按 RT 的日历划分训练外生模型，对全量 t_dec 生成 [H, 5] 预测，便于与 RT 样本对齐。
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.split import SplitConfig


# 与 run_exog_suite 一致
EXOG_KEY5 = [
    "系统负荷实际值",
    "光伏实际值",
    "联络线实际值",
    "上旋备用实际值",
    "下旋备用实际值",
]
BEST_METHODS: Dict[str, str] = {
    "系统负荷实际值": "hgbt_residual",
    "光伏实际值": "hgbt_residual",
    "联络线实际值": "use_pred",
    "上旋备用实际值": "hgbt_pure",
    "下旋备用实际值": "hgbt_pure",
}


def _rt_calendar_mask(
    dt: np.ndarray, split: SplitConfig, step_minutes: int, H: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按 RT 日历划分：t_dec 落在 train/val/test 的 mask。"""
    from pandas import Timestamp

    t_max = pd.to_datetime(dt) + pd.to_timedelta(step_minutes * H, unit="m")
    train_end = Timestamp(split.val_start) - pd.Timedelta(minutes=1)
    val_end = Timestamp(split.test_start) - pd.Timedelta(minutes=1)
    test_end = Timestamp(split.end) + pd.Timedelta(days=1)

    m_train = (pd.to_datetime(dt) <= train_end) & (t_max >= Timestamp(split.train_start))
    m_val = (pd.to_datetime(dt) <= val_end) & (pd.to_datetime(dt) >= Timestamp(split.val_start))
    m_test = (pd.to_datetime(dt) >= Timestamp(split.test_start)) & (pd.to_datetime(dt) <= test_end)

    return m_train, m_val, m_test


def build_future_exog_for_rt(
    df: pd.DataFrame,
    time_col: str,
    lookback: int,
    H: int,
    step_minutes: int,
    split: SplitConfig,
    target_cols: Optional[List[str]] = None,
    best_methods: Optional[Dict[str, str]] = None,
    ridge_alpha: float = 10.0,
    feat_nan_strategy: str = "ffill_bfill",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    为与 RT 对齐的 t_dec 生成未来 H 步的 5 条外生预测 [N, H, 5]。

    使用 RT 日历划分：仅在 train 时段训练外生模型，对全量样本预测。
    Returns:
      t_dec: [N] datetime64，与 exog 样本顺序一致
      pred: [N, H, 5] float32
    """
    from src.exog.run import (
        apply_feat_nan_strategy,
        build_samples,
        find_pred_col,
    )
    from sklearn.linear_model import Ridge
    from sklearn.multioutput import MultiOutputRegressor

    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        _HGBT = True
    except Exception:
        _HGBT = False

    target_cols = target_cols or list(EXOG_KEY5)
    best_methods = best_methods or dict(BEST_METHODS)

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    # 用第一个 target 得到参考 t_dec 网格（与 RT 同 lookback/H 时一致）
    tc0 = target_cols[0]
    pred_col0 = find_pred_col(df, tc0)
    if pred_col0:
        df = apply_feat_nan_strategy(df, [pred_col0], feat_nan_strategy)
    X0, Y0, DT_ref, PF0 = build_samples(
        df, time_col, tc0, pred_col0, lookback, H
    )
    if len(DT_ref) == 0:
        return np.array([], dtype="datetime64[ns]"), np.zeros((0, H, 5), dtype=np.float32)

    m_tr, m_va, m_te = _rt_calendar_mask(DT_ref, split, step_minutes, H)
    N = len(DT_ref)
    out = np.zeros((N, H, 5), dtype=np.float32)
    dt_ref_pd = pd.to_datetime(DT_ref)

    def _hgbt_est():
        return HistGradientBoostingRegressor(
            max_depth=6, learning_rate=0.08, max_iter=300, random_state=0
        )

    for col_idx, target_col in enumerate(target_cols):
        if target_col not in df.columns:
            continue
        pred_col = find_pred_col(df, target_col)
        if pred_col and pred_col not in df.columns:
            pred_col = None
        if pred_col:
            df = apply_feat_nan_strategy(df, [pred_col], feat_nan_strategy)
        X, Y, DT_t, PF = build_samples(df, time_col, target_col, pred_col, lookback, H)
        dt_t_pd = pd.to_datetime(DT_t)

        method = best_methods.get(target_col, "ridge_residual")

        if method == "use_pred" and PF is not None:
            for i in range(N):
                j = np.where(dt_t_pd == dt_ref_pd[i])[0]
                if len(j):
                    out[i, :, col_idx] = PF[j[0]].astype(np.float32)
            continue

        m_tr_t, _, _ = _rt_calendar_mask(DT_t, split, step_minutes, H)
        if method == "ridge_pure":
            model = Ridge(alpha=ridge_alpha, random_state=0)
            model.fit(X[m_tr_t], Y[m_tr_t])
            pred_t = model.predict(X)
        elif method == "ridge_residual" and PF is not None:
            Xr = np.concatenate([X, PF], axis=1)
            Yres = Y - PF
            model = Ridge(alpha=ridge_alpha, random_state=0)
            model.fit(Xr[m_tr_t], Yres[m_tr_t])
            pred_t = PF + model.predict(Xr)
        elif method == "hgbt_pure" and _HGBT:
            model = MultiOutputRegressor(_hgbt_est(), n_jobs=1)
            model.fit(X[m_tr_t], Y[m_tr_t])
            pred_t = model.predict(X)
        elif method == "hgbt_residual" and PF is not None and _HGBT:
            Xr = np.concatenate([X, PF], axis=1)
            Yres = Y - PF
            model = MultiOutputRegressor(_hgbt_est(), n_jobs=1)
            model.fit(Xr[m_tr_t], Yres[m_tr_t])
            pred_t = PF + model.predict(Xr)
        else:
            model = Ridge(alpha=ridge_alpha, random_state=0)
            model.fit(X[m_tr_t], Y[m_tr_t])
            pred_t = model.predict(X)

        pred_t = np.asarray(pred_t, dtype=np.float32)
        for i in range(N):
            j = np.where(dt_t_pd == dt_ref_pd[i])[0]
            if len(j):
                out[i, :, col_idx] = pred_t[j[0]]

    return DT_ref, out


def align_future_exog_to_rt_samples(
    rt_t_dec: np.ndarray,
    exog_t_dec: np.ndarray,
    exog_pred: np.ndarray,
) -> np.ndarray:
    """
    将 exog 预测按 RT 样本的 t_dec 对齐，返回 [N_rt, H, 5]。
    exog_t_dec 与 exog_pred 一一对应；rt_t_dec 可能为 RT 的 train/val/test 子集。
    """
    from pandas import Series

    rt_dt = pd.to_datetime(rt_t_dec)
    exog_dt = pd.to_datetime(exog_t_dec)
    # 精确匹配
    exog_series = Series(range(len(exog_dt)), index=exog_dt)
    idx = exog_series.reindex(rt_dt).values
    valid = ~np.isnan(idx)
    out = np.full((len(rt_t_dec), exog_pred.shape[1], exog_pred.shape[2]), np.nan, dtype=np.float32)
    out[valid] = exog_pred[idx[valid].astype(int)]
    return out
