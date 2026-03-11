# -*- coding: utf-8 -*-
"""时间特征等。"""

import numpy as np
import pandas as pd


def build_time_features(ts: pd.DatetimeIndex) -> np.ndarray:
    """11 维时间特征（与 RT v0 一致）。"""
    hour = ts.hour.values.astype(np.float32)
    minute = ts.minute.values.astype(np.float32)
    quarter = (minute // 15).astype(np.float32)
    weekday = ts.weekday.values.astype(np.float32)
    month = ts.month.values.astype(np.float32)
    is_weekend = (weekday >= 5).astype(np.float32)

    hour_sin = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)
    w_sin = np.sin(2 * np.pi * weekday / 7.0).astype(np.float32)
    w_cos = np.cos(2 * np.pi * weekday / 7.0).astype(np.float32)
    m_sin = np.sin(2 * np.pi * (month - 1) / 12.0).astype(np.float32)
    m_cos = np.cos(2 * np.pi * (month - 1) / 12.0).astype(np.float32)

    feats = np.stack(
        [
            hour,
            quarter,
            weekday,
            month,
            is_weekend,
            hour_sin,
            hour_cos,
            w_sin,
            w_cos,
            m_sin,
            m_cos,
        ],
        axis=1,
    )
    return feats
