# -*- coding: utf-8 -*-
"""通用工具：随机种子、指标、目录。"""

import os
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def bias(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a - b))


def safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)
