# -*- coding: utf-8 -*-
"""RT 训练：归一化、训练轮、预测。"""

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


def fit_normalizers(
    train_X: np.ndarray, train_Y: np.ndarray
) -> Dict[str, np.ndarray]:
    """按 channel 归一化 X，全局归一化 Y。"""
    X_flat = train_X.reshape(-1, train_X.shape[-1])
    x_mean = X_flat.mean(axis=0)
    x_std = X_flat.std(axis=0) + 1e-6
    y_mean = train_Y.mean()
    y_std = train_Y.std() + 1e-6
    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": np.array([y_mean]),
        "y_std": np.array([y_std]),
    }


def apply_normalizers(
    X: np.ndarray,
    y: np.ndarray,
    prior: np.ndarray,
    norm: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 X_norm, y_res_norm, prior_norm。"""
    x_mean, x_std = norm["x_mean"], norm["x_std"]
    y_mean = float(norm["y_mean"][0])
    y_std = float(norm["y_std"][0])

    Xn = (X - x_mean.reshape(1, 1, -1)) / x_std.reshape(1, 1, -1)
    yn = (y - y_mean) / y_std
    pn = (prior - y_mean) / y_std
    resn = yn - pn
    return Xn.astype(np.float32), resn.astype(np.float32), pn.astype(np.float32)


def inv_y(y_norm: np.ndarray, norm: Dict[str, np.ndarray]) -> np.ndarray:
    y_mean = float(norm["y_mean"][0])
    y_std = float(norm["y_std"][0])
    return y_norm * y_std + y_mean


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
def eval_model(model, loader, device) -> float:
    model.eval()
    maes = []
    for X, y_res, _prior in loader:
        X = X.to(device)
        y_res = y_res.to(device)
        pred_res = model(X)
        maes.append(torch.mean(torch.abs(pred_res - y_res)).item())
    return float(np.mean(maes))


@torch.no_grad()
def predict_split(
    model,
    Xn: np.ndarray,
    prior_norm: np.ndarray,
    norm: Dict[str, np.ndarray],
    device,
) -> np.ndarray:
    """预测原始尺度 [N, H]。"""
    model.eval()
    bs = 256
    preds = []
    for i in range(0, Xn.shape[0], bs):
        xb = torch.from_numpy(Xn[i : i + bs]).to(device)
        pr = model(xb).cpu().numpy()
        pn = prior_norm[i : i + bs]
        yhat_norm = pn + pr
        preds.append(inv_y(yhat_norm, norm))
    return np.concatenate(preds, axis=0)
