# -*- coding: utf-8 -*-
"""PyTorch Dataset：RT seq2seq。"""

import numpy as np
import torch
from torch.utils.data import Dataset


class RTSeq2SeqDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        Y_res_norm: np.ndarray,
        prior_norm: np.ndarray,
        t_dec: np.ndarray,
    ):
        self.X = X.astype(np.float32)
        self.Y = Y_res_norm.astype(np.float32)
        self.prior = prior_norm.astype(np.float32)
        self.t_dec = t_dec

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.X[idx]),
            torch.from_numpy(self.Y[idx]),
            torch.from_numpy(self.prior[idx]),
        )


class RTDualBranchDataset(Dataset):
    """双分支：past [N,L,C] + future exog [N,H,5]。"""

    def __init__(
        self,
        X_past: np.ndarray,
        X_future: np.ndarray,
        Y_res_norm: np.ndarray,
        prior_norm: np.ndarray,
    ):
        self.X_past = X_past.astype(np.float32)
        self.X_future = X_future.astype(np.float32)
        self.Y = Y_res_norm.astype(np.float32)
        self.prior = prior_norm.astype(np.float32)

    def __len__(self) -> int:
        return self.X_past.shape[0]

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.X_past[idx]),
            torch.from_numpy(self.X_future[idx]),
            torch.from_numpy(self.Y[idx]),
            torch.from_numpy(self.prior[idx]),
        )
