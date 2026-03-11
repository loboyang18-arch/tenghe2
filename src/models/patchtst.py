# -*- coding: utf-8 -*-
"""PatchTST 模型（RT 残差预测）。"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_dim: int, patch_len: int, d_model: int):
        super().__init__()
        self.in_dim = in_dim
        self.patch_len = patch_len
        self.proj = nn.Linear(in_dim * patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        P = self.patch_len
        assert L % P == 0, f"L({L}) must be divisible by patch_len({P})"
        n_patches = L // P
        x = x.reshape(B, n_patches, P * C)
        x = self.proj(x)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.patch_embed(x) + self.pos_embed
        z = self.encoder(z)
        z = self.norm(z)
        pooled = z.mean(dim=1)
        out = self.head(pooled)
        return out
