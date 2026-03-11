# -*- coding: utf-8 -*-
"""
双分支 RT 模型（ROADMAP 方案 3）：past branch + future exog branch，concat 后 head。
"""

import torch
import torch.nn as nn

from src.models.patchtst import PatchEmbedding


class RT_PatchTST_DualBranch(nn.Module):
    """
    Branch A: 历史 [B, L, C_past] -> PatchTST -> vec_a [B, d_model]
    Branch B: 未来外生 [B, H, 5] -> flatten -> MLP -> vec_b [B, d_future]
    融合: concat(vec_a, vec_b) -> MLP head -> 残差 [B, H]
    """

    def __init__(
        self,
        past_dim: int,
        L: int,
        H: int,
        patch_len: int = 16,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_ff: int = 256,
        dropout: float = 0.1,
        d_future: int = 64,
    ):
        super().__init__()
        self.L = L
        self.H = H
        self.past_dim = past_dim
        assert L % patch_len == 0
        n_patches = L // patch_len

        self.patch_embed = PatchEmbedding(past_dim, patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm_past = nn.LayerNorm(d_model)

        self.future_mlp = nn.Sequential(
            nn.Linear(H * 5, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, d_future),
        )

        self.head = nn.Sequential(
            nn.Linear(d_model + d_future, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, H),
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self, x_past: torch.Tensor, x_future: torch.Tensor
    ) -> torch.Tensor:
        """
        x_past: [B, L, C_past]
        x_future: [B, H, 5]
        """
        B = x_past.size(0)
        z = self.patch_embed(x_past) + self.pos_embed
        z = self.encoder(z)
        z = self.norm_past(z)
        vec_a = z.mean(dim=1)

        xf = x_future.reshape(B, -1)
        vec_b = self.future_mlp(xf)

        out = self.head(torch.cat([vec_a, vec_b], dim=1))
        return out
