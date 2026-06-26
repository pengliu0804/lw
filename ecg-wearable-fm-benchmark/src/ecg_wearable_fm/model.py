from __future__ import annotations

import torch
from torch import nn


class PatchTransformerECG(nn.Module):
    """Compact patch Transformer for 12-lead 10-second ECG."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 12,
        signal_length: int = 1000,
        patch_size: int = 25,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 192,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if signal_length % patch_size != 0:
            raise ValueError("signal_length must be divisible by patch_size")
        num_patches = signal_length // patch_size
        self.patch_embed = nn.Conv1d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x).transpose(1, 2)
        tokens = tokens + self.pos_embed[:, : tokens.shape[1]]
        encoded = self.encoder(tokens)
        pooled = self.norm(encoded.mean(dim=1))
        return pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))
