"""Cross-Modal Transformer Fusion."""
from __future__ import annotations
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CMTFusion(nn.Module):
    def __init__(self, visual_dim: int = 256, spectral_dim: int = 128,
                 color_dim: int = 64, d_model: int = 256,
                 n_heads: int = 8, n_layers: int = 3, ffn_dim: int = 1024,
                 dropout: float = 0.1, out_dim: int = 512):
        super().__init__()
        self.d_model = d_model
        self.out_dim = out_dim

        # Projections to common dimension
        # Visual is already 256-d: no projection needed if visual_dim == d_model
        self.v_proj = nn.Identity() if visual_dim == d_model else nn.Linear(visual_dim, d_model)
        self.s_proj = nn.Linear(spectral_dim, d_model)
        self.c_proj = nn.Linear(color_dim, d_model)

        # Modality-type embeddings
        self.modality_embed = nn.Embedding(3, d_model)

        # Cross-modal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Entropy scaling weights (set via set_entropy_weights)
        self.w_v = nn.Parameter(torch.ones(1), requires_grad=False)
        self.w_s = nn.Parameter(torch.ones(1), requires_grad=False)
        self.w_c = nn.Parameter(torch.ones(1), requires_grad=False)

        # Output projection: 3 * d_model -> out_dim
        self.output_proj = nn.Sequential(
            nn.Linear(3 * d_model, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.last_attn_weights: Optional[torch.Tensor] = None

    def set_entropy_weights(self, w_v: float, w_s: float, w_c: float) -> None:
        self.w_v.data.fill_(w_v)
        self.w_s.data.fill_(w_s)
        self.w_c.data.fill_(w_c)
        logger.info(f"Entropy weights set: v={w_v:.3f}, s={w_s:.3f}, c={w_c:.3f}")

    def get_last_attention_weights(self) -> Optional[torch.Tensor]:
        return self.last_attn_weights

    def forward(self, v: torch.Tensor, s: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        v: (B, visual_dim), s: (B, spectral_dim), c: (B, color_dim)
        -> (B, out_dim)
        """
        B = v.shape[0]
        dev = v.device

        v_proj = self.v_proj(v)  # (B, d_model)
        s_proj = self.s_proj(s)  # (B, d_model)
        c_proj = self.c_proj(c)  # (B, d_model)

        # Add modality embeddings
        mod_ids = torch.arange(3, device=dev)  # [0,1,2]
        mod_emb = self.modality_embed(mod_ids)  # (3, d_model)
        v_proj = v_proj + mod_emb[0]
        s_proj = s_proj + mod_emb[1]
        c_proj = c_proj + mod_emb[2]

        # Stack to sequence: (B, 3, d_model)
        tokens = torch.stack([v_proj, s_proj, c_proj], dim=1)

        # Transformer
        tokens = self.transformer(tokens)  # (B, 3, d_model)

        # Entropy scaling
        tokens[:, 0] = tokens[:, 0] * self.w_v
        tokens[:, 1] = tokens[:, 1] * self.w_s
        tokens[:, 2] = tokens[:, 2] * self.w_c

        # Concatenate and project
        cat = tokens.reshape(B, 3 * self.d_model)
        out = self.output_proj(cat)  # (B, out_dim)
        assert out.shape[1] == self.out_dim, f"Expected (B,{self.out_dim}), got {out.shape}"
        return out
