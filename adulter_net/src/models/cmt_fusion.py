from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import get_logger

logger = get_logger(__name__)


class CMTFusion(nn.Module):
    """Cross-Modal Transformer Fusion for three modalities. Output: (B, 512)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        D = 256  # common projection dimension

        # Project spectral (128) and color (64) to D=256
        self.s_proj = nn.Linear(128, D)
        self.c_proj = nn.Linear(64, D)

        # Modality-type embeddings
        self.modality_embed = nn.Embedding(3, D)

        # Cross-modal transformer (3 layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=config.n_attention_heads,
            dim_feedforward=config.ffn_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_transformer_layers
        )

        # Entropy scaling weights (set after Phase A, not gradient-trained)
        self.register_buffer("w_v", torch.ones(1))
        self.register_buffer("w_s", torch.ones(1))
        self.register_buffer("w_c", torch.ones(1))

        # Output projection: 768 -> 512
        self.output_proj = nn.Sequential(
            nn.Linear(D * 3, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.last_attn_weights: Optional[torch.Tensor] = None

    def set_entropy_weights(self, w_v: float, w_s: float, w_c: float) -> None:
        self.w_v.fill_(w_v)
        self.w_s.fill_(w_s)
        self.w_c.fill_(w_c)
        logger.info(f"CMT entropy weights set: v={w_v:.3f}, s={w_s:.3f}, c={w_c:.3f}")

    def get_last_attention_weights(self) -> Optional[torch.Tensor]:
        return self.last_attn_weights

    def forward(
        self,
        v: torch.Tensor,  # (B, 256) visual
        s: torch.Tensor,  # (B, 128) spectral
        c: torch.Tensor,  # (B, 64) color
    ) -> torch.Tensor:
        B = v.shape[0]

        # Step 1: Project all to D=256
        v_proj = v  # already 256
        s_proj = self.s_proj(s)  # (B, 256)
        c_proj = self.c_proj(c)  # (B, 256)

        # Step 2: Add modality-type embeddings
        device = v.device
        v_proj = v_proj + self.modality_embed(torch.tensor(0, device=device))
        s_proj = s_proj + self.modality_embed(torch.tensor(1, device=device))
        c_proj = c_proj + self.modality_embed(torch.tensor(2, device=device))

        # Step 3: Stack as sequence (B, 3, 256)
        tokens = torch.stack([v_proj, s_proj, c_proj], dim=1)

        # Step 4: Apply entropy scaling weights
        tokens[:, 0] = tokens[:, 0] * self.w_v
        tokens[:, 1] = tokens[:, 1] * self.w_s
        tokens[:, 2] = tokens[:, 2] * self.w_c

        # Step 5: Cross-modal self-attention
        tokens = self.transformer(tokens)  # (B, 3, 256)

        # Store attention weights (approximate — last layer softmax)
        try:
            with torch.no_grad():
                # Compute attention approximation for explainability
                q = tokens
                attn_score = (q @ q.transpose(-2, -1)) / (256 ** 0.5)
                self.last_attn_weights = torch.softmax(attn_score, dim=-1).detach()
        except Exception:
            self.last_attn_weights = None

        # Step 6: Concatenate and project (B, 3*256) -> (B, 512)
        cat = tokens.reshape(B, -1)  # (B, 768)
        out = self.output_proj(cat)  # (B, 512)

        assert out.shape == (B, 512), f"Expected (B,512), got {out.shape}"
        return out
