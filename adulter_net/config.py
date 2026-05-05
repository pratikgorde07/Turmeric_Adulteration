from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class Config:
    # === PATHS ===
    project_root: Path = field(default_factory=lambda: Path(__file__).parent)
    data_dir: Path = field(default=None)
    processed_dir: Path = field(default=None)
    output_dir: Path = field(default=None)
    log_dir: Path = field(default=None)

    # === FILE NAMES (from README — do not change) ===
    image_feat_file: str = "EfficientNetB0_Image_Features.csv"
    ftir_file: str = "FTIR_Data.csv"
    color_file: str = "Hunter_Lab_colorimeter.csv"
    image_data_dir: str = "Image_Data"

    # === COLUMN NAMES ===
    sample_id_col: str = "Sample_ID"
    target_col: str = "Target"

    # === DATASET ===
    n_classes: int = 7
    class_labels: list = field(default_factory=lambda: [0, 1, 2, 3, 5, 7, 9])
    class_names: list = field(default_factory=lambda: ["0%", "1%", "2%", "3%", "5%", "7%", "9%"])
    n_samples: int = 350
    samples_per_class: int = 50
    sample_id_prefix: str = "T"

    # === RAW IMAGE SETTINGS ===
    use_raw_images: bool = True
    image_size: tuple = (224, 224)
    image_channels: int = 3

    # === FEATURE DIMENSIONS ===
    img_feat_dim_raw: int = 1280
    visual_embed_dim: int = 256
    spectral_embed_dim: int = 128
    color_embed_dim: int = 64
    fused_dim: int = 512

    # === FEATURE SELECTION ===
    shap_boruta_top_k: int = 200
    cars_n_iterations: int = 80
    spa_n_components: int = 40
    mrmr_n_features: int = 20
    shap_elbow_threshold: float = 0.01

    # === MODEL ARCHITECTURE ===
    n_attention_heads: int = 8
    n_transformer_layers: int = 3
    ffn_dim: int = 1024
    dropout: float = 0.25
    efficientnet_finetune_blocks: int = 3

    # === TRAINING ===
    batch_size: int = 16
    lr_head: float = 1e-3
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    n_epochs: int = 150
    early_stop_patience: int = 20
    swa_start_epoch: int = 100
    swa_lr: float = 5e-5
    swa_anneal_epochs: int = 20
    n_folds: int = 5
    grad_clip_norm: float = 1.0

    # === LOSS ===
    alpha_loss: float = 0.75
    label_smoothing: float = 0.05
    focal_gamma: float = 2.5
    supcon_temperature: float = 0.07
    supcon_weight: float = 0.2
    mixup_alpha: float = 0.4

    # === TTA ===
    tta_n_passes: int = 10

    # === MC DROPOUT UNCERTAINTY ===
    mc_dropout_passes: int = 30

    # === EXPLAINABILITY ===
    gradcam_n_samples: int = 3
    shap_background_samples: int = 50

    # === REPRODUCIBILITY ===
    seed: int = 42

    # === COMPUTE ===
    device: str = field(default_factory=lambda: (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    ))
    num_workers: int = 4
    pin_memory: bool = True
    use_amp: bool = True

    # === W&B ===
    use_wandb: bool = False

    def __post_init__(self):
        # Set derived paths
        if self.data_dir is None:
            self.data_dir = self.project_root / "data" / "raw"
        if self.processed_dir is None:
            self.processed_dir = self.project_root / "data" / "processed"
        if self.output_dir is None:
            self.output_dir = self.project_root / "outputs"
        if self.log_dir is None:
            self.log_dir = self.project_root / "logs"

        # Create directories
        for d in [self.output_dir / "checkpoints", self.output_dir / "figures",
                  self.output_dir / "reports", self.log_dir,
                  self.processed_dir / "scalers", self.processed_dir / "selectors",
                  self.processed_dir / "splits"]:
            d.mkdir(parents=True, exist_ok=True)

        # Log device selection
        print(f"[Config] Device selected: {self.device}")

        # Validate class count
        assert len(self.class_labels) == self.n_classes, (
            f"class_labels length {len(self.class_labels)} != n_classes {self.n_classes}"
        )


# Singleton config instance
config = Config()
