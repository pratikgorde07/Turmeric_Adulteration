from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import logging


@dataclass
class Config:
    # === PATHS ===
    project_root: Path = field(default_factory=lambda: Path(__file__).parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "raw")
    processed_dir: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "processed")
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent / "outputs")
    log_dir: Path = field(default_factory=lambda: Path(__file__).parent / "logs")

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
    device: str = field(default_factory=lambda: _get_device())
    num_workers: int = 4
    pin_memory: bool = True
    use_amp: bool = True

    # === WANDB ===
    use_wandb: bool = False

    def __post_init__(self):
        import torch
        # Create directories
        for d in [self.output_dir, self.log_dir,
                  self.output_dir / "checkpoints",
                  self.output_dir / "figures",
                  self.output_dir / "reports",
                  self.processed_dir / "scalers",
                  self.processed_dir / "selectors",
                  self.processed_dir / "augmented",
                  self.processed_dir / "splits"]:
            Path(d).mkdir(parents=True, exist_ok=True)
        # Validate
        assert len(self.class_labels) == self.n_classes, (
            f"class_labels length {len(self.class_labels)} != n_classes {self.n_classes}"
        )
        print(f"[Config] Device: {self.device}")


def _get_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
