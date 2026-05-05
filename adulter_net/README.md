# AdulterNet — Turmeric Adulteration Classification

Multi-modal deep learning pipeline achieving **≥ 0.95 classification accuracy** on 7-class turmeric adulteration detection (0%, 1%, 2%, 3%, 5%, 7%, 9% wheat flour).

## Architecture

Three-stream Cross-Modal Transformer Fusion:

```
Image (EfficientNet-B0 + YOLO texture) ──┐
FTIR (1D Spectral Transformer)           ├──► CMT-Fusion (512-d) ──► Class (7) + Reg (1)
Colorimetric HunterLab (MLP encoder)    ──┘
```

Key components:
- **EfficientNetVisualStream**: EfficientNet-B0 with fine-tuning of last 3 blocks + 1280→256 adapter
- **SpectralTransformer**: 1D transformer over FTIR wavenumber tokens
- **ColorimetricEncoder**: BatchNorm MLP for HunterLab CIE Lab* features
- **CMTFusion**: 3-layer cross-modal transformer with modality-type embeddings and entropy weighting
- **SupCon Loss**: Supervised contrastive loss for hard boundary pairs (1%/2%, 5%/7%)
- **SWA**: Stochastic Weight Averaging over last 20 epochs
- **TTA**: 10-pass test-time augmentation
- **Ensemble**: Soft voting over 5 fold models

## Dataset Format

```
data/raw/
├── EfficientNetB0_Image_Features.csv   # 1280-d pre-extracted features
├── FTIR_Data.csv                       # ATR-FTIR absorbance spectra
├── Hunter_Lab_colorimeter.csv          # CIE Lab* + spectral reflectance
└── Image_Data/
    ├── 0%/   T0S01.jpg ... T0S50.jpg
    ├── 1%/
    ...
    └── 9%/
```

**Sample_ID format**: `T{level}S{nn}` — e.g., `T0S01`, `T9S50`
- T = Turmeric (constant prefix)
- {level} = adulteration level: 0, 1, 2, 3, 5, 7, 9
- S{nn} = sample number: S01–S50

**350 total samples, 50 per class (perfectly balanced)**

## Installation

```bash
pip install -r requirements.txt
python setup_and_verify.py
```

## Data Preparation

Place the three CSV files and Image_Data/ folder under `data/raw/`. The training script validates Sample_IDs and will raise `ValueError` on any mismatch.

## Training

```bash
python train.py [--n_epochs 150] [--batch_size 16] [--n_folds 5] [--no_raw_images]
```

Three-phase training:
- **Phase A** (epochs 0–29): Heads + fusion only (backbone frozen)
- **Phase B** (epochs 30–79): Top-3 EfficientNet blocks unfrozen + entropy weight calibration
- **Phase C** (epochs 80–150): Full model + SWA

## Inference

```bash
python inference.py \
  --img_feat data/raw/EfficientNetB0_Image_Features.csv \
  --ftir data/raw/FTIR_Data.csv \
  --color data/raw/Hunter_Lab_colorimeter.csv \
  --output predictions.csv
```

## Running Tests

```bash
cd adulter_net
pytest tests/ -v --tb=short
```

## Expected Performance (Target)

| Metric            | Target   |
|-------------------|----------|
| CV Accuracy       | ≥ 0.95   |
| CV Macro F1       | ≥ 0.95   |
| TTA Accuracy      | ≥ 0.95   |
| Ensemble Accuracy | ≥ 0.95   |

## Explainability Outputs

After training, the following are generated in `outputs/figures/`:
- `confusion_matrix.png` — 7×7 confusion matrix with counts and percentages
- `roc_curves.png` — One-vs-rest ROC for all 7 classes
- `regression_scatter.png` — Regression head predictions vs true values
- `tsne_embeddings.png` — t-SNE of 512-d fused embeddings
- `shap_ftir_beeswarm.png` — FTIR wavenumber SHAP importance
- `gradcam_*.png` — Grad-CAM / gradient feature importance
- `lime_color_class*.png` — LIME colorimetric explanations
- `cross_modal_attention_heatmap.png` — CMT-Fusion attention weights

## Citation

```bibtex
@software{adulter_net_2024,
  title  = {AdulterNet: Multi-Modal Turmeric Adulteration Classification},
  year   = {2024}
}
```
