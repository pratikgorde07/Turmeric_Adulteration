# AdulterNet — Turmeric Adulteration Classification

## Overview

AdulterNet is a tri-modal deep learning pipeline for detecting wheat flour adulteration in turmeric powder across 7 concentration classes (0%, 1%, 2%, 3%, 5%, 7%, 9% w/w). It fuses image features, FTIR spectra, and colorimetric measurements via a Cross-Modal Transformer (CMT) to achieve ≥ 0.95 classification accuracy.

## Architecture

```
Image Stream      FTIR Stream       Color Stream
(EfficientNet-B0  (1D Spectral      (Colorimetric
 + YOLO texture)   Transformer)      MLP Encoder)
     ↓ (256-d)         ↓ (128-d)          ↓ (64-d)
     └────────────────────────────────────┘
                        ↓
            Cross-Modal Transformer Fusion
                     (512-d)
                        ↓
              ┌─────────┴──────────┐
        Classification          Regression
          Head (7-class)       Head (0–9%)
```

Key techniques: SHAP-Boruta / CARS+SPA / mRMR+SHAP feature selection, SupCon loss, Mixup augmentation, SWA, TTA (10 passes), 5-fold ensemble, temperature scaling calibration, and pseudo-label refinement.

## Dataset Format

```
data/raw/
├── EfficientNetB0_Image_Features.csv   # 1280-d deep features
├── FTIR_Data.csv                        # ATR-FTIR spectra
├── Hunter_Lab_colorimeter.csv           # CIE L*a*b* + reflectance
└── Image_Data/
    ├── 0%/    (50 images)
    ├── 1%/    ...
    └── 9%/
```

**Sample_ID format:** `T{level}S{nn}` — e.g., `T0S01`, `T9S50`.

All CSVs must contain `Sample_ID` and `Target` columns.

## Installation

```bash
pip install -r requirements.txt
```

## Data Preparation

Copy your data files into `data/raw/` matching the exact filenames above. Then verify:

```bash
python setup_and_verify.py
```

## Training

```bash
python train.py --n_epochs 150 --batch_size 16 --n_folds 5
```

Options:
- `--data_dir PATH` — override data directory
- `--output_dir PATH` — override output directory
- `--skip_baselines` — skip baseline model training
- `--use_wandb` — enable Weights & Biases logging

## Inference

```bash
python inference.py \
    --img_feat data/raw/EfficientNetB0_Image_Features.csv \
    --ftir data/raw/FTIR_Data.csv \
    --color data/raw/Hunter_Lab_colorimeter.csv \
    --output predictions.csv
```

## Reproducing Results

```bash
python setup_and_verify.py          # verify environment
python train.py                     # full 5-fold CV
pytest tests/ -v                    # run unit tests
```

## Expected Performance

| Model              | Accuracy (mean±std) | Macro-F1 |
|--------------------|---------------------|----------|
| AdulterNet (full)  | ≥ 0.95              | ≥ 0.95   |
| AdulterNet + TTA   | ≥ 0.96              | ≥ 0.96   |
| AdulterNet + Ensemble | ≥ 0.97           | ≥ 0.97   |

## Explainability Outputs

After training, find in `outputs/figures/`:
- `confusion_matrix.png` — 7×7 OOF confusion matrix
- `roc_curves.png` — One-vs-rest ROC for all 7 classes
- `tsne_embeddings.png` — t-SNE of 512-d fused embeddings
- `shap_ftir_beeswarm.png` — SHAP importances for FTIR wavenumbers
- `gradcam_*.png` — Gradient saliency maps
- `lime_color_class*.png` — LIME explanations for colorimetric stream
- `cross_modal_attention_heatmap.png` — Cross-modal attention weights
- `learning_curve_fold*.png` — Per-fold training curves

## Citation

```bibtex
@misc{adulter_net_2024,
  title  = {AdulterNet: Tri-Modal Deep Learning for Turmeric Adulteration Detection},
  year   = {2024},
}
```
