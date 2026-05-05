from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from src import set_all_seeds, get_logger
from src.tta import TTA_TRANSFORMS, tta_predict

logger = get_logger(__name__)


class AdulterNetInference:
    """Standalone inference using ensemble of trained fold models."""

    def __init__(self, config, weights_dir: str, scalers_dir: str, selectors_dir: str, n_folds: int = 5):
        self.config = config
        self.weights_dir = Path(weights_dir)
        self.scalers_dir = Path(scalers_dir)
        self.selectors_dir = Path(selectors_dir)
        self._load_scalers()
        self._load_selectors()
        self._load_models(n_folds)

    def _load_scalers(self) -> None:
        import joblib
        try:
            self.scaler_img = joblib.load(self.scalers_dir / "scaler_image_fold0.pkl")
            self.scaler_ftir = joblib.load(self.scalers_dir / "scaler_ftir_minmax_fold0.pkl")
            self.scaler_color = joblib.load(self.scalers_dir / "scaler_color_fold0.pkl")
            logger.info("Scalers loaded (fold 0)")
        except Exception as e:
            logger.warning(f"Could not load scalers: {e}")
            self.scaler_img = self.scaler_ftir = self.scaler_color = None

    def _load_selectors(self) -> None:
        import joblib
        try:
            self.sel_img = joblib.load(self.selectors_dir / "shap_boruta_fold0.pkl")
            self.sel_cars = joblib.load(self.selectors_dir / "cars_fold0.pkl")
            self.sel_spa = joblib.load(self.selectors_dir / "spa_fold0.pkl")
            self.sel_color = joblib.load(self.selectors_dir / "mrmr_shap_fold0.pkl")
            logger.info("Selectors loaded (fold 0)")
        except Exception as e:
            logger.warning(f"Could not load selectors: {e}")
            self.sel_img = self.sel_cars = self.sel_spa = self.sel_color = None

    def _load_models(self, n_folds: int) -> None:
        from src.models.adulter_net import AdulterNet
        self.models = []
        for fold_idx in range(n_folds):
            path = self.weights_dir / f"fold{fold_idx}_best.pt"
            if not path.exists():
                logger.warning(f"Model not found: {path}")
                continue
            n_ftir = getattr(self.sel_spa, "n_components", 40) if self.sel_spa else 40
            n_color = (len(getattr(self.sel_color, "selected_indices_", [None] * 15))
                       if self.sel_color else 15)
            model = AdulterNet(self.config, n_ftir_features=n_ftir, n_color_features=n_color)
            checkpoint = torch.load(path, map_location=self.config.device)
            model.load_state_dict(checkpoint.get("model_state", checkpoint))
            model.eval().to(self.config.device)
            self.models.append(model)
        logger.info(f"Loaded {len(self.models)} fold models")

    def preprocess(self, img_feat: np.ndarray, ftir: np.ndarray, color: np.ndarray) -> Dict:
        from src.data.preprocessing import apply_image_scaler, apply_ftir_pipeline, apply_color_scaler
        x_img = img_feat.reshape(1, -1).astype(np.float32)
        x_ftir = ftir.reshape(1, -1).astype(np.float32)
        x_color = color.reshape(1, -1).astype(np.float32)
        if self.scaler_img is not None:
            x_img = apply_image_scaler(x_img, self.scaler_img)
        if self.scaler_ftir is not None:
            x_ftir = apply_ftir_pipeline(x_ftir, self.scaler_ftir)
        if self.scaler_color is not None:
            x_color = apply_color_scaler(x_color, self.scaler_color)
        if self.sel_img is not None:
            x_img = self.sel_img.transform(x_img)
        if self.sel_cars is not None and self.sel_spa is not None:
            x_ftir = self.sel_spa.transform(self.sel_cars.transform(x_ftir))
        if self.sel_color is not None:
            x_color = self.sel_color.transform(x_color)
        return {
            "image_feat": torch.from_numpy(x_img).to(self.config.device),
            "ftir": torch.from_numpy(x_ftir).to(self.config.device),
            "color": torch.from_numpy(x_color).to(self.config.device),
            "label": torch.zeros(1, dtype=torch.long).to(self.config.device),
            "label_float": torch.zeros(1).to(self.config.device),
            "sample_id": ["unknown"],
            "image_raw": torch.zeros(1, 3, 224, 224).to(self.config.device),
        }

    def predict(self, img_feat: np.ndarray, ftir: np.ndarray, color: np.ndarray) -> Dict:
        if not self.models:
            raise RuntimeError("No models loaded")
        batch = self.preprocess(img_feat, ftir, color)
        all_probs = []
        for model in self.models:
            with torch.no_grad():
                out = model(batch)
                all_probs.append(F.softmax(out["logits"], dim=-1))
        mean_probs = torch.stack(all_probs).mean(0)
        pred_idx = mean_probs.argmax(-1).item()
        return {
            "predicted_class_idx": pred_idx,
            "predicted_level_pct": self.config.class_labels[pred_idx],
            "confidence": round(mean_probs.max(-1).values.item(), 4),
            "class_probabilities": {
                self.config.class_names[i]: round(p, 4)
                for i, p in enumerate(mean_probs.squeeze().tolist())
            },
        }

    def predict_batch(self, df_img: pd.DataFrame, df_ftir: pd.DataFrame, df_color: pd.DataFrame) -> pd.DataFrame:
        img_cols = [c for c in df_img.columns if c not in {"Sample_ID", "Target"}]
        ftir_cols = [c for c in df_ftir.columns if c not in {"Sample_ID", "Target"}]
        color_cols = [c for c in df_color.columns if c not in {"Sample_ID", "Target"}]
        ids = df_img.get("Sample_ID", pd.Series(range(len(df_img))))
        results = []
        for i in range(len(df_img)):
            pred = self.predict(df_img[img_cols].values[i], df_ftir[ftir_cols].values[i],
                                df_color[color_cols].values[i])
            pred["Sample_ID"] = str(ids.iloc[i])
            results.append(pred)
        return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="AdulterNet Inference")
    parser.add_argument("--img_feat", required=True)
    parser.add_argument("--ftir", required=True)
    parser.add_argument("--color", required=True)
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--weights_dir", default="outputs/checkpoints")
    parser.add_argument("--scalers_dir", default="data/processed/scalers")
    parser.add_argument("--selectors_dir", default="data/processed/selectors")
    args = parser.parse_args()

    set_all_seeds(42)
    from config import config

    for path, name in [(args.img_feat, "image"), (args.ftir, "ftir"), (args.color, "color")]:
        df = pd.read_csv(path)
        assert "Sample_ID" in df.columns, f"{name} CSV missing Sample_ID column"

    df_img = pd.read_csv(args.img_feat)
    df_ftir = pd.read_csv(args.ftir)
    df_color = pd.read_csv(args.color)

    engine = AdulterNetInference(config, args.weights_dir, args.scalers_dir, args.selectors_dir)
    results = engine.predict_batch(df_img, df_ftir, df_color)
    results.to_csv(args.output, index=False)
    logger.info(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
