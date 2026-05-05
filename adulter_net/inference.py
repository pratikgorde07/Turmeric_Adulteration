"""Standalone inference script for AdulterNet."""
from __future__ import annotations
import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(__file__))


class AdulterNetInference:
    def __init__(self, config, weights_dir: str, scalers_dir: str, selectors_dir: str,
                 fold_idx: int = 0):
        self.config = config
        self.weights_dir = Path(weights_dir)
        self.scalers_dir = Path(scalers_dir)
        self.selectors_dir = Path(selectors_dir)
        self.fold_idx = fold_idx

        import joblib
        import torch

        # Load scalers
        self.scaler_img = joblib.load(self.scalers_dir / f"scaler_image_fold{fold_idx}.pkl")
        self.scaler_ftir = joblib.load(self.scalers_dir / f"scaler_ftir_minmax_fold{fold_idx}.pkl")
        self.scaler_color = joblib.load(self.scalers_dir / f"scaler_color_fold{fold_idx}.pkl")

        # Load selectors
        self.sel_img = joblib.load(self.selectors_dir / f"shap_boruta_fold{fold_idx}.pkl")
        self.sel_cars = joblib.load(self.selectors_dir / f"cars_fold{fold_idx}.pkl")
        self.sel_spa = joblib.load(self.selectors_dir / f"spa_fold{fold_idx}.pkl")
        self.sel_color = joblib.load(self.selectors_dir / f"mrmr_shap_fold{fold_idx}.pkl")

        # Load model
        from src.models.adulter_net import AdulterNet
        ckpt_path = self.weights_dir / f"fold{fold_idx}_best.pt"
        checkpoint = torch.load(str(ckpt_path), map_location=config.device)
        model_kwargs = checkpoint.get("model_kwargs", {})
        self.model = AdulterNet(config, **model_kwargs)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.model.to(config.device)

    def preprocess(self, img_feat_vector, ftir_vector, color_vector):
        import numpy as np
        from src.data.preprocessing import apply_image_scaler, apply_ftir_pipeline, apply_color_scaler

        img = np.array(img_feat_vector).reshape(1, -1).astype(np.float32)
        ftir = np.array(ftir_vector).reshape(1, -1).astype(np.float32)
        color = np.array(color_vector).reshape(1, -1).astype(np.float32)

        img = apply_image_scaler(img, self.scaler_img)
        ftir = apply_ftir_pipeline(ftir, self.scaler_ftir)
        color = apply_color_scaler(color, self.scaler_color)

        img = self.sel_img.transform(img)
        ftir_cars = self.sel_cars.transform(ftir)
        ftir = self.sel_spa.transform(ftir_cars)
        color = self.sel_color.transform(color)

        import torch
        return {
            "image_feat": torch.from_numpy(img).float().to(self.config.device),
            "ftir": torch.from_numpy(ftir).float().to(self.config.device),
            "color": torch.from_numpy(color).float().to(self.config.device),
            "image_raw": torch.zeros(1, 3, 224, 224).to(self.config.device),
            "sample_id": ["unknown"],
        }

    def predict(self, img_feat_vector, ftir_vector, color_vector) -> dict:
        import torch
        import torch.nn.functional as F
        from src.models.adulter_net import mc_dropout_predict

        batch = self.preprocess(img_feat_vector, ftir_vector, color_vector)
        with torch.no_grad():
            out = self.model(batch)
            probs = F.softmax(out["logits"], dim=-1)

        pred_idx = int(probs.argmax(-1).item())
        confidence = float(probs.max().item())
        reg_est = float(out["reg_pred"].item())

        # MC dropout uncertainty
        mc_out = mc_dropout_predict(self.model, batch, self.config.mc_dropout_passes, self.config.device)
        uncertainty = float(mc_out["std_logits"].max().item())

        return {
            "predicted_class_idx": pred_idx,
            "predicted_level_pct": self.config.class_labels[pred_idx],
            "confidence": round(confidence, 4),
            "class_probabilities": {
                self.config.class_names[i]: round(float(p), 4)
                for i, p in enumerate(probs.squeeze().tolist())
            },
            "uncertainty_std": round(uncertainty, 4),
            "regression_estimate_pct": round(reg_est, 3),
        }

    def predict_batch(self, df_img, df_ftir, df_color, config):
        import pandas as pd
        import numpy as np

        records = []
        sample_ids = df_img[config.sample_id_col].tolist()
        img_cols = [c for c in df_img.columns if c not in [config.sample_id_col, config.target_col]]
        ftir_cols = [c for c in df_ftir.columns if c not in [config.sample_id_col, config.target_col]]
        color_cols = [c for c in df_color.columns if c not in [config.sample_id_col, config.target_col]]

        # Align on Sample_ID
        df_img = df_img.set_index(config.sample_id_col)
        df_ftir = df_ftir.set_index(config.sample_id_col)
        df_color = df_color.set_index(config.sample_id_col)

        for sid in sample_ids:
            try:
                img_v = df_img.loc[sid, img_cols].values.astype(np.float32)
                ftir_v = df_ftir.loc[sid, ftir_cols].values.astype(np.float32)
                color_v = df_color.loc[sid, color_cols].values.astype(np.float32)
                result = self.predict(img_v, ftir_v, color_v)
                result["Sample_ID"] = sid
                true_level = df_img.loc[sid, config.target_col] if config.target_col in df_img.columns else None
                result["True_Level"] = true_level
                records.append(result)
            except Exception as e:
                print(f"Prediction failed for {sid}: {e}")

        return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="AdulterNet Inference")
    parser.add_argument("--img_feat", type=str, required=True,
                        help="Path to EfficientNetB0_Image_Features.csv")
    parser.add_argument("--ftir", type=str, required=True, help="Path to FTIR_Data.csv")
    parser.add_argument("--color", type=str, required=True,
                        help="Path to Hunter_Lab_colorimeter.csv")
    parser.add_argument("--output", type=str, default="predictions.csv")
    parser.add_argument("--fold", type=int, default=0, help="Which fold model to use")
    args = parser.parse_args()

    from config import Config
    config = Config()

    import pandas as pd
    df_img = pd.read_csv(args.img_feat)
    df_ftir = pd.read_csv(args.ftir)
    df_color = pd.read_csv(args.color)

    for df, name in [(df_img, "img_feat"), (df_ftir, "ftir"), (df_color, "color")]:
        assert config.sample_id_col in df.columns, f"{name} missing {config.sample_id_col}"

    weights_dir = str(config.output_dir / "checkpoints")
    scalers_dir = str(config.processed_dir / "scalers")
    selectors_dir = str(config.processed_dir / "selectors")

    inference = AdulterNetInference(config, weights_dir, scalers_dir, selectors_dir, args.fold)
    df_pred = inference.predict_batch(df_img, df_ftir, df_color, config)
    df_pred.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
    print(df_pred.head())


if __name__ == "__main__":
    main()
