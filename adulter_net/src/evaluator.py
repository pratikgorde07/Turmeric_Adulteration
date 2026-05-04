"""Evaluation and metrics aggregation."""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, k: int = 2) -> float:
    top_k = np.argsort(y_proba, axis=1)[:, -k:]
    correct = sum(y_true[i] in top_k[i] for i in range(len(y_true)))
    return correct / len(y_true)


def aggregate_oof_predictions(fold_results: List[dict], config) -> dict:
    """Stack OOF predictions from all folds."""
    all_true, all_pred, all_proba, all_reg, all_ids = [], [], [], [], []
    all_conf, all_unc = [], []

    for fold in fold_results:
        all_true.extend(fold["y_true"])
        all_pred.extend(fold["y_pred"])
        all_proba.extend(fold["y_proba"])
        all_reg.extend(fold["y_reg"])
        all_ids.extend(fold["sample_ids"])
        all_conf.extend(fold.get("confidence", [0.0] * len(fold["y_true"])))
        all_unc.extend(fold.get("uncertainty", [0.0] * len(fold["y_true"])))

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    y_proba = np.array(all_proba)
    y_reg = np.array(all_reg)
    assert len(y_true) == config.n_samples, (
        f"Expected {config.n_samples} OOF samples, got {len(y_true)}"
    )
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "y_reg": y_reg,
        "sample_ids": all_ids,
        "confidence": np.array(all_conf),
        "uncertainty": np.array(all_unc),
    }


def compute_full_metrics(oof: dict, config) -> dict:
    from sklearn.metrics import (
        f1_score, cohen_kappa_score, matthews_corrcoef, classification_report
    )
    y_true = oof["y_true"]
    y_pred = oof["y_pred"]
    y_proba = oof["y_proba"]
    y_reg = oof["y_reg"]

    acc = (y_true == y_pred).mean()
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    top2_acc = top_k_accuracy(y_true, y_proba, k=2)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    y_reg_true = np.array([config.class_labels[l] for l in y_true])
    rmse = float(np.sqrt(np.mean((y_reg - y_reg_true) ** 2)))
    mae = float(np.mean(np.abs(y_reg - y_reg_true)))

    report = classification_report(
        y_true, y_pred,
        target_names=config.class_names,
        digits=4,
    )

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "top2_accuracy": float(top2_acc),
        "kappa": float(kappa),
        "mcc": float(mcc),
        "rmse": float(rmse),
        "mae": float(mae),
        "classification_report": report,
    }


def save_classification_report(metrics: dict, fold_accs: List[float], config) -> None:
    report_path = Path(config.output_dir) / "reports" / "classification_report.txt"
    acc_mean = np.mean(fold_accs)
    acc_std = np.std(fold_accs)
    lines = [
        "=== AdulterNet — Classification Report (5-Fold CV, n=350) ===\n",
        metrics["classification_report"],
        f"\nCohen's Kappa           : {metrics['kappa']:.4f}",
        f"Matthews Corr Coeff     : {metrics['mcc']:.4f}",
        f"Top-2 Accuracy          : {metrics['top2_accuracy']:.4f}",
        f"Mean Uncertainty (std)  : {0.0:.4f}",
        f"\nPer-fold accuracy: {[round(a, 4) for a in fold_accs]}",
        f"Mean ± Std:  {acc_mean:.4f} ± {acc_std:.4f}",
    ]
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Classification report saved: {report_path}")


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, config) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm_norm, annot=False, fmt="", cmap="RdYlGn",
                    xticklabels=config.class_names, yticklabels=config.class_names, ax=ax)

        # Annotate with count + percent
        for i in range(len(cm)):
            for j in range(len(cm[0])):
                ax.text(
                    j + 0.5, i + 0.5,
                    f"{cm[i,j]}\n{cm_norm[i,j]*100:.1f}%",
                    ha="center", va="center", fontsize=9,
                    color="white" if cm_norm[i, j] < 0.5 else "black",
                )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("AdulterNet 5-Fold OOF Predictions (n=350)")
        plt.tight_layout()
        out_path = Path(config.output_dir) / "figures" / "confusion_matrix.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Confusion matrix saved: {out_path}")
    except Exception as e:
        logger.warning(f"Could not save confusion matrix: {e}")


def save_roc_curves(y_true: np.ndarray, y_proba: np.ndarray, config) -> None:
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize

        y_bin = label_binarize(y_true, classes=list(range(config.n_classes)))
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, config.n_classes))
        for i, (name, color) in enumerate(zip(config.class_names, colors)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("One-vs-Rest ROC Curves")
        ax.legend(loc="lower right")
        plt.tight_layout()
        out_path = Path(config.output_dir) / "figures" / "roc_curves.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"ROC curves saved: {out_path}")
    except Exception as e:
        logger.warning(f"Could not save ROC curves: {e}")


def save_regression_scatter(y_true_cls: np.ndarray, y_reg: np.ndarray, config) -> None:
    try:
        import matplotlib.pyplot as plt
        y_true_pct = np.array([config.class_labels[i] for i in y_true_cls])
        fig, ax = plt.subplots(figsize=(8, 7))
        colors = plt.cm.tab10(np.linspace(0, 1, config.n_classes))
        for i, (lbl, color) in enumerate(zip(config.class_labels, colors)):
            mask = y_true_cls == i
            jitter = np.random.uniform(-0.05, 0.05, mask.sum())
            ax.scatter(y_true_pct[mask] + jitter, y_reg[mask], color=color,
                       alpha=0.6, s=30, label=config.class_names[i])
        ax.plot([0, 9], [0, 9], "k-", lw=2, label="y=x")
        ax.fill_between([0, 9], [-1, 8], [1, 10], alpha=0.1, color="gray", label="±1% band")
        rmse = float(np.sqrt(np.mean((y_reg - y_true_pct) ** 2)))
        mae = float(np.mean(np.abs(y_reg - y_true_pct)))
        r2 = float(1 - np.var(y_reg - y_true_pct) / np.var(y_true_pct))
        ax.set_xlabel("True Adulteration (%)")
        ax.set_ylabel("Predicted Adulteration (%)")
        ax.set_title(f"Regression Head — RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}")
        ax.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        out_path = Path(config.output_dir) / "figures" / "regression_scatter.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"Regression scatter saved: {out_path}")
    except Exception as e:
        logger.warning(f"Could not save regression scatter: {e}")


def save_tsne(embeddings: np.ndarray, labels: np.ndarray, config) -> None:
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=config.seed)
        emb_2d = tsne.fit_transform(embeddings)
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, config.n_classes))
        for i, (name, color) in enumerate(zip(config.class_names, colors)):
            mask = labels == i
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=[color], label=name,
                       alpha=0.7, s=40, edgecolors="none")
        ax.set_title("t-SNE of Fused Embeddings (512-d)")
        ax.legend(loc="best")
        plt.tight_layout()
        out_path = Path(config.output_dir) / "figures" / "tsne_embeddings.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        logger.info(f"t-SNE plot saved: {out_path}")
    except Exception as e:
        logger.warning(f"Could not save t-SNE: {e}")


def save_final_summary(metrics: dict, fold_accs: List[float], config,
                       ensemble_acc: float = 0.0, model_params: int = 0) -> None:
    acc_mean = float(np.mean(fold_accs))
    summary = {
        "target_achieved": acc_mean >= 0.95,
        "cv_accuracy_mean": acc_mean,
        "cv_accuracy_std": float(np.std(fold_accs)),
        "cv_macro_f1_mean": metrics.get("macro_f1", 0.0),
        "cv_macro_f1_std": 0.0,
        "cv_rmse_mean": metrics.get("rmse", 0.0),
        "cv_kappa": metrics.get("kappa", 0.0),
        "cv_mcc": metrics.get("mcc", 0.0),
        "ensemble_accuracy": ensemble_acc,
        "ensemble_macro_f1": 0.0,
        "n_samples": config.n_samples,
        "n_classes": config.n_classes,
        "n_folds": config.n_folds,
        "total_model_params": model_params,
        "training_device": config.device,
    }
    out_path = Path(config.output_dir) / "reports" / "final_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Final summary saved: {out_path}")
    return summary
