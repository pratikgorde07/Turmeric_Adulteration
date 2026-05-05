from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import get_logger

logger = get_logger(__name__)


def compute_full_metrics(y_true, y_pred, y_proba, y_reg, config) -> Dict:
    """Compute all classification and regression metrics."""
    from sklearn.metrics import (classification_report, cohen_kappa_score,
                                  matthews_corrcoef, f1_score)
    acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    try:
        kappa = float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        kappa = 0.0
    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        mcc = 0.0
    top_k = np.argsort(-y_proba, axis=1)[:, :2]
    top2_acc = float(np.mean([y_true[i] in top_k[i] for i in range(len(y_true))]))
    y_reg_true = np.array([config.class_labels[l] for l in y_true])
    rmse = float(np.sqrt(np.mean((y_reg - y_reg_true) ** 2)))
    mae = float(np.abs(y_reg - y_reg_true).mean())
    ss_res = np.sum((y_reg - y_reg_true) ** 2)
    ss_tot = np.sum((y_reg_true - y_reg_true.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-10))
    cls_report = classification_report(y_true, y_pred, target_names=config.class_names,
                                        zero_division=0, output_dict=True)
    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
            "top2_accuracy": top2_acc, "kappa": kappa, "mcc": mcc,
            "rmse": rmse, "mae": mae, "r2": r2, "classification_report": cls_report}


def save_classification_report(y_true, y_pred, y_proba, y_reg, fold_accs, config) -> None:
    from sklearn.metrics import classification_report
    metrics = compute_full_metrics(y_true, y_pred, y_proba, y_reg, config)
    lines = [
        "=== AdulterNet — Classification Report (5-Fold CV, n=350) ===", "",
        classification_report(y_true, y_pred, target_names=config.class_names, zero_division=0),
        f"Cohen's Kappa        : {metrics['kappa']:.4f}",
        f"Matthews Corr Coeff  : {metrics['mcc']:.4f}",
        f"Top-2 Accuracy       : {metrics['top2_accuracy']:.4f}",
        f"Regression RMSE      : {metrics['rmse']:.4f}",
        f"Regression R²        : {metrics['r2']:.4f}", "",
        f"Per-fold accuracy: {[round(a, 4) for a in fold_accs]}",
        f"Mean ± Std: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}",
    ]
    path = config.output_dir / "reports" / "classification_report.txt"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Classification report saved: {path}")


def save_confusion_matrix(y_true, y_pred, config) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm_norm, annot=False, cmap="Blues", ax=ax,
                    xticklabels=config.class_names, yticklabels=config.class_names)
        for i in range(len(config.class_names)):
            for j in range(len(config.class_names)):
                color = "green" if i == j else ("red" if cm[i, j] > 0 else "black")
                ax.text(j + 0.5, i + 0.3, str(cm[i, j]), ha="center", va="center",
                        fontsize=11, fontweight="bold", color=color)
                ax.text(j + 0.5, i + 0.7, f"{cm_norm[i,j]*100:.1f}%", ha="center",
                        va="center", fontsize=9, color="gray")
        ax.set_title("AdulterNet 5-Fold OOF Predictions (n=350)", fontsize=14)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        plt.tight_layout()
        path = config.output_dir / "figures" / "confusion_matrix.png"
        plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close()
        logger.info(f"Confusion matrix saved: {path}")
    except Exception as e:
        logger.warning(f"Could not save confusion matrix: {e}")


def save_roc_curves(y_true, y_proba, config) -> None:
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=list(range(config.n_classes)))
        colors = plt.cm.tab10(np.linspace(0, 1, config.n_classes))
        fig, ax = plt.subplots(figsize=(10, 8))
        mean_fpr = np.linspace(0, 1, 100)
        all_tprs = []
        for i, (cls_name, color) in enumerate(zip(config.class_names, colors)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{cls_name} (AUC={roc_auc:.3f})")
            interp_tpr = np.interp(mean_fpr, fpr, tpr); interp_tpr[0] = 0.0
            all_tprs.append(interp_tpr)
        mean_tpr = np.mean(all_tprs, axis=0); std_tpr = np.std(all_tprs, axis=0)
        mean_tpr[-1] = 1.0; mean_auc = auc(mean_fpr, mean_tpr)
        ax.plot(mean_fpr, mean_tpr, "k-", lw=3, label=f"Mean ROC (AUC={mean_auc:.3f})")
        ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.2, color="gray")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — AdulterNet (One-vs-Rest)")
        ax.legend(loc="lower right", fontsize=9); plt.tight_layout()
        path = config.output_dir / "figures" / "roc_curves.png"
        plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
        logger.info(f"ROC curves saved: {path}")
    except Exception as e:
        logger.warning(f"Could not save ROC curves: {e}")


def save_regression_scatter(y_true, y_reg, config) -> None:
    try:
        import matplotlib.pyplot as plt
        y_true_pct = np.array([config.class_labels[l] for l in y_true])
        colors = plt.cm.tab10(np.linspace(0, 1, config.n_classes))
        fig, ax = plt.subplots(figsize=(10, 8))
        for i, cls_name in enumerate(config.class_names):
            mask = y_true == i
            jitter = np.random.uniform(-0.05, 0.05, mask.sum())
            ax.scatter(y_true_pct[mask] + jitter, y_reg[mask], color=colors[i],
                       label=cls_name, alpha=0.7, s=40)
        ax.plot([0, 9], [0, 9], "k-", lw=2, label="Identity (y=x)")
        ax.fill_between([0, 9], [-1, 8], [1, 10], alpha=0.1, color="gray", label="±1% band")
        mae = float(np.abs(y_reg - y_true_pct).mean())
        rmse = float(np.sqrt(np.mean((y_reg - y_true_pct) ** 2)))
        ss_res = np.sum((y_reg - y_true_pct) ** 2)
        r2 = 1 - ss_res / (np.sum((y_true_pct - y_true_pct.mean()) ** 2) + 1e-10)
        ax.text(0.05, 0.95, f"RMSE={rmse:.3f}\nMAE={mae:.3f}\nR²={r2:.3f}",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(facecolor="white", alpha=0.8))
        ax.set_xlabel("True Adulteration %"); ax.set_ylabel("Predicted Adulteration %")
        ax.set_title("Regression Head Predictions vs True Values")
        ax.legend(loc="upper left", fontsize=8); ax.set_xlim(-0.5, 9.5); ax.set_ylim(-0.5, 9.5)
        plt.tight_layout()
        path = config.output_dir / "figures" / "regression_scatter.png"
        plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
        logger.info(f"Regression scatter saved: {path}")
    except Exception as e:
        logger.warning(f"Could not save regression scatter: {e}")


def save_tsne_embeddings(embeddings, labels, uncertainty, config) -> None:
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        logger.info("Running t-SNE on fused embeddings...")
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=config.seed)
        emb_2d = tsne.fit_transform(embeddings)
        fig, ax = plt.subplots(figsize=(12, 10))
        colors = plt.cm.tab10(np.linspace(0, 1, config.n_classes))
        for i, cls_name in enumerate(config.class_names):
            mask = labels == i
            sizes = 50 if uncertainty is None else 20 + 200 * (uncertainty[mask] / (uncertainty.max() + 1e-8))
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=[colors[i]], label=cls_name,
                       s=sizes, alpha=0.7)
        ax.set_title("t-SNE of Fused Embeddings (512-d) — AdulterNet")
        ax.legend(loc="best", fontsize=10); ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        plt.tight_layout()
        path = config.output_dir / "figures" / "tsne_embeddings.png"
        plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
        logger.info(f"t-SNE plot saved: {path}")
    except Exception as e:
        logger.warning(f"Could not save t-SNE: {e}")


def save_final_summary(metrics: Dict, config) -> None:
    path = config.output_dir / "reports" / "final_summary.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Final summary saved: {path}")
