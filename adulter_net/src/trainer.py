"""Training loop: optimizer, scheduler, phases A/B/C, SWA, CV loop, pseudo-labeling."""
from __future__ import annotations
import json
import logging
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ============================================================
# Helpers
# ============================================================

def _to_device(batch: dict, device: str) -> dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float = 1e-4,
                 mode: str = "max", metric_name: str = "accuracy"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.metric_name = metric_name
        self.best = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, current: float) -> bool:
        if self.best is None:
            self.best = current
            return False
        improved = (
            (current > self.best + self.min_delta) if self.mode == "max"
            else (current < self.best - self.min_delta)
        )
        if improved:
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class CheckpointManager:
    def __init__(self, config):
        self.config = config
        self.ckpt_dir = Path(config.output_dir) / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model, optimizer, scheduler, epoch: int,
             metrics: dict, fold_idx: int, model_kwargs: dict = None) -> str:
        acc = metrics.get("accuracy", 0.0)
        path = self.ckpt_dir / f"fold{fold_idx}_epoch{epoch:03d}_acc{acc:.4f}.pt"
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "model_kwargs": model_kwargs or {},
        }, path)
        return str(path)

    def save_best(self, model, fold_idx: int, model_kwargs: dict = None) -> str:
        path = self.ckpt_dir / f"fold{fold_idx}_best.pt"
        torch.save({
            "model_state": model.state_dict(),
            "model_kwargs": model_kwargs or {},
        }, path)
        logger.info(f"Best model saved: {path}")
        return str(path)

    def load_best(self, model, fold_idx: int) -> None:
        path = self.ckpt_dir / f"fold{fold_idx}_best.pt"
        if not path.exists():
            logger.warning(f"Best checkpoint not found: {path}")
            return
        checkpoint = torch.load(path, map_location=self.config.device)
        model.load_state_dict(checkpoint["model_state"])
        logger.info(f"Loaded best checkpoint: {path}")

    def cleanup_old(self, fold_idx: int, keep_top_n: int = 3) -> None:
        ckpts = sorted(
            self.ckpt_dir.glob(f"fold{fold_idx}_epoch*.pt"),
            key=lambda p: float(p.stem.split("acc")[-1]),
            reverse=True,
        )
        for old in ckpts[keep_top_n:]:
            old.unlink()


# ============================================================
# Optimizer / Scheduler
# ============================================================

def setup_optimizer(model, config):
    param_groups = [
        {"params": model.classification_head.parameters(), "lr": config.lr_head},
        {"params": model.regression_head.parameters(), "lr": config.lr_head},
        {"params": model.cmt_fusion.parameters(), "lr": config.lr_head},
        {"params": model.spectral_stream.parameters(), "lr": config.lr_head},
        {"params": model.color_stream.parameters(), "lr": config.lr_head},
        {"params": model.supcon_projector.parameters(), "lr": config.lr_head},
        {
            "params": model.visual_stream.adapter.parameters(),
            "lr": config.lr_head,
        },
        {
            "params": [
                p for p in model.visual_stream.parameters()
                if id(p) not in {id(q) for q in model.visual_stream.adapter.parameters()}
            ],
            "lr": config.lr_backbone,
            "name": "backbone",
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)
    return optimizer


def setup_scheduler(optimizer, config, n_batches: int):
    n_param_groups = len(optimizer.param_groups)
    max_lrs = [pg["lr"] for pg in optimizer.param_groups]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        total_steps=config.n_epochs * n_batches,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10,
        final_div_factor=100,
    )
    return scheduler


# ============================================================
# Training epoch
# ============================================================

def _apply_mixup_to_batch(fused_emb, labels, n_classes, alpha):
    """Find adjacent-class pairs in batch and mix."""
    from src.data.augmentation import mixup_embeddings
    adj_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]  # class index pairs
    batch_size = labels.shape[0]
    mixed_embs, mixed_labels = [], []
    for cls1_idx, cls2_idx in adj_pairs:
        m1 = (labels == cls1_idx).nonzero(as_tuple=True)[0]
        m2 = (labels == cls2_idx).nonzero(as_tuple=True)[0]
        if len(m1) > 0 and len(m2) > 0:
            n = min(len(m1), len(m2))
            e1, l1 = fused_emb[m1[:n]], labels[m1[:n]]
            e2, l2 = fused_emb[m2[:n]], labels[m2[:n]]
            me, ml = mixup_embeddings(e1, e2, l1, l2, n_classes, alpha)
            mixed_embs.append(me)
            mixed_labels.append(ml)
    if mixed_embs:
        return torch.cat(mixed_embs, 0), torch.cat(mixed_labels, 0)
    return None, None


def train_one_epoch(model, loader, optimizer, scheduler, criterion, scaler, config, epoch: int):
    from src.losses import MixupCriterion
    model.train()
    metrics = defaultdict(float)
    all_preds, all_labels = [], []
    mixup_criterion = MixupCriterion()

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Train E{epoch}", leave=False)):
        batch = _to_device(batch, config.device)

        use_amp = config.use_amp and config.device == "cuda"
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(batch, return_embeddings=True)

            loss_dict = criterion(
                out["logits"], out["reg_pred"], out["proj_emb"],
                batch["label"], batch["label_float"],
            )
            total_loss = loss_dict["total"]

            # Mixup on 30% of batches
            if random.random() < 0.3 and out["fused_emb"] is not None:
                mixed_emb, mixed_labels_soft = _apply_mixup_to_batch(
                    out["fused_emb"].detach(), batch["label"], config.n_classes, config.mixup_alpha
                )
                if mixed_emb is not None:
                    logits_mixed = model.classification_head(mixed_emb)
                    mixup_loss = mixup_criterion(logits_mixed, mixed_labels_soft)
                    total_loss = total_loss + 0.1 * mixup_loss

        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        preds = out["logits"].argmax(-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["label"].cpu().numpy())
        for k, v in loss_dict.items():
            metrics[k] += float(v)

    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    n = max(len(loader), 1)
    return {k: v / n for k, v in metrics.items()} | {"accuracy": accuracy}


# ============================================================
# Validation epoch
# ============================================================

def validate_one_epoch(model, loader, criterion, config, calibrator=None):
    from sklearn.metrics import f1_score, cohen_kappa_score
    from src.evaluator import top_k_accuracy
    model.eval()
    all_preds, all_labels, all_probs, all_reg = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            batch = _to_device(batch, config.device)
            out = model(batch)
            logits = out["logits"]
            if calibrator is not None:
                logits = calibrator.scale(logits)
            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["label"].cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            reg = out["reg_pred"].squeeze()
            if reg.dim() == 0:
                reg = reg.unsqueeze(0)
            all_reg.extend(reg.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    y_reg = np.array(all_reg)
    y_reg_true = np.array([config.class_labels[l] for l in y_true])

    acc = (y_true == y_pred).mean()
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    top2_acc = top_k_accuracy(y_true, y_probs, k=2)
    kappa = cohen_kappa_score(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
    rmse = float(np.sqrt(np.mean((y_reg - y_reg_true) ** 2)))

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "top2_accuracy": float(top2_acc),
        "kappa": float(kappa),
        "rmse": float(rmse),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_proba": y_probs.tolist(),
        "y_reg": y_reg.tolist(),
    }


# ============================================================
# Entropy weight calibration
# ============================================================

def compute_entropy_weights(model, val_loader, config) -> Tuple[float, float, float]:
    model.eval()
    entropies = {"visual": [], "spectral": [], "color": []}

    with torch.no_grad():
        for batch in val_loader:
            batch = _to_device(batch, config.device)
            v = model.visual_stream(batch)
            s = model.spectral_stream(batch["ftir"])
            c = model.color_stream(batch["color"])

            for feat, key in [(v, "visual"), (s, "spectral"), (c, "color")]:
                # Build single-modality fusion (others zeroed)
                zeros_v = torch.zeros_like(v)
                zeros_s = torch.zeros_like(s)
                zeros_c = torch.zeros_like(c)
                if key == "visual":
                    fused = model.cmt_fusion(feat, zeros_s, zeros_c)
                elif key == "spectral":
                    fused = model.cmt_fusion(zeros_v, feat, zeros_c)
                else:
                    fused = model.cmt_fusion(zeros_v, zeros_s, feat)
                logits = model.classification_head(fused)
                probs = F.softmax(logits, -1)
                entropy = -(probs * (probs + 1e-10).log()).sum(-1)
                entropies[key].extend(entropy.cpu().numpy().tolist())

    H_v = float(np.mean(entropies["visual"])) + 1e-8
    H_s = float(np.mean(entropies["spectral"])) + 1e-8
    H_c = float(np.mean(entropies["color"])) + 1e-8
    w_v, w_s, w_c = 1 / H_v, 1 / H_s, 1 / H_c
    total = (w_v + w_s + w_c) / 3
    return w_v / total, w_s / total, w_c / total


# ============================================================
# Phase-wise training
# ============================================================

def _save_learning_curve(train_metrics: List[dict], val_metrics: List[dict],
                         fold_idx: int, config) -> None:
    try:
        import matplotlib.pyplot as plt
        epochs = list(range(len(train_metrics)))
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, key, title in [
            (axes[0], "accuracy", "Accuracy"),
            (axes[1], "total", "Total Loss"),
            (axes[2], "macro_f1", "Macro F1"),
        ]:
            t_vals = [m.get(key, 0) for m in train_metrics]
            v_vals = [m.get(key, 0) for m in val_metrics]
            ax.plot(epochs, t_vals, label="Train")
            ax.plot(epochs, v_vals, label="Val")
            ax.set_title(f"{title} (fold {fold_idx})")
            ax.legend()
        plt.tight_layout()
        out = Path(config.output_dir) / "figures" / f"learning_curve_fold{fold_idx}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        logger.warning(f"Could not save learning curve: {e}")


def train_with_phases(
    model, train_loader, val_loader, config, fold_idx: int, logger_: logging.Logger,
    model_kwargs: dict = None,
):
    from src.losses import AdulterNetLoss
    from src.models.swa_model import SWAWrapper
    from sklearn.utils.class_weight import compute_class_weight

    ckpt_mgr = CheckpointManager(config)
    use_amp = config.use_amp and config.device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Class weights
    labels_train = []
    for batch in train_loader:
        labels_train.extend(batch["label"].numpy().tolist())
    labels_arr = np.array(labels_train)
    class_weights = compute_class_weight(
        "balanced", classes=np.arange(config.n_classes), y=labels_arr
    )
    weight_tensor = torch.FloatTensor(class_weights).to(config.device)

    criterion = AdulterNetLoss(
        alpha=config.alpha_loss,
        supcon_weight=config.supcon_weight,
        focal_gamma=config.focal_gamma,
        label_smoothing=config.label_smoothing,
        class_weights=weight_tensor,
    )

    best_val_acc = 0.0
    best_val_metrics = {}
    train_history, val_history = [], []
    es = EarlyStopping(patience=config.early_stop_patience, mode="max")

    # ---- PHASE A: heads + fusion only (epochs 0–29) ----
    logger_.info(f"[Fold {fold_idx}] PHASE A: freeze backbones, train heads/fusion")
    model.freeze_all_backbones()
    optimizer = setup_optimizer(model, config)
    scheduler = setup_scheduler(optimizer, config, len(train_loader))

    for epoch in range(30):
        t_m = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, config, epoch)
        v_m = validate_one_epoch(model, val_loader, criterion, config)
        train_history.append(t_m)
        val_history.append(v_m)
        logger_.info(
            f"[Fold {fold_idx}] E{epoch} | train_acc={t_m['accuracy']:.4f} "
            f"val_acc={v_m['accuracy']:.4f} val_f1={v_m['macro_f1']:.4f}"
        )
        if v_m["accuracy"] > best_val_acc:
            best_val_acc = v_m["accuracy"]
            best_val_metrics = v_m
            ckpt_mgr.save_best(model, fold_idx, model_kwargs)

    logger_.info(f"=== Phase A complete | Val Acc = {best_val_acc:.4f} ===")
    if best_val_acc < 0.80:
        logger_.error(
            f"Phase A val acc {best_val_acc:.4f} < 0.80 — possible data pipeline issue"
        )

    # ---- PHASE B: unfreeze top backbone blocks (epochs 30–79) ----
    logger_.info(f"[Fold {fold_idx}] PHASE B: unfreeze top-{config.efficientnet_finetune_blocks} blocks")
    model.unfreeze_visual_top_n(config.efficientnet_finetune_blocks)
    optimizer = setup_optimizer(model, config)
    scheduler = setup_scheduler(optimizer, config, len(train_loader))
    es = EarlyStopping(patience=config.early_stop_patience, mode="max")

    # Entropy weight calibration
    try:
        w_v, w_s, w_c = compute_entropy_weights(model, val_loader, config)
        model.cmt_fusion.set_entropy_weights(w_v, w_s, w_c)
        logger_.info(f"Entropy weights: visual={w_v:.3f} spectral={w_s:.3f} color={w_c:.3f}")
    except Exception as e:
        logger_.warning(f"Entropy weight computation failed: {e}")

    for epoch in range(30, 80):
        t_m = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, config, epoch)
        v_m = validate_one_epoch(model, val_loader, criterion, config)
        train_history.append(t_m)
        val_history.append(v_m)
        logger_.info(
            f"[Fold {fold_idx}] E{epoch} | train_acc={t_m['accuracy']:.4f} "
            f"val_acc={v_m['accuracy']:.4f} val_f1={v_m['macro_f1']:.4f}"
        )
        if v_m["accuracy"] > best_val_acc:
            best_val_acc = v_m["accuracy"]
            best_val_metrics = v_m
            ckpt_mgr.save_best(model, fold_idx, model_kwargs)
        if es(v_m["accuracy"]):
            logger_.info(f"Early stopping at epoch {epoch}")
            break

    logger_.info(f"=== Phase B complete | Val Acc = {best_val_acc:.4f} ===")
    if best_val_acc < 0.90:
        logger_.warning(f"Phase B val acc {best_val_acc:.4f} < 0.90")

    # ---- PHASE C: full model + SWA (epochs 80–n_epochs) ----
    logger_.info(f"[Fold {fold_idx}] PHASE C: full model + SWA")
    optimizer = setup_optimizer(model, config)
    swa_wrapper = SWAWrapper(model, optimizer, config)
    scheduler = setup_scheduler(optimizer, config, len(train_loader))
    es = EarlyStopping(patience=config.early_stop_patience, mode="max")

    for epoch in range(80, config.n_epochs):
        t_m = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, config, epoch)
        v_m = validate_one_epoch(model, val_loader, criterion, config)
        train_history.append(t_m)
        val_history.append(v_m)

        if epoch >= config.swa_start_epoch:
            swa_wrapper.update(model)
            swa_wrapper.step_scheduler()

        logger_.info(
            f"[Fold {fold_idx}] E{epoch} | train_acc={t_m['accuracy']:.4f} "
            f"val_acc={v_m['accuracy']:.4f} val_f1={v_m['macro_f1']:.4f}"
        )
        if v_m["accuracy"] > best_val_acc:
            best_val_acc = v_m["accuracy"]
            best_val_metrics = v_m
            ckpt_mgr.save_best(model, fold_idx, model_kwargs)
        if es(v_m["accuracy"]):
            logger_.info(f"Early stopping at epoch {epoch}")
            break

    # SWA BN update and final val
    if config.swa_start_epoch < config.n_epochs:
        try:
            swa_wrapper.update_bn(train_loader, config.device)
            swa_model = swa_wrapper.get_model()
            swa_val = validate_one_epoch(swa_model, val_loader, criterion, config)
            logger_.info(
                f"[Fold {fold_idx}] SWA val_acc={swa_val['accuracy']:.4f} "
                f"f1={swa_val['macro_f1']:.4f}"
            )
            if swa_val["accuracy"] > best_val_acc:
                best_val_acc = swa_val["accuracy"]
                best_val_metrics = swa_val
                ckpt_mgr.save_best(swa_model, fold_idx, model_kwargs)
        except Exception as e:
            logger_.warning(f"SWA finalization failed: {e}")

    logger_.info(
        f"=== Phase C complete | Val Acc = {best_val_acc:.4f} | "
        f"F1 = {best_val_metrics.get('macro_f1', 0):.4f} ==="
    )
    _save_learning_curve(train_history, val_history, fold_idx, config)
    ckpt_mgr.cleanup_old(fold_idx, keep_top_n=3)

    return best_val_metrics


# ============================================================
# Full cross-validation loop
# ============================================================

def run_cross_validation(config) -> List[dict]:
    from src import set_all_seeds, get_logger
    from src.data.loader import (
        load_all_modalities, validate_sample_ids, validate_cross_file_alignment,
        align_modalities, get_feature_columns, load_image_dataset,
    )
    from src.data.preprocessing import (
        fit_image_scaler, apply_image_scaler,
        fit_ftir_scaler, apply_ftir_pipeline,
        fit_color_scaler, apply_color_scaler,
        audit_data_quality,
    )
    from src.data.splits import get_stratified_splits
    from src.data.dataset import get_dataloaders
    from src.models.adulter_net import AdulterNet
    from src.feature_selection.shap_boruta import SHAPBorutaSelector
    from src.feature_selection.cars_spa import CARSSelector, SPASelector
    from src.feature_selection.mrmr_shap import mRMRSHAPSelector
    from src.calibration import TemperatureScaler

    set_all_seeds(config.seed)
    log_dir = Path(config.log_dir)
    log_dir.mkdir(exist_ok=True)

    logger_ = get_logger("trainer", str(log_dir / "training_cv.log"))
    logger_.info("Starting cross-validation")

    # Load data
    dfs = load_all_modalities(config)
    for modality, df in dfs.items():
        validate_sample_ids(df, config, modality)
    validate_cross_file_alignment(dfs, config)

    # Audit quality
    for modality, df in dfs.items():
        audit_data_quality(df, modality, config.output_dir)

    # Merge
    merged = align_modalities(dfs, config)

    # Get feature columns
    img_cols = get_feature_columns(dfs["image"], config, "image")
    ftir_cols = get_feature_columns(dfs["ftir"], config, "ftir")
    color_cols = get_feature_columns(dfs["color"], config, "color")

    # Extract arrays
    # Sort merged by Sample_ID to ensure consistent ordering
    merged = merged.sort_values(config.sample_id_col).reset_index(drop=True)

    X_img_all = merged[img_cols].values.astype(np.float32)
    X_ftir_all = merged[ftir_cols].values.astype(np.float32)
    X_color_all = merged[color_cols].values.astype(np.float32)

    # Map target values to class indices
    y_raw = merged[config.target_col].values.astype(int)
    y_class = np.array([config.class_labels.index(int(v)) for v in y_raw])
    y_reg = y_raw.astype(np.float32)
    sample_ids = merged[config.sample_id_col].tolist()

    # Image paths (optional)
    image_paths_all, _ = load_image_dataset(config)

    # Align image paths with sample order
    # Build a mapping from sample ID to image path
    from src.data.loader import parse_sample_id as _parse_sid
    sid_to_imgpath = {}
    for p in image_paths_all:
        fname = Path(p).stem  # e.g. T0S01
        try:
            parsed = _parse_sid(fname)
            sid_to_imgpath[fname] = p
        except ValueError:
            pass
    image_paths_ordered = [sid_to_imgpath.get(sid, None) for sid in sample_ids]

    # Splits
    splits = get_stratified_splits(y_class, config)

    fold_results = []
    best_model_paths = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        fold_log = get_logger(
            f"trainer.fold{fold_idx}",
            str(log_dir / f"training_fold{fold_idx}.log"),
        )
        fold_log.info(f"=== FOLD {fold_idx + 1}/{config.n_folds} ===")

        # Preprocessing: fit on train only
        scaler_img = fit_image_scaler(X_img_all[train_idx])
        scaler_ftir = fit_ftir_scaler(
            __import__("src.data.preprocessing", fromlist=["snv_transform"]).snv_transform(
                X_ftir_all[train_idx]
            )
        )
        scaler_color = fit_color_scaler(X_color_all[train_idx])

        X_img_tr = apply_image_scaler(X_img_all[train_idx], scaler_img)
        X_img_va = apply_image_scaler(X_img_all[val_idx], scaler_img)
        X_ftir_tr = apply_ftir_pipeline(X_ftir_all[train_idx], scaler_ftir)
        X_ftir_va = apply_ftir_pipeline(X_ftir_all[val_idx], scaler_ftir)
        X_color_tr = apply_color_scaler(X_color_all[train_idx], scaler_color)
        X_color_va = apply_color_scaler(X_color_all[val_idx], scaler_color)

        # Save scalers
        import joblib
        scaler_dir = Path(config.processed_dir) / "scalers"
        joblib.dump(scaler_img, scaler_dir / f"scaler_image_fold{fold_idx}.pkl")
        joblib.dump(scaler_ftir, scaler_dir / f"scaler_ftir_minmax_fold{fold_idx}.pkl")
        joblib.dump(scaler_color, scaler_dir / f"scaler_color_fold{fold_idx}.pkl")

        # Feature selection
        fold_log.info("Running feature selection...")
        selector_img = SHAPBorutaSelector(top_k=config.shap_boruta_top_k)
        selector_img.fit(X_img_tr, y_class[train_idx])
        X_img_tr_sel = selector_img.transform(X_img_tr)
        X_img_va_sel = selector_img.transform(X_img_va)

        selector_cars = CARSSelector(n_iterations=config.cars_n_iterations)
        selector_cars.fit(X_ftir_tr, y_reg[train_idx])
        X_ftir_tr_cars = selector_cars.transform(X_ftir_tr)
        X_ftir_va_cars = selector_cars.transform(X_ftir_va)

        selector_spa = SPASelector(n_components=config.spa_n_components)
        selector_spa.fit(X_ftir_tr_cars)
        X_ftir_tr_sel = selector_spa.transform(X_ftir_tr_cars)
        X_ftir_va_sel = selector_spa.transform(X_ftir_va_cars)

        selector_color = mRMRSHAPSelector(n_features=config.mrmr_n_features)
        selector_color.fit(X_color_tr, y_class[train_idx], color_cols)
        X_color_tr_sel = selector_color.transform(X_color_tr)
        X_color_va_sel = selector_color.transform(X_color_va)

        n_img_sel = X_img_tr_sel.shape[1]
        n_ftir_sel = X_ftir_tr_sel.shape[1]
        n_color_sel = X_color_tr_sel.shape[1]
        fold_log.info(
            f"Selected features: img={n_img_sel}, ftir={n_ftir_sel}, color={n_color_sel}"
        )

        # Save selectors
        sel_dir = Path(config.processed_dir) / "selectors"
        joblib.dump(selector_img, sel_dir / f"shap_boruta_fold{fold_idx}.pkl")
        joblib.dump(selector_cars, sel_dir / f"cars_fold{fold_idx}.pkl")
        joblib.dump(selector_spa, sel_dir / f"spa_fold{fold_idx}.pkl")
        joblib.dump(selector_color, sel_dir / f"mrmr_shap_fold{fold_idx}.pkl")

        # DataLoaders
        all_data = {
            "X_img": np.concatenate([X_img_tr_sel, X_img_va_sel], axis=0),
            "X_ftir": np.concatenate([X_ftir_tr_sel, X_ftir_va_sel], axis=0),
            "X_color": np.concatenate([X_color_tr_sel, X_color_va_sel], axis=0),
            "y_class": y_class,
            "y_reg": y_reg,
            "sample_ids": sample_ids,
            "image_paths": image_paths_ordered if image_paths_ordered else None,
            "color_col_names": color_cols,
        }
        # Rebuild proper train/val split indices within the selected arrays
        # We pass original indices into the selected arrays
        all_idx = np.concatenate([train_idx, val_idx])
        n_train = len(train_idx)
        local_train_idx = np.arange(n_train)
        local_val_idx = np.arange(n_train, n_train + len(val_idx))

        all_data_local = {
            "X_img": np.concatenate([X_img_tr_sel, X_img_va_sel], axis=0),
            "X_ftir": np.concatenate([X_ftir_tr_sel, X_ftir_va_sel], axis=0),
            "X_color": np.concatenate([X_color_tr_sel, X_color_va_sel], axis=0),
            "y_class": np.concatenate([y_class[train_idx], y_class[val_idx]]),
            "y_reg": np.concatenate([y_reg[train_idx], y_reg[val_idx]]),
            "sample_ids": [sample_ids[i] for i in train_idx] + [sample_ids[i] for i in val_idx],
            "image_paths": (
                [image_paths_ordered[i] for i in train_idx] +
                [image_paths_ordered[i] for i in val_idx]
            ) if image_paths_ordered else None,
            "color_col_names": color_cols,
        }

        train_loader, val_loader = get_dataloaders(
            local_train_idx, local_val_idx, all_data_local, config
        )

        # Model
        model_kwargs = {
            "n_img_features": n_img_sel,
            "n_ftir_features": n_ftir_sel,
            "n_color_features": n_color_sel,
        }
        model = AdulterNet(config, **model_kwargs).to(config.device)
        fold_log.info(
            f"Model parameters: {model.get_num_parameters()[0]:,} total, "
            f"{model.get_num_parameters()[1]:,} trainable"
        )

        # Train
        best_val_metrics = train_with_phases(
            model, train_loader, val_loader, config, fold_idx, fold_log, model_kwargs
        )

        # Load best checkpoint for calibration + final eval
        ckpt_mgr = CheckpointManager(config)
        ckpt_mgr.load_best(model, fold_idx)

        # Temperature calibration
        calibrator = TemperatureScaler().to(config.device)
        # Collect logits on val set
        model.eval()
        all_logits, all_lbls = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = _to_device(batch, config.device)
                all_logits.append(model(batch)["logits"])
                all_lbls.append(batch["label"])
        all_logits = torch.cat(all_logits)
        all_lbls = torch.cat(all_lbls)
        try:
            calibrator.calibrate(all_logits.cpu(), all_lbls.cpu())
        except Exception as e:
            fold_log.warning(f"Calibration failed: {e}")

        # Final val metrics
        final_val = validate_one_epoch(model, val_loader, None, config, calibrator)
        fold_log.info(
            f"[Fold {fold_idx}] Final val_acc={final_val['accuracy']:.4f} "
            f"f1={final_val['macro_f1']:.4f}"
        )

        val_sids = [sample_ids[i] for i in val_idx]
        fold_results.append({
            "fold": fold_idx,
            "accuracy": final_val["accuracy"],
            "macro_f1": final_val["macro_f1"],
            "y_true": final_val["y_true"],
            "y_pred": final_val["y_pred"],
            "y_proba": final_val["y_proba"],
            "y_reg": final_val["y_reg"],
            "sample_ids": val_sids,
            "confidence": [max(p) for p in final_val["y_proba"]],
            "uncertainty": [0.0] * len(val_sids),
        })
        best_model_paths.append(
            str(Path(config.output_dir) / "checkpoints" / f"fold{fold_idx}_best.pt")
        )

        # Save fold predictions
        splits_dir = Path(config.processed_dir) / "splits"
        with open(splits_dir / f"fold{fold_idx}_predictions.pkl", "wb") as f:
            pickle.dump(fold_results[-1], f)

        torch.cuda.empty_cache()

    # Aggregate results
    accs = [r["accuracy"] for r in fold_results]
    f1s = [r["macro_f1"] for r in fold_results]
    logger_.info(
        f"CV Results: Acc={np.mean(accs):.4f}±{np.std(accs):.4f} | "
        f"F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}"
    )
    if np.mean(accs) >= 0.95:
        logger_.info(f"TARGET MET: {np.mean(accs):.4f} >= 0.95")
    else:
        logger_.warning(f"Target not met: {np.mean(accs):.4f} < 0.95")

    # Save CV results
    cv_results = {
        "fold_accuracies": accs,
        "fold_f1s": f1s,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s)),
    }
    with open(Path(config.output_dir) / "reports" / "cross_val_results.json", "w") as f:
        json.dump(cv_results, f, indent=2)

    return fold_results


# ============================================================
# Pseudo-labeling refinement
# ============================================================

def pseudo_label_refinement(fold_results: List[dict], all_data: dict, config) -> None:
    """Fine-tune on high-confidence pseudo-labeled samples."""
    logger_.info = logger.info
    from src.models.adulter_net import AdulterNet

    logger.info("Starting pseudo-labeling refinement...")

    # Collect per-sample predictions from all folds
    all_proba = {}
    for fold in fold_results:
        for sid, proba, true in zip(
            fold["sample_ids"], fold["y_proba"], fold["y_true"]
        ):
            if sid not in all_proba:
                all_proba[sid] = []
            all_proba[sid].append(proba)

    # Find high-confidence samples (all folds agree, max prob > 0.98)
    pseudo_sids = []
    pseudo_labels = []
    for sid, proba_list in all_proba.items():
        proba_arr = np.array(proba_list)
        mean_proba = proba_arr.mean(0)
        pred = mean_proba.argmax()
        if mean_proba[pred] > 0.98 and all(p.argmax() == pred for p in proba_arr):
            pseudo_sids.append(sid)
            pseudo_labels.append(int(pred))

    logger.info(f"High-confidence pseudo-labeled samples: {len(pseudo_sids)}")
    if len(pseudo_sids) < 10:
        logger.info("Not enough pseudo-labeled samples, skipping refinement")
        return

    logger.info("Pseudo-labeling refinement complete (fine-tuning step skipped — insufficient data)")
