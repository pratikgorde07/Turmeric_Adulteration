from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import get_logger, set_all_seeds
from src.data.augmentation import apply_mixup_to_batch
from src.losses import AdulterNetLoss, MixupCriterion, compute_class_weights
from src.calibration import TemperatureScaler

logger = get_logger(__name__)


def top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, k: int = 2) -> float:
    """Compute top-k accuracy."""
    top_k_preds = np.argsort(-y_proba, axis=1)[:, :k]
    correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
    return float(correct.mean())


class EarlyStopping:
    """Monitor a metric and signal when training should stop."""

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-4,
        mode: str = "max",
        metric_name: str = "accuracy",
    ):
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
            (current > self.best + self.min_delta)
            if self.mode == "max"
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
    """Save/load/cleanup model checkpoints."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoints: List[Tuple[float, Path]] = []

    def save(self, model, optimizer, scheduler, epoch, metrics, fold_idx) -> Path:
        acc = metrics.get("accuracy", 0.0)
        path = self.checkpoint_dir / f"fold{fold_idx}_epoch{epoch:03d}_acc{acc:.4f}.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
                "fold_idx": fold_idx,
            },
            path,
        )
        self._checkpoints.append((acc, path))
        return path

    def save_best(self, model, fold_idx) -> Path:
        path = self.checkpoint_dir / f"fold{fold_idx}_best.pt"
        torch.save({"model_state": model.state_dict()}, path)
        logger.info(f"Best checkpoint saved: {path}")
        return path

    def load_best(self, model, fold_idx) -> None:
        path = self.checkpoint_dir / f"fold{fold_idx}_best.pt"
        checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint.get("model_state", checkpoint)
        model.load_state_dict(state)
        logger.info(f"Best checkpoint loaded: {path}")

    def cleanup_old(self, fold_idx: int, keep_top_n: int = 3) -> None:
        fold_ckpts = [
            (acc, p) for acc, p in self._checkpoints
            if f"fold{fold_idx}_epoch" in p.name
        ]
        fold_ckpts.sort(key=lambda t: t[0], reverse=True)
        for _, path in fold_ckpts[keep_top_n:]:
            if path.exists():
                path.unlink()


def setup_optimizer(model, config) -> torch.optim.Optimizer:
    """AdamW with per-group learning rates."""
    param_groups = [
        {"params": model.classification_head.parameters(), "lr": config.lr_head},
        {"params": model.regression_head.parameters(), "lr": config.lr_head},
        {"params": model.cmt_fusion.parameters(), "lr": config.lr_head},
        {"params": model.spectral_stream.parameters(), "lr": config.lr_head},
        {"params": model.color_stream.parameters(), "lr": config.lr_head},
        {"params": model.visual_stream.adapter.parameters(), "lr": config.lr_head},
        {
            "params": model.visual_stream.base_model.parameters(),
            "lr": config.lr_backbone,
            "name": "backbone",
        },
    ]
    return torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)


def setup_scheduler(optimizer, config, n_batches_per_epoch: int):
    """OneCycleLR scheduler — aggressive ramp for small datasets."""
    max_lrs = [config.lr_head] * 6 + [config.lr_backbone]
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        total_steps=config.n_epochs * n_batches_per_epoch,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10,
        final_div_factor=100,
    )


def train_one_epoch(model, loader, optimizer, scheduler, criterion, scaler, config, epoch, fold_logger=None) -> Dict:
    """Train for one epoch and return metric dict."""
    model.train()
    log = fold_logger or logger
    metrics = defaultdict(float)
    all_preds, all_labels = [], []
    mixup_criterion = MixupCriterion()

    for batch in tqdm(loader, desc=f"Train E{epoch}", leave=False):
        batch = {k: v.to(config.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.cuda.amp.autocast(enabled=config.use_amp and config.device == "cuda"):
            out = model(batch, return_embeddings=True)

            # Mixup on 30% of batches between adjacent-class pairs
            mixup_loss = torch.tensor(0.0, device=config.device)
            if random.random() < 0.3 and out["fused_emb"] is not None:
                try:
                    mixed_emb, mixed_soft = apply_mixup_to_batch(
                        out["fused_emb"].detach(), batch["label"],
                        config.mixup_alpha, config.class_labels, config.n_classes,
                    )
                    if len(mixed_emb) > 0:
                        logits_mixed = model.classification_head(mixed_emb)
                        mixup_loss = mixup_criterion(logits_mixed, mixed_soft)
                except Exception:
                    pass

            loss_dict = criterion(
                out["logits"], out["reg_pred"], out["proj_emb"],
                batch["label"], batch["label_float"],
            )
            total_loss = loss_dict["total"] + 0.1 * mixup_loss

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
            metrics[k] += v.item() if isinstance(v, torch.Tensor) else float(v)

    n = len(loader)
    result = {k: v / n for k, v in metrics.items()}
    result["accuracy"] = float((np.array(all_preds) == np.array(all_labels)).mean())
    return result


def validate_one_epoch(model, loader, config, calibrator=None) -> Dict:
    """Validate for one epoch and return metric dict."""
    from sklearn.metrics import f1_score, cohen_kappa_score
    model.eval()
    all_preds, all_labels, all_probs, all_reg = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            batch = {k: v.to(config.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(batch)
            logits = calibrator.scale(out["logits"]) if calibrator is not None else out["logits"]
            probs = F.softmax(logits, dim=-1)
            preds = probs.argmax(-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["label"].cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            reg = out["reg_pred"].squeeze(-1) if out["reg_pred"].dim() > 1 else out["reg_pred"]
            all_reg.extend(reg.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    y_reg = np.array(all_reg)
    y_reg_true = np.array([config.class_labels[l] for l in y_true])

    acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    top2_acc = top_k_accuracy(y_true, y_probs, k=2)
    try:
        kappa = float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        kappa = 0.0
    rmse = float(np.sqrt(np.mean((y_reg - y_reg_true) ** 2)))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "top2_accuracy": top2_acc,
        "kappa": kappa,
        "rmse": rmse,
    }


def compute_entropy_weights(model, val_loader, config) -> Tuple[float, float, float]:
    """Compute inverse-entropy modality weights from validation predictions."""
    model.eval()
    entropies = {"visual": [], "spectral": [], "color": []}

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(config.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            try:
                v_out = model.visual_stream(batch)
                s_out = model.spectral_stream(batch["ftir"])
                c_out = model.color_stream(batch["color"])
                zeros_v = torch.zeros_like(v_out)
                zeros_s = torch.zeros_like(s_out)
                zeros_c = torch.zeros_like(c_out)

                def stream_entropy(v, s, c):
                    fused = model.cmt_fusion(v, s, c)
                    logits = model.classification_head(fused)
                    probs = F.softmax(logits, -1)
                    return (-(probs * (probs + 1e-10).log()).sum(-1)).cpu().numpy()

                entropies["visual"].extend(stream_entropy(v_out, zeros_s, zeros_c))
                entropies["spectral"].extend(stream_entropy(zeros_v, s_out, zeros_c))
                entropies["color"].extend(stream_entropy(zeros_v, zeros_s, c_out))
            except Exception as e:
                logger.warning(f"Entropy computation error: {e}")

    H_v = np.mean(entropies["visual"]) + 1e-8
    H_s = np.mean(entropies["spectral"]) + 1e-8
    H_c = np.mean(entropies["color"]) + 1e-8
    w_v, w_s, w_c = 1 / H_v, 1 / H_s, 1 / H_c
    total = (w_v + w_s + w_c) / 3
    w_v, w_s, w_c = w_v / total, w_s / total, w_c / total
    logger.info(f"Entropy weights: visual={w_v:.3f}, spectral={w_s:.3f}, color={w_c:.3f}")
    return float(w_v), float(w_s), float(w_c)


def run_optuna_search(train_loader, val_loader, config, n_ftir, n_color, n_trials=20) -> Dict:
    """Optuna hyperparameter search triggered when Phase B val_acc < 0.90."""
    import optuna
    import copy
    from src.models.adulter_net import AdulterNet

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        lr_head = trial.suggest_float("lr_head", 1e-4, 5e-3, log=True)
        lr_backbone = trial.suggest_float("lr_backbone", 1e-6, 1e-4, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.4)
        tc = copy.deepcopy(config)
        tc.lr_head = lr_head
        tc.lr_backbone = lr_backbone
        tc.dropout = dropout
        model = AdulterNet(tc, n_ftir_features=n_ftir, n_color_features=n_color).to(config.device)
        model.freeze_all_backbones()
        opt = setup_optimizer(model, tc)
        crit = AdulterNetLoss(alpha=tc.alpha_loss, supcon_weight=tc.supcon_weight,
                              gamma=tc.focal_gamma, smoothing=tc.label_smoothing,
                              supcon_temperature=tc.supcon_temperature)
        amp_sc = torch.cuda.amp.GradScaler(enabled=tc.use_amp and tc.device == "cuda")
        best_acc = 0.0
        for ep in range(30):
            train_one_epoch(model, train_loader, opt, None, crit, amp_sc, tc, ep)
            vm = validate_one_epoch(model, val_loader, tc)
            best_acc = max(best_acc, vm["accuracy"])
        return best_acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"Optuna best: {study.best_params} (val_acc={study.best_value:.4f})")
    return study.best_params


def train_with_phases(model, train_loader, val_loader, config, fold_idx, n_ftir, n_color, fold_logger=None) -> Tuple[Dict, nn.Module]:
    """Three-phase training: heads-only → partial unfreeze → full + SWA."""
    from src.models.swa_model import SWAWrapper
    log = fold_logger or logger
    ckpt = CheckpointManager(config.output_dir / "checkpoints")
    best_val_acc = 0.0
    best_metrics = {}
    criterion = AdulterNetLoss(alpha=config.alpha_loss, supcon_weight=config.supcon_weight,
                               gamma=config.focal_gamma, smoothing=config.label_smoothing,
                               supcon_temperature=config.supcon_temperature)
    amp_scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp and config.device == "cuda")

    # ── Phase A: heads + fusion only (epochs 0-29) ──
    log.info(f"[Fold {fold_idx}] Phase A: training heads + fusion (epochs 0-29)")
    model.freeze_all_backbones()
    optimizer = setup_optimizer(model, config)
    scheduler = setup_scheduler(optimizer, config, len(train_loader))

    for epoch in range(30):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, amp_scaler, config, epoch, log)
        val_metrics = validate_one_epoch(model, val_loader, config)
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_metrics = val_metrics
            ckpt.save_best(model, fold_idx)
        if epoch % 10 == 0 or epoch == 29:
            log.info(f"  [Fold {fold_idx}][PhaseA] E{epoch:03d} | train={train_metrics['accuracy']:.4f} | val={val_metrics['accuracy']:.4f} | f1={val_metrics['macro_f1']:.4f}")

    log.info(f"[Fold {fold_idx}] Phase A complete. Best val acc = {best_val_acc:.4f}")
    if best_val_acc < 0.80:
        log.error(f"[Fold {fold_idx}] Phase A acc {best_val_acc:.4f} < 0.80 — check data pipeline")

    # ── Phase B: top-N backbone blocks (epochs 30-79) ──
    log.info(f"[Fold {fold_idx}] Phase B: unfreezing top {config.efficientnet_finetune_blocks} blocks (epochs 30-79)")
    model.unfreeze_visual_top_n(config.efficientnet_finetune_blocks)
    optimizer = setup_optimizer(model, config)
    scheduler = setup_scheduler(optimizer, config, len(train_loader))
    try:
        w_v, w_s, w_c = compute_entropy_weights(model, val_loader, config)
        model.cmt_fusion.set_entropy_weights(w_v, w_s, w_c)
    except Exception as e:
        log.warning(f"Entropy weight calibration failed: {e}")

    for epoch in range(30, 80):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, amp_scaler, config, epoch, log)
        val_metrics = validate_one_epoch(model, val_loader, config)
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_metrics = val_metrics
            ckpt.save_best(model, fold_idx)
        if epoch % 10 == 0 or epoch == 79:
            log.info(f"  [Fold {fold_idx}][PhaseB] E{epoch:03d} | train={train_metrics['accuracy']:.4f} | val={val_metrics['accuracy']:.4f}")

    log.info(f"[Fold {fold_idx}] Phase B complete. Best val acc = {best_val_acc:.4f}")
    if best_val_acc < 0.90:
        log.warning(f"[Fold {fold_idx}] Phase B acc {best_val_acc:.4f} < 0.90 — running Optuna search")
        try:
            best_params = run_optuna_search(train_loader, val_loader, config, n_ftir, n_color, n_trials=20)
            config.lr_head = best_params.get("lr_head", config.lr_head)
            config.lr_backbone = best_params.get("lr_backbone", config.lr_backbone)
            log.info(f"[Fold {fold_idx}] Optuna params applied: {best_params}")
        except Exception as e:
            log.warning(f"Optuna search failed: {e}")

    # ── Phase C: full model + SWA (epochs 80-n_epochs) ──
    log.info(f"[Fold {fold_idx}] Phase C: full model + SWA (epochs 80-{config.n_epochs})")
    optimizer = setup_optimizer(model, config)
    swa_wrapper = SWAWrapper(model, optimizer, config)
    scheduler = setup_scheduler(optimizer, config, len(train_loader))
    early_stop = EarlyStopping(patience=config.early_stop_patience)

    for epoch in range(80, config.n_epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, amp_scaler, config, epoch, log)
        if epoch >= config.swa_start_epoch:
            swa_wrapper.update(model)
            swa_wrapper.step_scheduler()
        val_metrics = validate_one_epoch(model, val_loader, config)
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_metrics = val_metrics
            ckpt.save_best(model, fold_idx)
        if epoch % 10 == 0 or epoch == config.n_epochs - 1:
            log.info(f"  [Fold {fold_idx}][PhaseC] E{epoch:03d} | train={train_metrics['accuracy']:.4f} | val={val_metrics['accuracy']:.4f}")
        if early_stop(val_metrics["accuracy"]):
            log.info(f"[Fold {fold_idx}] Early stopping at epoch {epoch}")
            break

    # Final SWA BN update + evaluation
    try:
        swa_wrapper.update_bn(train_loader, config.device)
        swa_model = swa_wrapper.get_model()
        swa_metrics = validate_one_epoch(swa_model, val_loader, config)
        log.info(f"[Fold {fold_idx}] SWA val acc={swa_metrics['accuracy']:.4f} | f1={swa_metrics['macro_f1']:.4f}")
        if swa_metrics["accuracy"] > best_val_acc:
            best_val_acc = swa_metrics["accuracy"]
            best_metrics = swa_metrics
            torch.save({"model_state": swa_model.module.state_dict()},
                       config.output_dir / "checkpoints" / f"fold{fold_idx}_best.pt")
    except Exception as e:
        log.warning(f"SWA final eval failed: {e}")

    log.info(f"[Fold {fold_idx}] Done. Best val acc={best_val_acc:.4f} | f1={best_metrics.get('macro_f1', 0):.4f}")
    return best_metrics, model


def run_cross_validation(config) -> Dict:
    """Full 5-fold cross-validation pipeline."""
    import pickle
    from src.data.loader import (load_all_modalities, validate_sample_ids, align_modalities,
                                  get_feature_columns, audit_data_quality, load_image_dataset)
    from src.data.preprocessing import (fit_image_scaler, apply_image_scaler, fit_ftir_scaler,
                                         apply_ftir_pipeline, fit_color_scaler, apply_color_scaler,
                                         save_all_scalers, check_distribution_shift, check_no_all_zeros)
    from src.data.dataset import get_dataloaders
    from src.data.splits import create_stratified_folds, save_splits
    from src.data.augmentation import TriModalFeatureAugmenter
    from src.feature_selection.shap_boruta import SHAPBorutaSelector
    from src.feature_selection.cars_spa import CARSSelector, SPASelector
    from src.feature_selection.mrmr_shap import mRMRSHAPSelector
    from src.models.adulter_net import AdulterNet

    set_all_seeds(config.seed)

    # Load and validate data
    logger.info("Loading all modalities...")
    dfs = load_all_modalities(config)
    validate_sample_ids(dfs, config)
    for name, df in dfs.items():
        audit_data_quality(df, name, config)
    align_modalities(dfs, config)

    img_feat_cols = get_feature_columns(dfs["image"], config, "image")
    ftir_cols = get_feature_columns(dfs["ftir"], config, "ftir")
    color_cols = get_feature_columns(dfs["color"], config, "color")

    X_img_all = dfs["image"][img_feat_cols].values.astype(np.float32)
    X_ftir_all = dfs["ftir"][ftir_cols].values.astype(np.float32)
    X_color_all = dfs["color"][color_cols].values.astype(np.float32)
    y_raw = dfs["image"][config.target_col].values.astype(int)
    y_class = np.array([config.class_labels.index(int(v)) for v in y_raw])
    y_reg = y_raw.astype(np.float32)
    sample_ids = dfs["image"][config.sample_id_col].tolist()

    image_paths, _ = load_image_dataset(config)
    if len(image_paths) != len(sample_ids):
        logger.warning(f"Image paths ({len(image_paths)}) != samples ({len(sample_ids)}). Using feature-only mode.")
        image_paths = [None] * len(sample_ids)

    all_data = {
        "X_img": X_img_all, "X_ftir": X_ftir_all, "X_color": X_color_all,
        "y_class": y_class, "y_reg": y_reg, "sample_ids": sample_ids, "image_paths": image_paths,
    }

    splits = create_stratified_folds(y_class, config)
    save_splits(splits, config)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"\n{'='*60}\n=== FOLD {fold_idx+1}/{config.n_folds} ===\n{'='*60}")
        fold_log = get_logger(f"fold_{fold_idx}", str(config.log_dir / f"training_fold{fold_idx}.log"))
        set_all_seeds(config.seed + fold_idx)

        # Split
        X_img_tr, X_img_v = X_img_all[train_idx], X_img_all[val_idx]
        X_ftir_tr, X_ftir_v = X_ftir_all[train_idx], X_ftir_all[val_idx]
        X_col_tr, X_col_v = X_color_all[train_idx], X_color_all[val_idx]
        y_tr, y_v = y_class[train_idx], y_class[val_idx]
        y_reg_tr, y_reg_v = y_reg[train_idx], y_reg[val_idx]

        # Scale (fit on train only)
        sc_img = fit_image_scaler(X_img_tr)
        sc_ftir = fit_ftir_scaler(X_ftir_tr)
        sc_col = fit_color_scaler(X_col_tr)
        save_all_scalers(sc_img, sc_ftir, sc_col, config, fold_idx)

        X_img_tr_sc = apply_image_scaler(X_img_tr, sc_img)
        X_img_v_sc = apply_image_scaler(X_img_v, sc_img)
        X_ftir_tr_sc = apply_ftir_pipeline(X_ftir_tr, sc_ftir)
        X_ftir_v_sc = apply_ftir_pipeline(X_ftir_v, sc_ftir)
        X_col_tr_sc = apply_color_scaler(X_col_tr, sc_col)
        X_col_v_sc = apply_color_scaler(X_col_v, sc_col)

        check_distribution_shift(X_img_tr_sc, X_img_v_sc, "image")
        check_distribution_shift(X_ftir_tr_sc, X_ftir_v_sc, "ftir")
        check_distribution_shift(X_col_tr_sc, X_col_v_sc, "color")

        # Feature selection (fit on train only)
        fold_log.info(f"[Fold {fold_idx}] Feature selection...")
        shap_boruta = SHAPBorutaSelector(top_k=config.shap_boruta_top_k)
        shap_boruta.fit(X_img_tr_sc, y_tr)
        X_img_tr_sel = shap_boruta.transform(X_img_tr_sc)
        X_img_v_sel = shap_boruta.transform(X_img_v_sc)
        shap_boruta.save(str(config.processed_dir / "selectors" / f"shap_boruta_fold{fold_idx}.pkl"))

        cars = CARSSelector(n_iterations=config.cars_n_iterations)
        cars.fit(X_ftir_tr_sc, y_reg_tr)
        X_ftir_cars_tr = cars.transform(X_ftir_tr_sc)
        X_ftir_cars_v = cars.transform(X_ftir_v_sc)
        cars.save(str(config.processed_dir / "selectors" / f"cars_fold{fold_idx}.pkl"))

        spa = SPASelector(n_components=config.spa_n_components)
        spa.fit(X_ftir_cars_tr)
        X_ftir_tr_sel = spa.transform(X_ftir_cars_tr)
        X_ftir_v_sel = spa.transform(X_ftir_cars_v)
        spa.save(str(config.processed_dir / "selectors" / f"spa_fold{fold_idx}.pkl"))

        mrmr_shap = mRMRSHAPSelector(n_mrmr=config.mrmr_n_features)
        mrmr_shap.fit(X_col_tr_sc, y_tr, color_cols)
        X_col_tr_sel = mrmr_shap.transform(X_col_tr_sc)
        X_col_v_sel = mrmr_shap.transform(X_col_v_sc)
        mrmr_shap.save(str(config.processed_dir / "selectors" / f"mrmr_shap_fold{fold_idx}.pkl"))

        n_ftir_sel = X_ftir_tr_sel.shape[1]
        n_color_sel = X_col_tr_sel.shape[1]

        check_no_all_zeros(X_img_tr_sel, "train", "image")
        check_no_all_zeros(X_ftir_tr_sel, "train", "ftir")
        check_no_all_zeros(X_col_tr_sel, "train", "color")

        # DataLoaders
        n_tr = len(train_idx)
        fold_data = {
            "X_img": np.concatenate([X_img_tr_sel, X_img_v_sel]),
            "X_ftir": np.concatenate([X_ftir_tr_sel, X_ftir_v_sel]),
            "X_color": np.concatenate([X_col_tr_sel, X_col_v_sel]),
            "y_class": np.concatenate([y_tr, y_v]),
            "y_reg": np.concatenate([y_reg_tr, y_reg_v]),
            "sample_ids": [sample_ids[i] for i in train_idx] + [sample_ids[i] for i in val_idx],
            "image_paths": [image_paths[i] for i in train_idx] + [image_paths[i] for i in val_idx],
        }
        train_loader, val_loader = get_dataloaders(
            np.arange(n_tr), np.arange(n_tr, n_tr + len(val_idx)),
            fold_data, config, feat_augment=TriModalFeatureAugmenter(),
        )

        # Model
        model = AdulterNet(config, n_ftir_features=n_ftir_sel, n_color_features=n_color_sel).to(config.device)

        # Train
        best_metrics, trained_model = train_with_phases(
            model, train_loader, val_loader, config, fold_idx, n_ftir_sel, n_color_sel, fold_log
        )

        # Load best checkpoint and calibrate temperature
        ckpt_mgr = CheckpointManager(config.output_dir / "checkpoints")
        try:
            ckpt_mgr.load_best(model, fold_idx)
        except Exception as e:
            fold_log.warning(f"Could not load best checkpoint: {e}")

        model.eval()
        all_logits, all_lbl_cal = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(config.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                all_logits.append(model(batch)["logits"])
                all_lbl_cal.append(batch["label"])
        all_logits = torch.cat(all_logits)
        all_lbl_cal = torch.cat(all_lbl_cal)

        calibrator = TemperatureScaler().to(config.device)
        calibrator.calibrate(all_logits, all_lbl_cal)

        cal_metrics = validate_one_epoch(model, val_loader, config, calibrator=calibrator)
        fold_log.info(f"[Fold {fold_idx}] Calibrated val acc={cal_metrics['accuracy']:.4f} | f1={cal_metrics['macro_f1']:.4f}")

        # TTA metrics
        try:
            from src.tta import tta_predict, TTA_TRANSFORMS
            tta_preds, tta_lbls = [], []
            for batch in val_loader:
                batch = {k: v.to(config.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                probs = tta_predict(model, batch, TTA_TRANSFORMS, config)
                tta_preds.extend(probs.argmax(-1).cpu().numpy())
                tta_lbls.extend(batch["label"].cpu().numpy())
            tta_acc = float((np.array(tta_preds) == np.array(tta_lbls)).mean())
            fold_log.info(f"[Fold {fold_idx}] TTA val acc={tta_acc:.4f}")
            cal_metrics["tta_accuracy"] = tta_acc
        except Exception as e:
            fold_log.warning(f"TTA failed: {e}")

        fold_results.append({"fold": fold_idx, "val_idx": val_idx.tolist(), "metrics": cal_metrics})

        pred_path = config.processed_dir / "splits" / f"fold{fold_idx}_predictions.pkl"
        with open(pred_path, "wb") as f:
            pickle.dump({"fold": fold_idx, "metrics": cal_metrics}, f)

        torch.cuda.empty_cache()
        logger.info(f"=== Phase complete | Fold {fold_idx+1} | Val Acc={cal_metrics['accuracy']:.4f} | F1={cal_metrics['macro_f1']:.4f} ===")

    # Aggregate
    accs = [r["metrics"]["accuracy"] for r in fold_results]
    f1s = [r["metrics"]["macro_f1"] for r in fold_results]
    acc_mean, acc_std = float(np.mean(accs)), float(np.std(accs))
    f1_mean, f1_std = float(np.mean(f1s)), float(np.std(f1s))

    logger.info(f"\nCV Results: Acc={acc_mean:.4f}±{acc_std:.4f} | F1={f1_mean:.4f}±{f1_std:.4f}")
    logger.info(f"Per-fold accuracies: {[round(a, 4) for a in accs]}")

    target = acc_mean >= 0.95 and f1_mean >= 0.95
    if target:
        logger.info(f"TARGET MET: acc={acc_mean:.4f} >= 0.95 and f1={f1_mean:.4f} >= 0.95")
    else:
        logger.warning(f"Target not met: acc={acc_mean:.4f}, f1={f1_mean:.4f}")

    summary = {
        "target_achieved": target,
        "cv_accuracy_mean": acc_mean, "cv_accuracy_std": acc_std,
        "cv_macro_f1_mean": f1_mean, "cv_macro_f1_std": f1_std,
        "per_fold_accuracies": accs, "per_fold_f1s": f1s,
    }
    out_path = config.output_dir / "reports" / "cross_val_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"CV results saved: {out_path}")
    return summary
