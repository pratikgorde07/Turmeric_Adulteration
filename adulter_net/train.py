"""Main training entry point."""
from __future__ import annotations
import argparse
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(description="AdulterNet Training")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override data directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--n_folds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--skip_baselines", action="store_true")
    args = parser.parse_args()

    from src import set_all_seeds
    set_all_seeds(args.seed)

    from config import Config
    config = Config()

    if args.data_dir:
        config.data_dir = Path(args.data_dir)
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    if args.n_epochs:
        config.n_epochs = args.n_epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.n_folds:
        config.n_folds = args.n_folds
    config.seed = args.seed
    config.use_wandb = args.use_wandb

    from src import get_logger
    logger = get_logger("train", str(config.log_dir / "train.log"))
    logger.info(f"Starting AdulterNet training | device={config.device}")
    logger.info(f"Config: epochs={config.n_epochs}, batch={config.batch_size}, folds={config.n_folds}")

    # Cross-validation
    from src.trainer import run_cross_validation, pseudo_label_refinement
    fold_results = run_cross_validation(config)

    # Aggregate OOF predictions
    from src.evaluator import (
        aggregate_oof_predictions, compute_full_metrics,
        save_classification_report, save_confusion_matrix,
        save_roc_curves, save_regression_scatter,
        save_final_summary,
    )
    import numpy as np

    oof = aggregate_oof_predictions(fold_results, config)
    metrics = compute_full_metrics(oof, config)

    fold_accs = [r["accuracy"] for r in fold_results]
    save_classification_report(metrics, fold_accs, config)
    save_confusion_matrix(oof["y_true"], oof["y_pred"], config)
    save_roc_curves(oof["y_true"], oof["y_proba"], config)
    save_regression_scatter(oof["y_true"], oof["y_reg"], config)

    # Pseudo-labeling (optional refinement)
    try:
        pseudo_label_refinement(fold_results, {}, config)
    except Exception as e:
        logger.warning(f"Pseudo-labeling failed: {e}")

    # Final summary
    summary = save_final_summary(metrics, fold_accs, config)
    logger.info(f"Final summary: {json.dumps(summary, indent=2)}")

    # Baselines (optional)
    if not args.skip_baselines:
        try:
            from baselines.train_baselines import run_baselines
            run_baselines(config)
        except Exception as e:
            logger.warning(f"Baseline training failed: {e}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
