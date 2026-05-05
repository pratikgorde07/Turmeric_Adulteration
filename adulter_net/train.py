from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src import set_all_seeds, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="AdulterNet Training — >=0.95 classification accuracy target")
    parser.add_argument("--data_dir", default=None, help="Override data/raw directory")
    parser.add_argument("--output_dir", default=None, help="Override outputs directory")
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--n_folds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--no_raw_images", action="store_true")
    args = parser.parse_args()

    set_all_seeds(args.seed)

    from config import Config
    cfg_kwargs = {}
    if args.data_dir:
        cfg_kwargs["data_dir"] = Path(args.data_dir)
    if args.output_dir:
        cfg_kwargs["output_dir"] = Path(args.output_dir)

    config = Config(**cfg_kwargs)
    if args.n_epochs is not None:
        config.n_epochs = args.n_epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.n_folds is not None:
        config.n_folds = args.n_folds
    if args.use_wandb:
        config.use_wandb = True
    if args.no_raw_images:
        config.use_raw_images = False

    logger.info(f"AdulterNet training started | device={config.device} | epochs={config.n_epochs} | folds={config.n_folds}")

    from src.trainer import run_cross_validation
    summary = run_cross_validation(config)

    logger.info("=" * 60)
    logger.info(f"Training complete!")
    logger.info(f"CV Accuracy : {summary['cv_accuracy_mean']:.4f} ± {summary['cv_accuracy_std']:.4f}")
    logger.info(f"CV Macro F1 : {summary['cv_macro_f1_mean']:.4f} ± {summary['cv_macro_f1_std']:.4f}")
    logger.info(f"Target {'ACHIEVED ✓' if summary['target_achieved'] else 'NOT MET ✗'} (>= 0.95)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
