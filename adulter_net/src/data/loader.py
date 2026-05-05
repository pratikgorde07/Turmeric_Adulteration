"""CSV loading, Sample_ID parsing and validation."""
from __future__ import annotations
import re
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def load_all_modalities(config) -> Dict[str, pd.DataFrame]:
    """Load all three modality CSVs and return as dict."""
    data_dir = Path(config.data_dir)
    files = {
        "image": data_dir / config.image_feat_file,
        "ftir": data_dir / config.ftir_file,
        "color": data_dir / config.color_file,
    }
    dfs = {}
    for modality, fpath in files.items():
        if not fpath.exists():
            raise FileNotFoundError(f"CSV not found: {fpath}")
        df = pd.read_csv(fpath)
        assert config.sample_id_col in df.columns, f"{modality}: missing {config.sample_id_col}"
        assert config.target_col in df.columns, f"{modality}: missing {config.target_col}"
        nan_counts = df.isnull().sum().sum()
        logger.info(
            f"[{modality}] shape={df.shape}, dtypes={df.dtypes.value_counts().to_dict()}, "
            f"NaN_total={nan_counts}"
        )
        dfs[modality] = df
    return dfs


def parse_sample_id(sample_id: str) -> Dict[str, int]:
    """Parse T{level}S{nn} format. Raises ValueError on mismatch."""
    pattern = r"^T(\d+)S(\d{2})$"
    m = re.match(pattern, str(sample_id).strip())
    if not m:
        raise ValueError(f"Invalid Sample_ID format: '{sample_id}' (expected T{{level}}S{{nn}})")
    return {"adulteration_level": int(m.group(1)), "sample_number": int(m.group(2))}


def validate_sample_ids(df: pd.DataFrame, config, modality_name: str = "unknown") -> None:
    """Validate all Sample_IDs and cross-check against Target column."""
    errors = []
    for idx, row in df.iterrows():
        sid = row[config.sample_id_col]
        target = row[config.target_col]
        try:
            parsed = parse_sample_id(sid)
            level = parsed["adulteration_level"]
            num = parsed["sample_number"]
            if level not in config.class_labels:
                errors.append(f"Row {idx}: level {level} not in class_labels {config.class_labels}")
            if int(target) != level:
                errors.append(
                    f"Row {idx}: Sample_ID {sid} has level {level} but Target={target}"
                )
            if num < 1 or num > 50:
                errors.append(f"Row {idx}: sample number {num} out of range [01,50]")
        except ValueError as e:
            errors.append(str(e))

    # Check sample numbers per class
    for lbl in config.class_labels:
        rows = df[df[config.target_col] == lbl]
        nums = set()
        for sid in rows[config.sample_id_col]:
            try:
                p = parse_sample_id(sid)
                nums.add(p["sample_number"])
            except ValueError:
                pass
        expected = set(range(1, 51))
        missing = expected - nums
        if missing:
            logger.warning(f"[{modality_name}] level={lbl}: missing sample nums {missing}")

    if errors:
        msg = f"[{modality_name}] Sample_ID validation failed ({len(errors)} errors):\n" + "\n".join(errors[:20])
        raise ValueError(msg)
    logger.info(f"[{modality_name}] Sample_ID validation passed for {len(df)} rows.")


def validate_cross_file_alignment(dfs: Dict[str, pd.DataFrame], config) -> None:
    """Verify all CSVs have the same set of Sample_IDs."""
    id_sets = {}
    for modality, df in dfs.items():
        id_sets[modality] = set(df[config.sample_id_col].astype(str))
    keys = list(id_sets.keys())
    reference = id_sets[keys[0]]
    for k in keys[1:]:
        if id_sets[k] != reference:
            extra = id_sets[k] - reference
            missing = reference - id_sets[k]
            raise ValueError(
                f"Sample_ID mismatch between {keys[0]} and {k}: "
                f"extra={extra}, missing={missing}"
            )
    logger.info(f"Cross-file alignment OK: all modalities share {len(reference)} Sample_IDs.")


def align_modalities(dfs: Dict[str, pd.DataFrame], config) -> pd.DataFrame:
    """Merge all three DataFrames on Sample_ID."""
    img_df = dfs["image"]
    ftir_df = dfs["ftir"]
    color_df = dfs["color"]

    merged = img_df.merge(ftir_df, on=config.sample_id_col, suffixes=("", "_ftir"))
    merged = merged.merge(color_df, on=config.sample_id_col, suffixes=("", "_color"))

    assert len(merged) == config.n_samples, (
        f"Expected {config.n_samples} rows after merge, got {len(merged)}"
    )

    out_path = Path(config.processed_dir) / "merged_all_modalities.csv"
    merged.to_csv(out_path, index=False)
    logger.info(f"Saved merged DataFrame to {out_path}, shape={merged.shape}")
    return merged


def get_feature_columns(df: pd.DataFrame, config, modality: str = "") -> List[str]:
    """Return feature column names, excluding ID, target, and merge artifacts."""
    exclude = {config.sample_id_col, config.target_col}
    cols = [
        c for c in df.columns
        if c not in exclude and not c.endswith("_x") and not c.endswith("_y")
        and not c.endswith("_ftir") and not c.endswith("_color")
    ]
    logger.info(f"Detected {len(cols)} feature columns in {modality} CSV")
    return cols


def load_image_dataset(config) -> Tuple[List[str], List[int]]:
    """Walk Image_Data directory and return (image_paths, labels).
    Does NOT load images into RAM.
    """
    img_root = Path(config.data_dir) / config.image_data_dir
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    image_paths = []
    labels = []

    if not img_root.exists():
        logger.warning(f"Image_Data directory not found: {img_root}")
        return [], []

    for level in config.class_labels:
        # Try both "0" and "0%" folder names
        for folder_name in [f"{level}%", str(level)]:
            level_dir = img_root / folder_name
            if level_dir.exists():
                break
        else:
            logger.warning(f"No folder found for level={level} in {img_root}")
            continue

        imgs = sorted([
            str(f) for f in level_dir.iterdir()
            if f.suffix.lower() in supported
        ])
        if len(imgs) != config.samples_per_class:
            logger.warning(
                f"Level {level}: expected {config.samples_per_class} images, found {len(imgs)}"
            )
        image_paths.extend(imgs)
        labels.extend([level] * len(imgs))

    logger.info(f"Loaded {len(image_paths)} image paths across {len(config.class_labels)} classes")
    return image_paths, labels
