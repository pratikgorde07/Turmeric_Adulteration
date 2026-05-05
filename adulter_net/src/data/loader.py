from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import get_logger

logger = get_logger(__name__)


def load_all_modalities(config) -> Dict[str, pd.DataFrame]:
    """Load all three CSV modalities and return as dict of DataFrames."""
    dfs = {}
    for key, fname in [
        ("image", config.image_feat_file),
        ("ftir", config.ftir_file),
        ("color", config.color_file),
    ]:
        path = config.data_dir / fname
        logger.info(f"Loading {key} CSV from: {path}")
        df = pd.read_csv(path)
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Columns (first 5): {list(df.columns[:5])}")
        logger.info(f"  NaN count: {df.isna().sum().sum()}")
        assert config.sample_id_col in df.columns, (
            f"{key} CSV missing column '{config.sample_id_col}'"
        )
        assert config.target_col in df.columns, (
            f"{key} CSV missing column '{config.target_col}'"
        )
        dfs[key] = df

    return dfs


def parse_sample_id(sample_id: str) -> Dict[str, int]:
    """Parse T{level}S{nn} format. Raises ValueError on mismatch."""
    pattern = r"^T(\d+)S(\d{2})$"
    m = re.match(pattern, str(sample_id).strip())
    if m is None:
        raise ValueError(
            f"Sample_ID '{sample_id}' does not match expected format T{{level}}S{{nn}}"
        )
    return {
        "adulteration_level": int(m.group(1)),
        "sample_number": int(m.group(2)),
    }


def validate_sample_ids(dfs: Dict[str, pd.DataFrame], config) -> None:
    """Validate all Sample_IDs across all modality DataFrames."""
    logger.info("Validating Sample_IDs...")

    for modality_name, df in dfs.items():
        logger.info(f"  Validating {modality_name}...")
        errors = []
        for idx, row in df.iterrows():
            sid = row[config.sample_id_col]
            target = row[config.target_col]
            try:
                parsed = parse_sample_id(sid)
            except ValueError as e:
                errors.append(str(e))
                continue

            level = parsed["adulteration_level"]
            if level != int(target):
                errors.append(
                    f"Row {idx}: Sample_ID '{sid}' level={level} != Target={target}"
                )

            if parsed["sample_number"] < 1 or parsed["sample_number"] > 50:
                errors.append(
                    f"Row {idx}: sample_number {parsed['sample_number']} out of [01,50]"
                )

        if errors:
            for e in errors[:10]:
                logger.error(f"    {e}")
            raise ValueError(
                f"{modality_name}: {len(errors)} Sample_ID validation error(s). "
                f"First: {errors[0]}"
            )

        # Check sample numbers per level
        for level in config.class_labels:
            level_df = df[df[config.target_col] == level]
            nums = sorted(
                [parse_sample_id(sid)["sample_number"] for sid in level_df[config.sample_id_col]]
            )
            expected = list(range(1, len(nums) + 1))
            if nums != expected and len(nums) > 0:
                logger.warning(
                    f"  {modality_name} level={level}: sample numbers {nums[:5]}... "
                    f"(expected consecutive from 1)"
                )

        logger.info(f"  {modality_name}: {len(df)} rows validated OK")

    # Cross-file alignment check
    id_sets = {name: set(df[config.sample_id_col]) for name, df in dfs.items()}
    names = list(id_sets.keys())
    for i in range(1, len(names)):
        diff = id_sets[names[0]].symmetric_difference(id_sets[names[i]])
        if diff:
            raise ValueError(
                f"Sample_ID mismatch between '{names[0]}' and '{names[i]}': {len(diff)} differences. "
                f"Examples: {list(diff)[:5]}"
            )

    logger.info("Sample_ID validation PASSED for all modalities.")


def align_modalities(dfs: Dict[str, pd.DataFrame], config) -> pd.DataFrame:
    """Merge all three DataFrames on Sample_ID. Returns merged DataFrame."""
    df_img = dfs["image"]
    df_ftir = dfs["ftir"]
    df_color = dfs["color"]

    merged = df_img.merge(df_ftir, on=config.sample_id_col, suffixes=("", "_ftir"))
    merged = merged.merge(df_color, on=config.sample_id_col, suffixes=("", "_color"))

    assert len(merged) == config.n_samples, (
        f"Merged DataFrame has {len(merged)} rows, expected {config.n_samples}"
    )

    out_path = config.processed_dir / "merged_all_modalities.csv"
    merged.to_csv(out_path, index=False)
    logger.info(f"Merged DataFrame saved to {out_path} (shape={merged.shape})")

    return merged


def get_feature_columns(df: pd.DataFrame, config, modality: str = "unknown") -> List[str]:
    """Return feature column names excluding ID, target, and merge artifacts."""
    exclude = {config.sample_id_col, config.target_col}
    cols = [
        c for c in df.columns
        if c not in exclude and not c.endswith("_x") and not c.endswith("_y")
    ]
    logger.info(f"Detected {len(cols)} feature columns in {modality} CSV")
    return cols


def audit_data_quality(df: pd.DataFrame, modality_name: str, config) -> Dict:
    """Audit NaN, Inf, near-constant columns. Fix in-place and return report."""
    report_lines = [f"=== Data Quality: {modality_name} ==="]
    result = {}

    nan_counts = df.isna().sum()
    inf_counts = df.apply(lambda col: np.isinf(col.values.astype(float)).sum()
                          if col.dtype.kind in "fiu" else 0)

    total_nan = int(nan_counts.sum())
    total_inf = int(inf_counts.sum())
    result["nan_count"] = total_nan
    result["inf_count"] = total_inf

    if total_nan > 0:
        for col in nan_counts[nan_counts > 0].index:
            med = df[col].median()
            logger.warning(f"[{modality_name}] Filling {nan_counts[col]} NaNs in '{col}' with median={med:.4f}")
            df[col].fillna(med, inplace=True)

    if total_inf > 0:
        for col in inf_counts[inf_counts > 0].index:
            finite_max = df[col].replace([np.inf, -np.inf], np.nan).max()
            replacement = finite_max * 1.5
            logger.warning(f"[{modality_name}] Replacing Inf in '{col}' with max*1.5={replacement:.4f}")
            df[col].replace([np.inf, -np.inf], replacement, inplace=True)

    # Near-constant columns (CV < 0.001)
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in {config.sample_id_col, config.target_col}]
    near_const = []
    for col in feature_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        if abs(mean_val) > 1e-10:
            cv = std_val / abs(mean_val)
            if cv < 0.001:
                near_const.append((col, cv))

    result["near_constant_cols"] = len(near_const)
    report_lines.append(f"NaN filled: {total_nan}")
    report_lines.append(f"Inf replaced: {total_inf}")
    report_lines.append(f"Near-constant cols: {len(near_const)}")
    for col, cv in near_const[:10]:
        report_lines.append(f"  {col}: CV={cv:.6f}")

    report_path = config.output_dir / "reports" / f"data_quality_{modality_name}.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Data quality report saved: {report_path}")
    return result


def load_image_dataset(config) -> Tuple[List[str], List[int]]:
    """Walk Image_Data subfolders and return (image_paths, labels).

    Returns empty lists if Image_Data directory doesn't exist (feature-only mode).
    """
    image_dir = config.data_dir / config.image_data_dir
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    if not image_dir.exists():
        logger.warning(f"Image_Data directory not found at {image_dir}. Using feature-only mode.")
        return [], []

    image_paths = []
    labels = []
    level_map = {str(level): level for level in config.class_labels}

    for level in sorted(config.class_labels):
        # Try both "0%" and "0" subfolder names
        candidates = [
            image_dir / f"{level}%",
            image_dir / str(level),
        ]
        subfolder = None
        for c in candidates:
            if c.exists():
                subfolder = c
                break

        if subfolder is None:
            logger.warning(f"No subfolder found for level={level} in {image_dir}")
            continue

        imgs = sorted([
            str(p) for p in subfolder.iterdir()
            if p.suffix.lower() in valid_exts
        ])

        if len(imgs) != config.samples_per_class:
            logger.warning(
                f"Level={level}: found {len(imgs)} images, expected {config.samples_per_class}"
            )

        image_paths.extend(imgs)
        labels.extend([level] * len(imgs))
        logger.info(f"  Level {level}%: {len(imgs)} images from {subfolder}")

    logger.info(f"Total images loaded: {len(image_paths)}")
    return image_paths, labels
