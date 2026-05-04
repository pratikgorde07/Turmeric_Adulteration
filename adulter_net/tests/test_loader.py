"""Tests for data loader and Sample_ID validation."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import pandas as pd


def _make_config():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from config import Config
    return Config()


def test_parse_sample_id_valid():
    from src.data.loader import parse_sample_id
    result = parse_sample_id("T0S01")
    assert result["adulteration_level"] == 0
    assert result["sample_number"] == 1


def test_parse_sample_id_multidigit_level():
    from src.data.loader import parse_sample_id
    result = parse_sample_id("T9S50")
    assert result["adulteration_level"] == 9
    assert result["sample_number"] == 50


def test_parse_sample_id_invalid_format():
    from src.data.loader import parse_sample_id
    with pytest.raises(ValueError, match="Invalid Sample_ID"):
        parse_sample_id("INVALID_001")


def test_parse_sample_id_missing_prefix():
    from src.data.loader import parse_sample_id
    with pytest.raises(ValueError):
        parse_sample_id("0S01")


def test_sample_id_validation_passes():
    from src.data.loader import validate_sample_ids
    config = _make_config()
    rows = []
    for level in config.class_labels:
        for n in range(1, 51):
            rows.append({
                config.sample_id_col: f"T{level}S{n:02d}",
                config.target_col: level,
            })
    df = pd.DataFrame(rows)
    validate_sample_ids(df, config, "test")


def test_sample_id_validation_fails_mismatch():
    from src.data.loader import validate_sample_ids
    config = _make_config()
    rows = [
        {config.sample_id_col: "T0S01", config.target_col: 1},  # mismatch!
    ]
    df = pd.DataFrame(rows)
    with pytest.raises(ValueError):
        validate_sample_ids(df, config, "test")


def test_get_feature_columns_excludes_metadata():
    from src.data.loader import get_feature_columns
    config = _make_config()
    df = pd.DataFrame({
        config.sample_id_col: ["T0S01"],
        config.target_col: [0],
        "feat_1": [1.0],
        "feat_2": [2.0],
        "feat_1_x": [3.0],  # merge artifact
    })
    cols = get_feature_columns(df, config, "test")
    assert "feat_1" in cols
    assert "feat_2" in cols
    assert config.sample_id_col not in cols
    assert config.target_col not in cols
    assert "feat_1_x" not in cols
