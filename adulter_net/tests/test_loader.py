from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_parse_sample_id_valid():
    from src.data.loader import parse_sample_id
    result = parse_sample_id("T0S01")
    assert result["adulteration_level"] == 0
    assert result["sample_number"] == 1


def test_parse_sample_id_level9():
    from src.data.loader import parse_sample_id
    result = parse_sample_id("T9S50")
    assert result["adulteration_level"] == 9
    assert result["sample_number"] == 50


def test_parse_sample_id_two_digit_level():
    from src.data.loader import parse_sample_id
    result = parse_sample_id("T10S25")
    assert result["adulteration_level"] == 10
    assert result["sample_number"] == 25


def test_parse_sample_id_invalid_format():
    from src.data.loader import parse_sample_id
    with pytest.raises(ValueError):
        parse_sample_id("INVALID")


def test_parse_sample_id_missing_s():
    from src.data.loader import parse_sample_id
    with pytest.raises(ValueError):
        parse_sample_id("T0_01")


def test_parse_sample_id_single_digit_sample():
    from src.data.loader import parse_sample_id
    # S1 (single digit) should fail — must be S{nn} two digits
    with pytest.raises(ValueError):
        parse_sample_id("T0S1")


def test_get_feature_columns_removes_ids():
    import pandas as pd
    from src.data.loader import get_feature_columns

    class MockConfig:
        sample_id_col = "Sample_ID"
        target_col = "Target"

    df = pd.DataFrame({
        "Sample_ID": ["T0S01"],
        "Target": [0],
        "feat1": [1.0],
        "feat2": [2.0],
        "feat3_x": [3.0],
    })
    cols = get_feature_columns(df, MockConfig())
    assert "Sample_ID" not in cols
    assert "Target" not in cols
    assert "feat3_x" not in cols
    assert "feat1" in cols
    assert "feat2" in cols


def test_get_feature_columns_no_y_suffix():
    import pandas as pd
    from src.data.loader import get_feature_columns

    class MockConfig:
        sample_id_col = "Sample_ID"
        target_col = "Target"

    df = pd.DataFrame({
        "Sample_ID": ["T0S01"],
        "Target": [0],
        "wn_1000": [0.5],
        "wn_2000_y": [0.3],
    })
    cols = get_feature_columns(df, MockConfig())
    assert "wn_1000" in cols
    assert "wn_2000_y" not in cols
