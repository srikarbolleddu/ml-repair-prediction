import math

import pandas as pd
import pytest

from app.features import (
    DERIVED_FEATURES,
    FEATURE_ORDER,
    LEAKY_COLUMNS,
    MODEL_FEATURES,
    TARGET,
    clean_raw,
    records_to_frame,
)


def _raw_row(**overrides):
    row = {
        "UDI": 1,
        "Product ID": "M14860",
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 10,
        "Machine failure": 0,
        **dict.fromkeys(LEAKY_COLUMNS, 0),
    }
    row.update(overrides)
    return row


def test_clean_raw_drops_leaky_failure_modes_and_ids():
    df = clean_raw(pd.DataFrame([_raw_row(), _raw_row(**{"Machine failure": 1, "HDF": 1})]))

    assert list(df.columns) == [*MODEL_FEATURES, TARGET]
    assert not set(LEAKY_COLUMNS) & set(df.columns)
    assert "UDI" not in df.columns and "Product ID" not in df.columns
    assert df[TARGET].tolist() == [0, 1]


def test_physics_features():
    row = clean_raw(pd.DataFrame([_raw_row()])).iloc[0]

    assert row["power_w"] == pytest.approx(42.8 * 1551 * 2 * math.pi / 60)
    assert row["temp_diff_k"] == pytest.approx(10.5)
    assert row["strain_min_nm"] == pytest.approx(428.0)


def test_records_to_frame_matches_training_columns():
    record = clean_raw(pd.DataFrame([_raw_row()])).iloc[0][FEATURE_ORDER].to_dict()

    frame = records_to_frame([record])

    assert list(frame.columns) == MODEL_FEATURES
    assert set(DERIVED_FEATURES) <= set(frame.columns)
    assert frame.loc[0, "temp_diff_k"] == pytest.approx(10.5)
