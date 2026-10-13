
"""Automated tests for cybersecurity dataset preprocessing."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    preprocess_dataset,
    prepare_features_and_target,
)


def create_sample_dataset():
    """Create a small dataset with common data-quality problems."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "dur": [0.1, np.inf, 0.3, 0.4],
            "proto": [" tcp ", "?", "udp", "tcp"],
            "attack_cat": ["Normal", "DoS", "Normal", "Exploits"],
            "label": [0, 1, 0, 1],
        }
    )


def test_preprocessing_preserves_row_count():
    """Cleaning should not silently delete records."""
    original = create_sample_dataset()

    cleaned = preprocess_dataset(original)

    assert len(cleaned) == len(original)


def test_preprocessing_does_not_modify_original_dataframe():
    """Preprocessing should operate on a separate copy."""
    original = create_sample_dataset()
    original_copy = original.copy(deep=True)

    preprocess_dataset(original)

    pd.testing.assert_frame_equal(original, original_copy)


def test_preprocessing_normalizes_categorical_values():
    """Text whitespace and missing-value markers should be normalized."""
    cleaned = preprocess_dataset(create_sample_dataset())

    assert cleaned.loc[0, "proto"] == "tcp"
    assert pd.isna(cleaned.loc[1, "proto"])


def test_preprocessing_replaces_infinite_numeric_values():
    """Positive and negative infinity should become missing values."""
    dataframe = create_sample_dataset()
    dataframe.loc[1, "dur"] = np.inf
    dataframe.loc[2, "dur"] = -np.inf

    cleaned = preprocess_dataset(dataframe)

    assert pd.isna(cleaned.loc[1, "dur"])
    assert pd.isna(cleaned.loc[2, "dur"])


def test_preprocessing_validates_binary_labels():
    """The binary target must contain valid normal and attack labels."""
    cleaned = preprocess_dataset(create_sample_dataset())

    assert set(cleaned["label"].unique()) == {0, 1}
    assert cleaned["label"].dtype == np.dtype("int64")


def test_preprocessing_rejects_missing_labels():
    """Missing target labels must not silently enter the training data."""
    dataframe = create_sample_dataset()
    dataframe.loc[0, "label"] = np.nan

    with pytest.raises(ValueError, match="target labels"):
        preprocess_dataset(dataframe)


def test_preprocessing_rejects_unsupported_labels():
    """Labels outside the expected binary classes must be rejected."""
    dataframe = create_sample_dataset()
    dataframe.loc[0, "label"] = 2

    with pytest.raises(ValueError, match="unsupported binary labels"):
        preprocess_dataset(dataframe)


def test_preprocessing_rejects_single_class_data():
    """Both normal and attack labels are required in the prepared split."""
    dataframe = create_sample_dataset()
    dataframe["label"] = 0

    with pytest.raises(ValueError, match="only one target class"):
        preprocess_dataset(dataframe)


def test_feature_target_separation_prevents_target_leakage():
    """Identifier, target and attack-category fields must not enter X."""
    cleaned = preprocess_dataset(create_sample_dataset())

    features, target = prepare_features_and_target(cleaned)

    assert "label" not in features.columns
    assert "id" not in features.columns
    assert "attack_cat" not in features.columns

    assert len(features) == len(target)
    assert set(target.unique()) == {0, 1}


def test_preprocessing_rejects_empty_dataset():
    """An empty DataFrame should produce a useful error."""
    with pytest.raises(ValueError, match="is empty"):
        preprocess_dataset(pd.DataFrame())
