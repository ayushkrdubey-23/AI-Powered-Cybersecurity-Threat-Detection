
"""Automated tests for numerical and categorical feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    engineer_features,
    identify_feature_types,
    validate_feature_frames,
)


def create_training_features():
    """Create training data with missing numerical and categorical values."""
    return pd.DataFrame(
        {
            "duration": [0.1, 0.2, np.nan, 0.4, 0.5],
            "bytes_sent": [100, 200, 150, 300, 250],
            "protocol": ["tcp", "udp", "tcp", None, "udp"],
        }
    )


def create_testing_features():
    """Include a protocol category absent from training."""
    return pd.DataFrame(
        {
            "duration": [0.6, 0.7],
            "bytes_sent": [400, 500],
            "protocol": ["icmp", "tcp"],
        }
    )


def test_identifies_numerical_and_categorical_columns():
    """Feature types should be identified using column data types."""
    training = create_training_features()

    numerical, categorical = identify_feature_types(training)

    assert set(numerical) == {"duration", "bytes_sent"}
    assert categorical == ["protocol"]


def test_training_and_testing_feature_schemas_align():
    """Matching columns with different ordering should be aligned."""
    training = create_training_features()

    testing = create_testing_features()[
        ["protocol", "bytes_sent", "duration"]
    ]

    aligned = validate_feature_frames(training, testing)

    assert aligned.columns.tolist() == training.columns.tolist()


def test_feature_engineering_handles_unseen_categories():
    """An unseen testing category should not crash the encoder."""
    training = create_training_features()
    testing = create_testing_features()

    (
        transformer,
        training_matrix,
        testing_matrix,
        numerical,
        categorical,
    ) = engineer_features(training, testing)

    assert training_matrix.shape[0] == len(training)
    assert testing_matrix.shape[0] == len(testing)

    assert training_matrix.shape[1] == testing_matrix.shape[1]
    assert training_matrix.shape[1] > 0

    assert "duration" in numerical
    assert "bytes_sent" in numerical
    assert "protocol" in categorical

    assert np.isfinite(training_matrix.data if hasattr(
        training_matrix, "data"
    ) and hasattr(training_matrix, "tocsr") else np.asarray(
        training_matrix
    )).all()

    assert hasattr(transformer, "transform")


def test_feature_engineering_rejects_mismatched_columns():
    """An unexpected training/testing schema must raise an error."""
    training = create_training_features()
    testing = create_testing_features().rename(
        columns={"bytes_sent": "unexpected_feature"}
    )

    with pytest.raises(ValueError, match="feature columns differ"):
        validate_feature_frames(training, testing)


def test_feature_engineering_rejects_empty_training_data():
    """No transformation should be fitted to an empty training frame."""
    with pytest.raises(ValueError, match="Training features are empty"):
        engineer_features(
            pd.DataFrame(),
            create_testing_features(),
        )


def test_transformed_matrix_has_consistent_dimensions():
    """Training and testing encodings must have matching column counts."""
    training = create_training_features()
    testing = create_testing_features()

    _, training_matrix, testing_matrix, _, _ = engineer_features(
        training,
        testing,
    )

    assert training_matrix.shape[1] == testing_matrix.shape[1]
