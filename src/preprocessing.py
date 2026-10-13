
"""
Data cleaning and preprocessing utilities.

This module prepares UNSW-NB15 data for binary network intrusion
classification. It does not train or evaluate a machine-learning model.
"""

from typing import Tuple

import numpy as np
import pandas as pd


# Common text values that should be interpreted as missing data.
MISSING_TEXT_VALUES = {
    "",
    "?",
    "na",
    "n/a",
    "nan",
    "null",
    "none",
}

# These columns are not used as predictive features in the initial
# binary normal-versus-attack classification model.
NON_FEATURE_COLUMNS = {
    "id",
    "attack_cat",
    "label",
}


def preprocess_dataset(
    dataframe: pd.DataFrame,
    dataset_name: str = "Dataset",
) -> pd.DataFrame:
    """
    Standardize common missing-value markers and validate labels.

    This function:
    1. Creates a copy so that the original DataFrame is unchanged.
    2. Cleans column names.
    3. Standardizes categorical text values.
    4. Converts common missing-value markers into actual missing values.
    5. Replaces infinite numeric values with missing values.
    6. Validates and normalizes the binary target label.

    Rows are not automatically deleted. Missing feature values are
    preserved for handling by the machine-learning pipeline later.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError(
            "Expected a pandas DataFrame for preprocessing."
        )

    if dataframe.empty:
        raise ValueError(
            f"{dataset_name} is empty. No preprocessing was performed."
        )

    cleaned = dataframe.copy(deep=True)

    # Normalize whitespace around column names.
    cleaned.columns = [
        str(column).strip() for column in cleaned.columns
    ]

    if cleaned.columns.duplicated().any():
        duplicate_columns = cleaned.columns[
            cleaned.columns.duplicated()
        ].tolist()

        raise ValueError(
            f"{dataset_name} contains duplicate column names: "
            f"{duplicate_columns}"
        )

    if "label" not in cleaned.columns:
        raise ValueError(
            f"{dataset_name} does not contain the required 'label' "
            "column. Confirm that the prepared UNSW-NB15 dataset "
            "was downloaded correctly."
        )

    # Standardize text columns without changing the original data.
    # Missing text markers are converted to NumPy NaN so downstream
    # Pandas and Scikit-learn processing can handle them consistently.
    text_columns = cleaned.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    for column in text_columns:
        normalized = cleaned[column].astype("string").str.strip()

        missing_mask = (
            normalized.str.casefold()
            .isin(MISSING_TEXT_VALUES)
            .fillna(False)
        )

        normalized = normalized.mask(missing_mask, pd.NA)

        # Convert to object dtype and replace nullable values with
        # np.nan for broad compatibility with preprocessing pipelines.
        normalized = normalized.astype(object)
        normalized[pd.isna(normalized)] = np.nan

        cleaned[column] = normalized

    # A binary classifier requires numeric target labels.
    numeric_labels = pd.to_numeric(
        cleaned["label"],
        errors="coerce",
    )

    if numeric_labels.isna().any():
        invalid_count = int(numeric_labels.isna().sum())

        raise ValueError(
            f"{dataset_name} contains {invalid_count:,} missing or "
            "non-numeric target labels. Correct the target data "
            "before training."
        )

    # Verify that the classification task is exactly normal vs attack.
    observed_labels = set(numeric_labels.unique().tolist())

    invalid_labels = observed_labels - {0, 1}

    if invalid_labels:
        raise ValueError(
            f"{dataset_name} contains unsupported binary labels: "
            f"{sorted(invalid_labels)}. Expected only 0 and 1."
        )

    if len(observed_labels) < 2:
        raise ValueError(
            f"{dataset_name} contains only one target class. "
            "Both normal and attack records are required for the "
            "planned classification workflow."
        )

    cleaned["label"] = numeric_labels.astype("int64")

    # Infinite numerical values cannot be used directly as reliable
    # model inputs. Replace them with missing values for later imputation.
    numeric_columns = cleaned.select_dtypes(
        include=[np.number]
    ).columns

    for column in numeric_columns:
        cleaned[column] = cleaned[column].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    return cleaned


def prepare_features_and_target(
    dataframe: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate model input features from the binary target label.

    Returns
    -------
    X:
        Predictive input features.
    y:
        Binary target labels (0 = normal, 1 = attack).

    Identifier and attack-category columns are excluded. The target
    label is kept separately and is never passed into X.
    """
    if "label" not in dataframe.columns:
        raise ValueError(
            "Cannot separate features and target: 'label' is missing."
        )

    target = dataframe["label"].copy()

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    if not feature_columns:
        raise ValueError(
            "No predictive features remain after excluding target "
            "and non-feature columns."
        )

    features = dataframe.loc[:, feature_columns].copy()

    if features.shape[1] == 0:
        raise ValueError("The feature DataFrame contains no columns.")

    return features, target


def print_preprocessing_summary(
    dataframe_before: pd.DataFrame,
    dataframe_after: pd.DataFrame,
    dataset_name: str = "Dataset",
) -> None:
    """
    Print a summary of the data-cleaning results.
    """
    print("\n" + "=" * 72)
    print(f"PREPROCESSING SUMMARY: {dataset_name}")
    print("=" * 72)

    print(f"Rows preserved: {len(dataframe_after):,}")
    print(f"Columns preserved: {len(dataframe_after.columns):,}")

    missing_before = int(dataframe_before.isna().sum().sum())
    missing_after = int(dataframe_after.isna().sum().sum())

    print(f"Missing cells before cleaning: {missing_before:,}")
    print(f"Missing cells after cleaning: {missing_after:,}")

    print(
        "Rows removed automatically: 0 "
        "(automatic row deletion is disabled)"
    )

    duplicate_count = int(dataframe_after.duplicated().sum())

    print(f"Fully duplicated rows detected: {duplicate_count:,}")
    print("Duplicate rows have been preserved.")

    print("\nBinary target distribution:")

    target_counts = dataframe_after["label"].value_counts().sort_index()
    total_rows = len(dataframe_after)

    for label, count in target_counts.items():
        percentage = count / total_rows * 100

        description = (
            "Normal traffic" if label == 0 else "Attack traffic"
        )

        print(
            f"  {label} ({description}): {count:,} records "
            f"({percentage:.2f}%)"
        )

    features, target = prepare_features_and_target(dataframe_after)

    print("\nModel input checks:")
    print(f"  Predictive feature count: {features.shape[1]:,}")
    print(f"  Target record count: {len(target):,}")
    print(f"  Target column passed to features: {'label' in features.columns}")
    print(f"  Identifier passed to features: {'id' in features.columns}")
    print(
        "  Attack category passed to features: "
        f"{'attack_cat' in features.columns}"
    )

    print("\nNote:")
    print(
        "Missing feature values remain available for handling by "
        "the training pipeline in a later step."
    )

    print("=" * 72)


def main() -> int:
    """
    Run a small synthetic check of preprocessing.

    The real dataset workflow is executed by the project's root main.py.
    This self-test verifies basic cleaning and feature separation without
    requiring the full UNSW-NB15 dataset.
    """
    sample = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "dur": [0.2, np.inf, 0.4, 0.5],
            "proto": [" tcp ", "udp", "?", "tcp"],
            "attack_cat": ["Normal", "DoS", "Normal", "Exploits"],
            "label": [0, 1, 0, 1],
        }
    )

    try:
        cleaned = preprocess_dataset(
            sample,
            dataset_name="Synthetic preprocessing self-test",
        )

        print_preprocessing_summary(
            dataframe_before=sample,
            dataframe_after=cleaned,
            dataset_name="Synthetic preprocessing self-test",
        )

        features, target = prepare_features_and_target(cleaned)

        assert len(features) == len(sample)
        assert len(target) == len(sample)
        assert "label" not in features.columns
        assert "id" not in features.columns
        assert "attack_cat" not in features.columns
        assert features["dur"].isna().sum() == 1
        assert features["proto"].isna().sum() == 1

        print("\nSUCCESS: Preprocessing self-test passed.")
        print("No real dataset was modified.")
        return 0

    except (AssertionError, TypeError, ValueError) as error:
        print(f"\nERROR: Preprocessing self-test failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
