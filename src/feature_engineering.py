
"""
Feature engineering for the AI-Powered Cybersecurity Threat Detection System.

Responsibilities:
- Identify numerical and categorical features.
- Impute missing feature values.
- Scale numerical features.
- One-hot encode categorical features.
- Fit transformations on training data only.
- Apply the same fitted transformations to testing data.
- Validate transformed output without converting sparse matrices to dense ones.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd

from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def identify_feature_types(
    dataframe: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """
    Identify numerical and categorical feature columns.

    Numerical columns are selected by Pandas numeric dtype.
    All remaining feature columns are treated as categorical.

    Returns
    -------
    numerical_columns, categorical_columns
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if dataframe.empty:
        raise ValueError("Cannot identify features in an empty DataFrame.")

    if dataframe.columns.duplicated().any():
        raise ValueError("Feature names must be unique.")

    numerical_columns = dataframe.select_dtypes(
        include=["number"]
    ).columns.tolist()

    numerical_set = set(numerical_columns)

    categorical_columns = [
        column
        for column in dataframe.columns
        if column not in numerical_set
    ]

    if not numerical_columns and not categorical_columns:
        raise ValueError("No usable feature columns were found.")

    return numerical_columns, categorical_columns


def build_preprocessor(
    training_features: pd.DataFrame,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Build the feature transformation pipeline.

    Numeric pipeline:
    - Replace missing values with the training median.
    - Standardize numeric values.

    Categorical pipeline:
    - Replace missing values with a constant category.
    - One-hot encode categories.
    - Ignore categories that were not observed during training.

    The returned transformer must be fitted on training data only.
    """
    if training_features.empty:
        raise ValueError("Training features are empty.")

    numerical_columns, categorical_columns = identify_feature_types(
        training_features
    )

    transformers = []

    if numerical_columns:
        numerical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        keep_empty_features=True,
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                numerical_pipeline,
                numerical_columns,
            )
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="__MISSING__",
                        keep_empty_features=True,
                    ),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=True,
                        dtype=np.float32,
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )

    return transformer, numerical_columns, categorical_columns


def validate_feature_frames(
    training_features: pd.DataFrame,
    testing_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Check that training and testing have the same feature names.

    If the names match but their order differs, testing features are
    reordered to match the training feature order.
    """
    if not isinstance(training_features, pd.DataFrame):
        raise TypeError("Training features must be a DataFrame.")

    if not isinstance(testing_features, pd.DataFrame):
        raise TypeError("Testing features must be a DataFrame.")

    if training_features.empty:
        raise ValueError("Training features are empty.")

    if testing_features.empty:
        raise ValueError("Testing features are empty.")

    if training_features.columns.duplicated().any():
        raise ValueError("Training feature names contain duplicates.")

    if testing_features.columns.duplicated().any():
        raise ValueError("Testing feature names contain duplicates.")

    training_names = set(training_features.columns)
    testing_names = set(testing_features.columns)

    if training_names != testing_names:
        missing_from_test = sorted(
            training_names - testing_names
        )
        extra_in_test = sorted(
            testing_names - training_names
        )

        raise ValueError(
            "Training and testing feature columns differ. "
            f"Missing from testing: {missing_from_test}. "
            f"Extra in testing: {extra_in_test}."
        )

    return testing_features.loc[
        :,
        training_features.columns,
    ].copy()


def check_transformed_values(matrix, dataset_name: str) -> None:
    """
    Ensure transformed features contain no NaN or infinite values.

    Sparse matrices are checked through their stored values without
    expanding them into large dense arrays.
    """
    if sparse.issparse(matrix):
        values = matrix.data
    else:
        values = np.asarray(matrix)

    if not np.isfinite(values).all():
        raise ValueError(
            f"{dataset_name} contains NaN or infinite transformed values."
        )


def get_matrix_density(matrix) -> float:
    """
    Return the fraction of nonzero entries in a matrix.
    """
    total_entries = matrix.shape[0] * matrix.shape[1]

    if total_entries == 0:
        return 0.0

    if sparse.issparse(matrix):
        return float(matrix.nnz / total_entries)

    return float(np.count_nonzero(matrix) / total_entries)


def engineer_features(
    training_features: pd.DataFrame,
    testing_features: pd.DataFrame,
):
    """
    Fit feature engineering on the training data and transform both sets.

    Returns
    -------
    transformer:
        Fitted ColumnTransformer.
    transformed_training:
        Training feature matrix.
    transformed_testing:
        Testing feature matrix.
    numerical_columns:
        Original numerical feature names.
    categorical_columns:
        Original categorical feature names.
    """
    aligned_testing = validate_feature_frames(
        training_features,
        testing_features,
    )

    # Build and fit every transformation using the training dataset only.
    transformer, numerical_columns, categorical_columns = (
        build_preprocessor(training_features)
    )

    transformed_training = transformer.fit_transform(
        training_features
    )

    # No fitting is performed on testing data.
    transformed_testing = transformer.transform(
        aligned_testing
    )

    if transformed_training.shape[0] != len(training_features):
        raise ValueError(
            "The transformed training row count does not match "
            "the original training row count."
        )

    if transformed_testing.shape[0] != len(aligned_testing):
        raise ValueError(
            "The transformed testing row count does not match "
            "the original testing row count."
        )

    if transformed_training.shape[1] != transformed_testing.shape[1]:
        raise ValueError(
            "Training and testing transformed feature dimensions differ."
        )

    check_transformed_values(
        transformed_training,
        "Transformed training data",
    )

    check_transformed_values(
        transformed_testing,
        "Transformed testing data",
    )

    return (
        transformer,
        transformed_training,
        transformed_testing,
        numerical_columns,
        categorical_columns,
    )


def print_feature_engineering_summary(
    training_features: pd.DataFrame,
    testing_features: pd.DataFrame,
    transformed_training,
    transformed_testing,
    numerical_columns: List[str],
    categorical_columns: List[str],
    transformed_feature_names,
) -> None:
    """Print the feature engineering results."""
    print("\n" + "=" * 72)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 72)

    print("\nOriginal input features:")
    print(f"  Training rows: {len(training_features):,}")
    print(f"  Testing rows: {len(testing_features):,}")
    print(f"  Original feature count: {training_features.shape[1]:,}")

    print("\nFeature types:")
    print(f"  Numerical columns: {len(numerical_columns):,}")
    print(f"  Categorical columns: {len(categorical_columns):,}")

    print("\nTransformation results:")
    print(
        f"  Transformed training shape: "
        f"{transformed_training.shape}"
    )
    print(
        f"  Transformed testing shape: "
        f"{transformed_testing.shape}"
    )
    print(
        f"  Transformed feature count: "
        f"{len(transformed_feature_names):,}"
    )

    print("\nMatrix storage:")
    print(
        "  Training matrix type: "
        f"{'sparse' if sparse.issparse(transformed_training) else 'dense'}"
    )
    print(
        "  Testing matrix type: "
        f"{'sparse' if sparse.issparse(transformed_testing) else 'dense'}"
    )
    print(
        f"  Training matrix density: "
        f"{get_matrix_density(transformed_training):.4f}"
    )
    print(
        f"  Testing matrix density: "
        f"{get_matrix_density(transformed_testing):.4f}"
    )

    print("\nSample transformed feature names:")

    for feature_name in transformed_feature_names[:15]:
        print(f"  {feature_name}")

    if len(transformed_feature_names) > 15:
        print("  ... additional feature names omitted")

    print("\nValidation:")
    print("  Missing transformed values: none")
    print("  Infinite transformed values: none")
    print("  Training/testing feature dimensions: consistent")
    print("  Test data used to fit the encoder: no")

    print("=" * 72)


def main() -> int:
    """
    Run an independent synthetic test of the feature engineering module.
    """
    training = pd.DataFrame(
        {
            "duration": [0.1, 0.2, np.nan, 0.4],
            "bytes_sent": [100, 200, 150, 300],
            "protocol": ["tcp", "udp", "tcp", None],
        }
    )

    testing = pd.DataFrame(
        {
            "duration": [0.5, 0.6],
            "bytes_sent": [400, 500],
            # 'icmp' is intentionally absent from training categories.
            "protocol": ["icmp", "tcp"],
        }
    )

    try:
        (
            transformer,
            transformed_training,
            transformed_testing,
            numerical_columns,
            categorical_columns,
        ) = engineer_features(training, testing)

        transformed_names = transformer.get_feature_names_out()

        assert transformed_training.shape[0] == len(training)
        assert transformed_testing.shape[0] == len(testing)

        assert (
            transformed_training.shape[1]
            == transformed_testing.shape[1]
        )

        assert "duration" in numerical_columns
        assert "bytes_sent" in numerical_columns
        assert "protocol" in categorical_columns

        check_transformed_values(
            transformed_training,
            "Synthetic training data",
        )

        check_transformed_values(
            transformed_testing,
            "Synthetic testing data",
        )

        print_feature_engineering_summary(
            training_features=training,
            testing_features=testing,
            transformed_training=transformed_training,
            transformed_testing=transformed_testing,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            transformed_feature_names=transformed_names,
        )

        print("\nSUCCESS: Feature engineering self-test passed.")
        print("The unseen 'icmp' category was handled successfully.")
        print("The test data did not fit or refit the encoder.")

        return 0

    except (AssertionError, TypeError, ValueError) as error:
        print(f"\nERROR: Feature engineering self-test failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
