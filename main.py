
"""
Entry point for the AI-Powered Cybersecurity Threat Detection System.

Current capabilities:
- Load the prepared UNSW-NB15 datasets.
- Clean and validate the datasets.
- Separate predictive features from target labels.
- Engineer numerical and categorical features.
- Validate the transformed training and testing matrices.
"""

from src.data_loader import (
    TRAINING_PATH,
    TESTING_PATH,
    load_csv,
)

from src.preprocessing import (
    prepare_features_and_target,
    preprocess_dataset,
)

from src.feature_engineering import (
    engineer_features,
    print_feature_engineering_summary,
)


def load_and_prepare_dataset(
    dataset_name: str,
    file_path,
):
    """Load, clean, and separate one dataset into features and labels."""
    print(f"\nLoading {dataset_name}...")

    original = load_csv(file_path)

    cleaned = preprocess_dataset(
        original,
        dataset_name=dataset_name,
    )

    # Avoid keeping unnecessary copies of the original DataFrame.
    del original

    features, target = prepare_features_and_target(cleaned)

    del cleaned

    print(f"{dataset_name} feature rows: {len(features):,}")
    print(f"{dataset_name} target rows: {len(target):,}")

    return features, target


def main() -> int:
    """Execute dataset preparation and feature engineering."""
    print("=" * 72)
    print("AI-POWERED CYBERSECURITY THREAT DETECTION SYSTEM")
    print("Step 7: Feature Engineering")
    print("=" * 72)

    try:
        X_train, y_train = load_and_prepare_dataset(
            "Training dataset",
            TRAINING_PATH,
        )

        X_test, y_test = load_and_prepare_dataset(
            "Testing dataset",
            TESTING_PATH,
        )

        # Validate label distributions without fitting any model.
        print("\nTraining target distribution:")
        print(y_train.value_counts().sort_index().to_string())

        print("\nTesting target distribution:")
        print(y_test.value_counts().sort_index().to_string())

        (
            transformer,
            transformed_training,
            transformed_testing,
            numerical_columns,
            categorical_columns,
        ) = engineer_features(
            training_features=X_train,
            testing_features=X_test,
        )

        feature_names = transformer.get_feature_names_out()

        print_feature_engineering_summary(
            training_features=X_train,
            testing_features=X_test,
            transformed_training=transformed_training,
            transformed_testing=transformed_testing,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            transformed_feature_names=feature_names,
        )

        print("\nSUCCESS: Feature engineering completed.")
        print("The transformation was fitted on training data only.")
        print(
            "No classifier or anomaly-detection model has been trained "
            "in this step."
        )

    except (OSError, ValueError, TypeError) as error:
        print("\nERROR: Feature engineering could not be completed.")
        print(error)
        return 1

    finally:
        # Release loaded feature frames and label series when possible.
        for variable_name in (
            "X_train",
            "X_test",
            "y_train",
            "y_test",
        ):
            if variable_name in locals():
                del locals()[variable_name]

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
