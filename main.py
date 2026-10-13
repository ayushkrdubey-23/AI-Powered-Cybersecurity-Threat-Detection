
"""
Entry point for the AI-Powered Cybersecurity Threat Detection System.

Current capabilities:
- Load the prepared UNSW-NB15 datasets.
- Inspect dataset quality.
- Apply basic data cleaning and label validation.
- Separate predictive features from target labels.
"""

from src.data_loader import (
    TRAINING_PATH,
    TESTING_PATH,
    load_csv,
    print_dataset_summary,
)

from src.preprocessing import (
    prepare_features_and_target,
    preprocess_dataset,
    print_preprocessing_summary,
)


def inspect_dataset(dataset_name: str, file_path) -> list:
    """
    Load, inspect, and preprocess one dataset.

    Returns the ordered predictive feature column names.
    """
    print(f"\nLoading {dataset_name}...")

    original = load_csv(file_path)

    print_dataset_summary(
        dataframe=original,
        dataset_name=f"{dataset_name} (before cleaning)",
        file_path=file_path,
    )

    cleaned = preprocess_dataset(
        original,
        dataset_name=dataset_name,
    )

    print_preprocessing_summary(
        dataframe_before=original,
        dataframe_after=cleaned,
        dataset_name=dataset_name,
    )

    features, target = prepare_features_and_target(cleaned)

    print("\nPrepared data dimensions:")
    print(f"  Feature rows: {features.shape[0]:,}")
    print(f"  Feature columns: {features.shape[1]:,}")
    print(f"  Target records: {len(target):,}")

    # Save only the column names, not another copy of the full dataset.
    feature_columns = features.columns.tolist()

    del original
    del cleaned
    del features
    del target

    return feature_columns


def main() -> int:
    """Run dataset loading, basic cleaning, and consistency checks."""
    print("=" * 72)
    print("AI-POWERED CYBERSECURITY THREAT DETECTION SYSTEM")
    print("Step 6: Data Cleaning and Preprocessing")
    print("=" * 72)

    try:
        training_columns = inspect_dataset(
            "Training dataset",
            TRAINING_PATH,
        )

        testing_columns = inspect_dataset(
            "Testing dataset",
            TESTING_PATH,
        )

    except (OSError, ValueError, TypeError) as error:
        print("\nERROR: Dataset preprocessing could not be completed.")
        print(error)
        return 1

    training_set = set(training_columns)
    testing_set = set(testing_columns)

    missing_from_test = sorted(training_set - testing_set)
    extra_in_test = sorted(testing_set - training_set)

    if missing_from_test or extra_in_test:
        print("\nERROR: Training and testing feature columns do not match.")

        if missing_from_test:
            print(f"Missing from testing dataset: {missing_from_test}")

        if extra_in_test:
            print(f"Additional testing columns: {extra_in_test}")

        print(
            "Resolve the schema mismatch before training so that "
            "both datasets use consistent features."
        )
        return 1

    if training_columns != testing_columns:
        print(
            "\nFeature names match, but their order differs. "
            "The testing dataset will need to be aligned to training "
            "feature order when the model pipeline is implemented."
        )
    else:
        print("\nTraining and testing feature names and order match.")

    print("\nSUCCESS: Dataset cleaning and validation completed.")
    print("Rows were not automatically deleted.")
    print("No machine-learning model has been trained in this step.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
