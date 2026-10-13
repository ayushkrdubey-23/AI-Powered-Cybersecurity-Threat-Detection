
"""
Model training for the AI-Powered Cybersecurity Threat Detection System.

Models:
1. Logistic Regression baseline
2. Random Forest classifier

Important:
- Only the prepared training dataset is used for fitting.
- The testing dataset is not used in this module.
- The fitted preprocessing transformer is saved for later inference.
- This module does not report test-set performance.
"""

from pathlib import Path
from time import perf_counter

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.data_loader import TRAINING_PATH, load_csv
from src.preprocessing import (
    preprocess_dataset,
    prepare_features_and_target,
)
from src.feature_engineering import (
    build_preprocessor,
    check_transformed_values,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


def load_training_data():
    """
    Load and clean the training dataset.

    Returns
    -------
    features : pandas.DataFrame
        Predictive features only.
    target : pandas.Series
        Binary label: 0 = normal, 1 = attack.
    """
    print("\nLoading training dataset...")

    original = load_csv(TRAINING_PATH)

    print(f"Original training rows: {len(original):,}")
    print(f"Original training columns: {len(original.columns):,}")

    cleaned = preprocess_dataset(
        original,
        dataset_name="Training dataset",
    )

    del original

    features, target = prepare_features_and_target(cleaned)

    del cleaned

    if len(features) != len(target):
        raise ValueError(
            "Training feature and target row counts do not match."
        )

    observed_labels = set(target.unique().tolist())

    if observed_labels != {0, 1}:
        raise ValueError(
            "Both classes are required for binary classification. "
            f"Observed labels: {sorted(observed_labels)}"
        )

    if features.empty:
        raise ValueError("Training features are empty.")

    if features.columns.duplicated().any():
        raise ValueError("Training feature names contain duplicates.")

    if "label" in features.columns:
        raise ValueError(
            "Target leakage detected: label appears among features."
        )

    if "attack_cat" in features.columns:
        raise ValueError(
            "Target leakage detected: attack_cat appears among features."
        )

    print("\nTraining label distribution:")

    for label, count in target.value_counts().sort_index().items():
        description = (
            "Normal traffic" if label == 0 else "Attack traffic"
        )
        percentage = count / len(target) * 100

        print(
            f"  {label} ({description}): {count:,} records "
            f"({percentage:.2f}%)"
        )

    print(f"\nPredictive input columns: {features.shape[1]:,}")
    print(f"Training records: {len(features):,}")

    return features, target


def create_models():
    """
    Create reproducible supervised classification models.

    Random Forest uses shallow trees and a minimum leaf size to
    help control model complexity.

    Logistic Regression provides a linear baseline for comparison.
    """
    logistic_regression = LogisticRegression(
        solver="liblinear",
        max_iter=300,
        class_weight="balanced",
        random_state=42,
    )

    random_forest = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    return {
        "logistic_regression": logistic_regression,
        "random_forest": random_forest,
    }


def train_and_save_model(
    model_name: str,
    estimator,
    training_matrix,
    target,
):
    """
    Fit one estimator, run a small prediction smoke test,
    and save the trained model locally.
    """
    print("\n" + "=" * 68)
    print(f"TRAINING MODEL: {model_name}")
    print("=" * 68)

    start_time = perf_counter()

    estimator.fit(training_matrix, target)

    elapsed_seconds = perf_counter() - start_time

    print(f"Training time: {elapsed_seconds:.2f} seconds")
    print(f"Training records used: {training_matrix.shape[0]:,}")
    print(f"Features used: {training_matrix.shape[1]:,}")

    # A small smoke test verifies that the fitted estimator can predict.
    # This is NOT an evaluation on an independent test set.
    preview_count = min(5, training_matrix.shape[0])
    predictions = estimator.predict(training_matrix[:preview_count])

    if len(predictions) != preview_count:
        raise RuntimeError(
            f"{model_name} returned an unexpected prediction count."
        )

    if not set(np.asarray(predictions).tolist()).issubset({0, 1}):
        raise RuntimeError(
            f"{model_name} returned unexpected binary class values."
        )

    print(f"Smoke-test predictions: {predictions.tolist()}")

    model_path = MODEL_DIR / f"{model_name}.joblib"

    joblib.dump(
        estimator,
        model_path,
        compress=3,
    )

    print(f"Saved model: {model_path}")

    return estimator


def main() -> int:
    """
    Execute the model training workflow.
    """
    print("=" * 72)
    print("AI-POWERED CYBERSECURITY THREAT DETECTION SYSTEM")
    print("Step 8: Machine-Learning Model Training")
    print("=" * 72)

    try:
        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        features, target = load_training_data()

        print("\nBuilding feature preprocessing pipeline...")

        transformer, numerical_columns, categorical_columns = (
            build_preprocessor(features)
        )

        print(f"Numerical features: {len(numerical_columns):,}")
        print(f"Categorical features: {len(categorical_columns):,}")

        print("\nFitting feature transformations on training data only...")

        training_matrix = transformer.fit_transform(features)

        check_transformed_values(
            training_matrix,
            "Transformed training data",
        )

        if training_matrix.shape[0] != len(target):
            raise ValueError(
                "Transformed training data and target lengths differ."
            )

        if training_matrix.shape[1] == 0:
            raise ValueError(
                "Feature engineering produced zero model features."
            )

        print(
            f"Transformed training shape: {training_matrix.shape}"
        )

        # Save the exact fitted transformer used by both classifiers.
        transformer_path = MODEL_DIR / "preprocessor.joblib"

        joblib.dump(
            transformer,
            transformer_path,
            compress=3,
        )

        print(f"Saved fitted preprocessor: {transformer_path}")

        del transformer
        del features

        models = create_models()

        for model_name, estimator in models.items():
            train_and_save_model(
                model_name=model_name,
                estimator=estimator,
                training_matrix=training_matrix,
                target=target,
            )

        print("\n" + "=" * 72)
        print("SUCCESS: MODEL TRAINING COMPLETED")
        print("=" * 72)

        print("\nLocally saved artifacts:")
        print("  models/preprocessor.joblib")
        print("  models/logistic_regression.joblib")
        print("  models/random_forest.joblib")

        print("\nTraining data was used for fitting.")
        print("Independent test-set evaluation has not been performed.")
        print("Model performance metrics will be calculated in a later step.")

        return 0

    except (OSError, ValueError, TypeError, RuntimeError, MemoryError) as error:
        print("\nERROR: Model training could not be completed.")
        print(error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
