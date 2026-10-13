
"""
Independent testing and evaluation of trained cybersecurity classifiers.

This module:
- Loads the held-out UNSW-NB15 testing dataset.
- Loads the previously trained feature transformer and classifiers.
- Evaluates Logistic Regression and Random Forest.
- Reports accuracy, precision, recall, F1 score, and confusion matrices.
- Saves structured evaluation results and prediction outputs.

The testing dataset is not used to fit or retrain the models.
"""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import matplotlib

# A non-interactive backend supports saving plots in terminal workflows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_loader import TESTING_PATH, load_csv
from src.preprocessing import (
    prepare_features_and_target,
    preprocess_dataset,
)
from src.feature_engineering import check_transformed_values


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"

MODEL_FILES = {
    "logistic_regression": MODEL_DIR / "logistic_regression.joblib",
    "random_forest": MODEL_DIR / "random_forest.joblib",
}

CLASS_NAMES = ["Normal", "Attack"]
CLASS_LABELS = [0, 1]


def load_testing_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and clean the independent testing dataset.

    Returns
    -------
    features:
        Predictive features without target labels or identifiers.
    target:
        Ground-truth binary labels.
    """
    print("\nLoading independent testing dataset...")

    dataframe = load_csv(TESTING_PATH)

    print(f"Testing records: {len(dataframe):,}")

    cleaned = preprocess_dataset(
        dataframe,
        dataset_name="Independent testing dataset",
    )

    del dataframe

    features, target = prepare_features_and_target(cleaned)

    del cleaned

    if len(features) != len(target):
        raise ValueError(
            "The testing feature and target row counts do not match."
        )

    if set(target.unique().tolist()) != {0, 1}:
        raise ValueError(
            "The testing dataset must contain both classes: "
            "0 (normal) and 1 (attack)."
        )

    if "label" in features.columns:
        raise ValueError(
            "Target leakage detected: label appears in testing features."
        )

    if "attack_cat" in features.columns:
        raise ValueError(
            "Target leakage detected: attack_cat appears in features."
        )

    print("\nGround-truth label distribution:")

    for label, count in target.value_counts().sort_index().items():
        description = "Normal traffic" if label == 0 else "Attack traffic"
        percentage = count / len(target) * 100

        print(
            f"  {label} ({description}): {count:,} "
            f"records ({percentage:.2f}%)"
        )

    return features, target


def align_testing_features(
    testing_features: pd.DataFrame,
    fitted_transformer: Any,
) -> pd.DataFrame:
    """
    Align testing columns with the exact feature schema learned at training.

    Raise an error when columns are missing or unexpected, rather than
    silently dropping potentially important information.
    """
    if not hasattr(fitted_transformer, "feature_names_in_"):
        raise ValueError(
            "The saved preprocessor does not expose its training "
            "feature names. Retrain and save it with the current pipeline."
        )

    expected_columns = list(fitted_transformer.feature_names_in_)
    actual_columns = list(testing_features.columns)

    missing_columns = sorted(
        set(expected_columns) - set(actual_columns)
    )

    extra_columns = sorted(
        set(actual_columns) - set(expected_columns)
    )

    if missing_columns or extra_columns:
        raise ValueError(
            "Testing feature columns do not match the training schema. "
            f"Missing columns: {missing_columns}. "
            f"Unexpected columns: {extra_columns}."
        )

    return testing_features.loc[:, expected_columns].copy()


def save_confusion_matrix(
    model_name: str,
    matrix: np.ndarray,
) -> Path:
    """
    Create and save a confusion matrix plot.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )

    figure, axis = plt.subplots(figsize=(7, 6))

    display.plot(
        ax=axis,
        cmap="Blues",
        values_format="d",
        colorbar=False,
    )

    axis.set_title(
        f"UNSW-NB15 Testing Results: {model_name.replace('_', ' ').title()}"
    )

    figure.tight_layout()

    plot_path = OUTPUT_DIR / f"confusion_matrix_{model_name}.png"

    figure.savefig(
        plot_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)

    return plot_path


def evaluate_model(
    model_name: str,
    estimator: Any,
    transformed_testing,
    target: pd.Series,
) -> Tuple[Dict[str, Any], str, np.ndarray, np.ndarray]:
    """
    Evaluate a fitted model on independent testing records.
    """
    print("\n" + "=" * 72)
    print(f"EVALUATING MODEL: {model_name}")
    print("=" * 72)

    predictions = np.asarray(
        estimator.predict(transformed_testing)
    )

    if predictions.ndim != 1:
        raise ValueError(
            f"{model_name} produced predictions with an unexpected shape."
        )

    if len(predictions) != len(target):
        raise ValueError(
            f"{model_name} returned an unexpected prediction count."
        )

    if not set(np.unique(predictions).tolist()).issubset({0, 1}):
        raise ValueError(
            f"{model_name} returned values outside binary classes 0 and 1."
        )

    accuracy = float(
        accuracy_score(target, predictions)
    )

    precision = float(
        precision_score(
            target,
            predictions,
            pos_label=1,
            zero_division=0,
        )
    )

    recall = float(
        recall_score(
            target,
            predictions,
            pos_label=1,
            zero_division=0,
        )
    )

    f1 = float(
        f1_score(
            target,
            predictions,
            pos_label=1,
            zero_division=0,
        )
    )

    balanced_accuracy = float(
        balanced_accuracy_score(target, predictions)
    )

    matrix = confusion_matrix(
        target,
        predictions,
        labels=CLASS_LABELS,
    )

    true_negative, false_positive, false_negative, true_positive = (
        int(value) for value in matrix.ravel()
    )

    normal_count = true_negative + false_positive

    false_positive_rate = (
        false_positive / normal_count
        if normal_count
        else 0.0
    )

    report = classification_report(
        target,
        predictions,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        zero_division=0,
        digits=4,
    )

    metrics = {
        "model": model_name,
        "testing_records": int(len(target)),
        "accuracy": accuracy,
        "precision_attack": precision,
        "recall_attack": recall,
        "f1_score_attack": f1,
        "balanced_accuracy": balanced_accuracy,
        "false_positive_rate": float(false_positive_rate),
        "confusion_matrix": {
            "true_normal": true_negative,
            "false_attack_alerts": false_positive,
            "missed_attacks": false_negative,
            "detected_attacks": true_positive,
        },
    }

    print(f"Testing records: {len(target):,}")
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Precision (attack): {precision:.4f}")
    print(f"Recall (attack):    {recall:.4f}")
    print(f"F1 score (attack):  {f1:.4f}")
    print(f"Balanced accuracy:  {balanced_accuracy:.4f}")
    print(f"False-positive rate:{false_positive_rate: .4f}")

    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print("                 Predicted Normal  Predicted Attack")
    print(
        f"Actual Normal    {true_negative:16,d}  "
        f"{false_positive:16,d}"
    )
    print(
        f"Actual Attack    {false_negative:16,d}  "
        f"{true_positive:16,d}"
    )

    print("\nClassification report:")
    print(report)

    plot_path = save_confusion_matrix(
        model_name=model_name,
        matrix=matrix,
    )

    print(f"Confusion matrix image saved to: {plot_path}")

    return metrics, report, predictions, matrix


def main() -> int:
    """
    Run independent evaluation for each previously trained classifier.
    """
    print("=" * 72)
    print("AI-POWERED CYBERSECURITY THREAT DETECTION SYSTEM")
    print("Step 9: Independent Model Evaluation")
    print("=" * 72)

    if not PREPROCESSOR_PATH.is_file():
        print(f"\nERROR: Missing fitted preprocessor: {PREPROCESSOR_PATH}")
        print("Complete Step 8 model training before running evaluation.")
        return 1

    for model_name, model_path in MODEL_FILES.items():
        if not model_path.is_file():
            print(f"\nERROR: Missing trained model: {model_path}")
            print("Complete Step 8 model training before running evaluation.")
            return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # Load artifacts created locally in Step 8.
        # Never load serialized model files from untrusted sources.
        print("\nLoading previously trained models and transformer...")

        fitted_transformer = joblib.load(PREPROCESSOR_PATH)

        features, target = load_testing_data()

        features = align_testing_features(
            testing_features=features,
            fitted_transformer=fitted_transformer,
        )

        print("\nApplying the saved training transformations...")
        print("No fitting or model retraining is performed on testing data.")

        transformed_testing = fitted_transformer.transform(features)

        check_transformed_values(
            transformed_testing,
            "Transformed testing data",
        )

        if transformed_testing.shape[0] != len(target):
            raise ValueError(
                "Transformed testing row count differs from the target."
            )

        print(f"Transformed testing shape: {transformed_testing.shape}")

        results = {
            "dataset": "UNSW-NB15",
            "evaluation_type": "independent prepared testing split",
            "testing_records": int(len(target)),
            "class_definition": {
                "0": "Normal traffic",
                "1": "Attack traffic",
            },
            "models": {},
        }

        report_sections = []
        prediction_output = pd.DataFrame(
            {
                "actual_label": target.to_numpy(),
            }
        )

        for model_name, model_path in MODEL_FILES.items():
            # These are trusted artifacts generated by this local project.
            estimator = joblib.load(model_path)

            metrics, report, predictions, matrix = evaluate_model(
                model_name=model_name,
                estimator=estimator,
                transformed_testing=transformed_testing,
                target=target,
            )

            results["models"][model_name] = metrics

            report_sections.append(
                "=" * 72
                + f"\n{model_name.upper()}\n"
                + "=" * 72
                + "\n"
                + report
            )

            prediction_output[
                f"{model_name}_prediction"
            ] = predictions.astype(int)

        # Save machine-readable metrics for later comparisons and reporting.
        metrics_path = OUTPUT_DIR / "evaluation_results.json"

        with metrics_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                results,
                file,
                indent=4,
            )

        # Save readable classification reports.
        report_path = OUTPUT_DIR / "classification_report.txt"

        report_path.write_text(
            "\n\n".join(report_sections),
            encoding="utf-8",
        )

        # Save row-level predictions for the later alert-generation module.
        prediction_path = OUTPUT_DIR / "predictions.csv"

        prediction_output.to_csv(
            prediction_path,
            index=False,
        )

        comparison_rows = []

        for model_name, metrics in results["models"].items():
            comparison_rows.append(
                {
                    "model": model_name,
                    "accuracy": metrics["accuracy"],
                    "precision_attack": metrics["precision_attack"],
                    "recall_attack": metrics["recall_attack"],
                    "f1_score_attack": metrics["f1_score_attack"],
                    "false_positive_rate": metrics["false_positive_rate"],
                }
            )

        comparison = pd.DataFrame(comparison_rows)

        comparison_path = OUTPUT_DIR / "model_comparison.csv"

        comparison.to_csv(
            comparison_path,
            index=False,
        )

        print("\n" + "=" * 72)
        print("MODEL COMPARISON")
        print("=" * 72)
        print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        print("\nEvaluation files saved:")
        print(f"  Metrics: {metrics_path}")
        print(f"  Classification reports: {report_path}")
        print(f"  Predictions: {prediction_path}")
        print(f"  Model comparison: {comparison_path}")

        print("\nSUCCESS: Independent model evaluation completed.")
        print("The testing data was not used for training or fitting transforms.")

        return 0

    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        MemoryError,
    ) as error:
        print("\nERROR: Model evaluation could not be completed.")
        print(error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
