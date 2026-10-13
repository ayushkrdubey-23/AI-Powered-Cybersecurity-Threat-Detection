
"""
Reproducible offline cyber-threat simulation.

This module selects a balanced sample from the prepared UNSW-NB15
testing dataset, applies the already trained Random Forest model,
and compares model predictions with the dataset's ground-truth labels.

Important safeguards:
- No model is trained in this module.
- The test dataset is not used to fit the transformer.
- Ground-truth labels are removed before model prediction.
- Labels are attached to the output only after predictions are created.
- Results from the balanced demonstration sample are not representative
  of real-world attack prevalence.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_loader import TESTING_PATH, load_csv
from src.feature_engineering import check_transformed_values
from src.threat_detection import (
    PREPROCESSOR_PATH,
    RANDOM_FOREST_PATH,
    build_detection_results,
    get_attack_scores,
    prepare_detection_features,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

SIMULATION_RESULTS_PATH = OUTPUT_DIR / "simulation_results.csv"
SIMULATION_ALERTS_PATH = OUTPUT_DIR / "simulation_alerts.csv"
SIMULATION_SUMMARY_PATH = OUTPUT_DIR / "simulation_summary.json"

NORMAL_RANDOM_STATE = 42
ATTACK_RANDOM_STATE = 43
SHUFFLE_RANDOM_STATE = 44


def select_balanced_sample(
    dataframe: pd.DataFrame,
    samples_per_class: int = 50,
) -> pd.DataFrame:
    """
    Select up to the requested number of records from each class.

    A fixed random seed makes selection reproducible.

    The label is used only for demonstration-sample selection here.
    The prediction pipeline removes it before transforming model inputs.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Expected a Pandas DataFrame.")

    if dataframe.empty:
        raise ValueError("The testing dataset is empty.")

    if samples_per_class < 1:
        raise ValueError("samples_per_class must be at least 1.")

    if "label" not in dataframe.columns:
        raise ValueError(
            "The testing dataset must contain the binary 'label' column."
        )

    labels = pd.to_numeric(
        dataframe["label"],
        errors="coerce",
    )

    if labels.isna().any():
        raise ValueError(
            "The testing dataset contains missing or non-numeric labels."
        )

    if not labels.isin([0, 1]).all():
        raise ValueError(
            "The testing dataset must use binary labels 0 and 1."
        )

    normalized = dataframe.copy(deep=True)
    normalized["label"] = labels.astype(int)

    normal_records = normalized.loc[
        normalized["label"] == 0
    ]

    attack_records = normalized.loc[
        normalized["label"] == 1
    ]

    if normal_records.empty or attack_records.empty:
        raise ValueError(
            "Both normal and attack records are required for the "
            "demonstration simulation."
        )

    normal_count = min(
        samples_per_class,
        len(normal_records),
    )

    attack_count = min(
        samples_per_class,
        len(attack_records),
    )

    selected_normal = normal_records.sample(
        n=normal_count,
        random_state=NORMAL_RANDOM_STATE,
    )

    selected_attacks = attack_records.sample(
        n=attack_count,
        random_state=ATTACK_RANDOM_STATE,
    )

    selected = pd.concat(
        [selected_normal, selected_attacks],
        axis=0,
    )

    selected = selected.sample(
        frac=1.0,
        random_state=SHUFFLE_RANDOM_STATE,
    ).reset_index(drop=True)

    return selected


def calculate_simulation_metrics(
    actual_labels: pd.Series,
    predicted_labels: pd.Series,
) -> dict:
    """
    Calculate binary classification metrics for the demonstration sample.

    These metrics describe only the selected sample, not real-world
    operational performance or attack prevalence.
    """
    actual = np.asarray(actual_labels, dtype=int)
    predicted = np.asarray(predicted_labels, dtype=int)

    if len(actual) != len(predicted):
        raise ValueError(
            "Actual and predicted label counts do not match."
        )

    if len(actual) == 0:
        raise ValueError("Cannot evaluate an empty simulation sample.")

    if not set(np.unique(actual).tolist()).issubset({0, 1}):
        raise ValueError("Actual labels must be binary.")

    if not set(np.unique(predicted).tolist()).issubset({0, 1}):
        raise ValueError("Predicted labels must be binary.")

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    )

    true_normal, false_alert, missed_attack, detected_attack = (
        int(value) for value in matrix.ravel()
    )

    total_normal = true_normal + false_alert

    false_positive_rate = (
        false_alert / total_normal
        if total_normal
        else 0.0
    )

    return {
        "sample_records": int(len(actual)),
        "accuracy": float(
            accuracy_score(actual, predicted)
        ),
        "precision_attack": float(
            precision_score(
                actual,
                predicted,
                pos_label=1,
                zero_division=0,
            )
        ),
        "recall_attack": float(
            recall_score(
                actual,
                predicted,
                pos_label=1,
                zero_division=0,
            )
        ),
        "f1_score_attack": float(
            f1_score(
                actual,
                predicted,
                pos_label=1,
                zero_division=0,
            )
        ),
        "false_positive_rate": float(false_positive_rate),
        "confusion_matrix": {
            "true_normal": true_normal,
            "false_attack_alerts": false_alert,
            "missed_attacks": missed_attack,
            "detected_attacks": detected_attack,
        },
    }


def run_simulation(
    samples_per_class: int = 50,
) -> dict:
    """
    Run the end-to-end offline simulation using saved model artifacts.
    """
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be at least 1.")

    missing_artifacts = [
        path
        for path in (
            PREPROCESSOR_PATH,
            RANDOM_FOREST_PATH,
        )
        if not path.is_file()
    ]

    if missing_artifacts:
        missing_text = "\n".join(
            f"  - {path}" for path in missing_artifacts
        )

        raise FileNotFoundError(
            "Required trained model artifacts are missing:\n"
            f"{missing_text}\n"
            "Complete model training before running the simulation."
        )

    print("=" * 72)
    print("AI-POWERED CYBERSECURITY THREAT DETECTION SYSTEM")
    print("Step 15: Reproducible Offline Threat Simulation")
    print("=" * 72)

    print(f"\nReading testing dataset: {TESTING_PATH}")

    testing_data = load_csv(TESTING_PATH)

    print(f"Available testing records: {len(testing_data):,}")

    selected_data = select_balanced_sample(
        dataframe=testing_data,
        samples_per_class=samples_per_class,
    )

    actual_labels = selected_data["label"].astype(int).reset_index(
        drop=True
    )

    if "attack_cat" in selected_data.columns:
        actual_categories = (
            selected_data["attack_cat"]
            .fillna("Unknown")
            .astype(str)
            .reset_index(drop=True)
        )
    else:
        actual_categories = pd.Series(
            ["Unknown"] * len(selected_data),
            name="attack_cat",
        )

    print("\nSelected simulation sample:")
    print(f"  Total records: {len(selected_data):,}")
    print(
        f"  Normal ground-truth records: "
        f"{int((actual_labels == 0).sum()):,}"
    )
    print(
        f"  Attack ground-truth records: "
        f"{int((actual_labels == 1).sum()):,}"
    )

    print("\nLoading the previously trained model artifacts...")

    # Only load artifacts generated by this local, trusted project.
    transformer = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(RANDOM_FOREST_PATH)

    # Ground-truth fields are explicitly removed by this function.
    # The prediction path receives only the expected network features.
    model_features = prepare_detection_features(
        raw_data=selected_data,
        fitted_transformer=transformer,
    )

    if "label" in model_features.columns:
        raise ValueError(
            "Target leakage prevention failed: label is present in "
            "the simulation model inputs."
        )

    if "attack_cat" in model_features.columns:
        raise ValueError(
            "Target leakage prevention failed: attack_cat is present "
            "in the simulation model inputs."
        )

    print("\nTransforming the simulation inputs...")

    transformed = transformer.transform(model_features)

    check_transformed_values(
        transformed,
        "Simulation features",
    )

    if transformed.shape[0] != len(actual_labels):
        raise ValueError(
            "Transformed feature count does not match selected records."
        )

    print(f"Transformed simulation shape: {transformed.shape}")

    print("\nGenerating model predictions...")

    scores = get_attack_scores(
        model=model,
        transformed_features=transformed,
    )

    # Predictions are created before ground-truth labels are attached
    # to the reporting DataFrame.
    detections = build_detection_results(
        raw_data=selected_data,
        attack_scores=scores,
    )

    predicted_labels = detections[
        "predicted_label"
    ].astype(int).reset_index(drop=True)

    # Ground truth is now joined solely for post-prediction assessment.
    detections["actual_label"] = actual_labels.to_numpy()

    detections["actual_class"] = np.where(
        actual_labels.to_numpy() == 1,
        "ATTACK",
        "NORMAL",
    )

    detections["actual_attack_category"] = (
        actual_categories.to_numpy()
    )

    detections["prediction_correct"] = (
        detections["actual_label"]
        == detections["predicted_label"]
    )

    metrics = calculate_simulation_metrics(
        actual_labels=actual_labels,
        predicted_labels=predicted_labels,
    )

    severity_counts = (
        detections.loc[
            detections["predicted_label"] == 1,
            "severity",
        ]
        .value_counts()
        .to_dict()
    )

    categories = (
        actual_categories.loc[actual_labels == 1]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    summary = {
        "project": "AI-Powered Cybersecurity Threat Detection System",
        "simulation_mode": "offline replay of a balanced public test sample",
        "source_dataset": "UNSW-NB15",
        "model": "Random Forest",
        "random_state": SHUFFLE_RANDOM_STATE,
        "samples_requested_per_class": int(samples_per_class),
        "selected_records": int(len(detections)),
        "ground_truth_normal_records": int(
            (actual_labels == 0).sum()
        ),
        "ground_truth_attack_records": int(
            (actual_labels == 1).sum()
        ),
        "predicted_normal_records": int(
            (predicted_labels == 0).sum()
        ),
        "predicted_attack_records": int(
            (predicted_labels == 1).sum()
        ),
        "predictions_matching_ground_truth": int(
            detections["prediction_correct"].sum()
        ),
        "predicted_alerts_by_severity": {
            str(key): int(value)
            for key, value in severity_counts.items()
        },
        "ground_truth_attack_categories": {
            str(key): int(value)
            for key, value in categories.items()
        },
        "metrics": metrics,
        "ground_truth_labels_used_as_model_features": False,
        "ground_truth_used_for_sample_selection": True,
        "ground_truth_joined_after_prediction": True,
        "sample_is_balanced": True,
        "representative_of_real_world_attack_prevalence": False,
        "live_network_access": False,
        "automated_response": False,
        "interpretation": (
            "This simulation demonstrates the workflow on a reproducible "
            "balanced sample. Metrics are specific to this sample and "
            "must not be interpreted as real-world operational performance."
        ),
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    alerts = detections.loc[
        detections["predicted_label"] == 1
    ].sort_values(
        by="attack_score",
        ascending=False,
    )

    detections.to_csv(
        SIMULATION_RESULTS_PATH,
        index=False,
    )

    alerts.to_csv(
        SIMULATION_ALERTS_PATH,
        index=False,
    )

    with SIMULATION_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    print("\n" + "=" * 72)
    print("SIMULATION RESULTS")
    print("=" * 72)

    print(f"Records processed: {metrics['sample_records']:,}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Attack precision: {metrics['precision_attack']:.4f}")
    print(f"Attack recall: {metrics['recall_attack']:.4f}")
    print(f"Attack F1 score: {metrics['f1_score_attack']:.4f}")
    print(f"False-positive rate: {metrics['false_positive_rate']:.4f}")

    print("\nConfusion matrix counts:")
    for key, value in metrics["confusion_matrix"].items():
        print(f"  {key}: {value:,}")

    print("\nTop alert records:")

    preview_columns = [
        column
        for column in (
            "event_index",
            "protocol",
            "actual_class",
            "actual_attack_category",
            "predicted_class",
            "attack_score",
            "severity",
            "prediction_correct",
        )
        if column in alerts.columns
    ]

    if alerts.empty:
        print("  No records met the detection threshold.")
    else:
        print(
            alerts[preview_columns]
            .head(10)
            .to_string(index=False)
        )

    print("\nGenerated files:")
    print(f"  {SIMULATION_RESULTS_PATH}")
    print(f"  {SIMULATION_ALERTS_PATH}")
    print(f"  {SIMULATION_SUMMARY_PATH}")

    print("\nSUCCESS: Offline threat simulation completed.")
    print("No live network traffic was captured or changed.")
    print("No model was trained or refitted in this step.")
    print("=" * 72)

    return summary


def main() -> int:
    """Parse command-line options and execute the simulation."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a reproducible offline cybersecurity threat simulation."
        )
    )

    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=50,
        help=(
            "Maximum number of normal and attack records to select "
            "from each class. Default: 50."
        ),
    )

    args = parser.parse_args()

    try:
        run_simulation(
            samples_per_class=args.samples_per_class,
        )
        return 0

    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        MemoryError,
    ) as error:
        print("\nERROR: Offline threat simulation could not be completed.")
        print(error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
