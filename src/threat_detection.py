
"""
Offline cybersecurity threat detection and alert generation.

This module:
1. Loads the fitted preprocessing transformer and Random Forest model.
2. Reads network traffic records for offline dataset replay.
3. Removes labels and metadata from predictive input features.
4. Applies the same transformations used during model training.
5. Computes model-generated attack scores.
6. Assigns alert severity using explicit score thresholds.
7. Saves detailed detections, alerts, and a JSON summary.

This is a defensive simulation, not a live network monitoring system.
The model score is not a calibrated probability of a real compromise.
No traffic is blocked and no automated response is performed.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_loader import TESTING_PATH, load_csv
from src.preprocessing import MISSING_TEXT_VALUES
from src.feature_engineering import check_transformed_values


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
RANDOM_FOREST_PATH = MODEL_DIR / "random_forest.joblib"

# Operational thresholds for this demonstration.
# These are project heuristics, not calibrated real-world risk probabilities.
DETECTION_THRESHOLD = 0.50
MEDIUM_SEVERITY_THRESHOLD = 0.65
HIGH_SEVERITY_THRESHOLD = 0.85

# Only these fields are copied into the detection reports as metadata.
# None of these fields is used to determine the model's prediction.
METADATA_COLUMNS = {
    "proto": "protocol",
    "service": "service",
    "state": "connection_state",
    "dur": "duration",
    "spkts": "source_packets",
    "dpkts": "destination_packets",
    "sbytes": "source_bytes",
    "dbytes": "destination_bytes",
}


def prepare_detection_features(
    raw_data: pd.DataFrame,
    fitted_transformer,
) -> pd.DataFrame:
    """
    Prepare model input for labelled or unlabelled network traffic.

    Identifiers, attack-category descriptions, and target labels are
    explicitly excluded from the model input.

    The function does not need ground-truth labels to make predictions.
    """
    if not isinstance(raw_data, pd.DataFrame):
        raise TypeError("Expected network traffic data as a DataFrame.")

    if raw_data.empty:
        raise ValueError("The network traffic dataset is empty.")

    data = raw_data.copy(deep=True)

    data.columns = [
        str(column).strip() for column in data.columns
    ]

    if data.columns.duplicated().any():
        raise ValueError(
            "The input dataset contains duplicate column names."
        )

    # Remove all non-predictive identifiers and targets.
    # This permits prediction on new records without a target label.
    excluded_columns = [
        "id",
        "label",
        "attack_cat",
    ]

    features = data.drop(
        columns=excluded_columns,
        errors="ignore",
    ).copy()

    # Apply the same basic missing-value conventions used in training.
    text_columns = features.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    for column in text_columns:
        normalized = (
            features[column]
            .astype("string")
            .str.strip()
        )

        missing_mask = (
            normalized.str.casefold()
            .isin(MISSING_TEXT_VALUES)
            .fillna(False)
        )

        normalized = normalized.mask(
            missing_mask,
            pd.NA,
        )

        normalized = normalized.astype(object)
        normalized[pd.isna(normalized)] = np.nan

        features[column] = normalized

    # Convert invalid infinite numeric values to missing values.
    numeric_columns = features.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    for column in numeric_columns:
        features[column] = features[column].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    # Enforce the same exact feature schema used during training.
    if not hasattr(fitted_transformer, "feature_names_in_"):
        raise ValueError(
            "The fitted transformer does not contain the expected "
            "training feature names. Retrain the pipeline from Step 8."
        )

    expected_columns = list(
        fitted_transformer.feature_names_in_
    )

    actual_columns = list(features.columns)

    missing_columns = sorted(
        set(expected_columns) - set(actual_columns)
    )

    extra_columns = sorted(
        set(actual_columns) - set(expected_columns)
    )

    if missing_columns or extra_columns:
        raise ValueError(
            "The input feature schema does not match the trained model. "
            f"Missing columns: {missing_columns}. "
            f"Unexpected columns: {extra_columns}."
        )

    return features.loc[:, expected_columns].copy()


def get_attack_scores(
    model,
    transformed_features,
) -> np.ndarray:
    """
    Obtain the model's class-1 scores.

    Scikit-learn can order class probabilities according to classes_.
    We therefore locate class 1 explicitly rather than assuming the
    attack class always occupies column 1.
    """
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "The selected classifier does not support predict_proba."
        )

    if not hasattr(model, "classes_"):
        raise ValueError(
            "The selected model has not been fitted."
        )

    classes = np.asarray(model.classes_)
    attack_positions = np.flatnonzero(classes == 1)

    if len(attack_positions) != 1:
        raise ValueError(
            "The trained model does not have a unique attack class 1."
        )

    probabilities = np.asarray(
        model.predict_proba(transformed_features),
        dtype=float,
    )

    if probabilities.ndim != 2:
        raise ValueError(
            "The classifier returned an unexpected score matrix."
        )

    if probabilities.shape[0] != transformed_features.shape[0]:
        raise ValueError(
            "The number of scores does not match the number of records."
        )

    scores = probabilities[:, int(attack_positions[0])]

    if not np.isfinite(scores).all():
        raise ValueError(
            "The model produced missing or infinite attack scores."
        )

    if ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError(
            "The model produced scores outside the range from 0 to 1."
        )

    return scores


def severity_for_score(
    score: float,
    predicted_attack: bool,
) -> str:
    """
    Assign a demonstration alert severity.

    Records below the detection threshold do not generate an alert.
    """
    if not predicted_attack:
        return "NONE"

    if score >= HIGH_SEVERITY_THRESHOLD:
        return "HIGH"

    if score >= MEDIUM_SEVERITY_THRESHOLD:
        return "MEDIUM"

    return "LOW"


def recommended_action(severity: str) -> str:
    """
    Return a defensive triage recommendation.

    Recommendations are informational. They never block or change traffic.
    """
    recommendations = {
        "HIGH": (
            "Prioritize analyst review and corroborate with independent "
            "network or endpoint logs."
        ),
        "MEDIUM": (
            "Review related network activity and compare with other "
            "available security telemetry."
        ),
        "LOW": (
            "Queue for review and compare the event with expected "
            "network behavior."
        ),
        "NONE": (
            "No model-generated alert; continue routine monitoring."
        ),
    }

    return recommendations[severity]


def build_detection_results(
    raw_data: pd.DataFrame,
    attack_scores: np.ndarray,
) -> pd.DataFrame:
    """
    Create a row-level detection report without exposing ground-truth
    labels or attack categories.

    A model score at or above the configured detection threshold is
    classified as an attack prediction for this simulation.
    """
    if len(raw_data) == 0:
        raise ValueError("Cannot create detection results for empty data.")

    scores = np.asarray(attack_scores, dtype=float)

    if scores.ndim != 1:
        raise ValueError("Attack scores must be a one-dimensional array.")

    if len(scores) != len(raw_data):
        raise ValueError(
            "The number of model scores does not match input records."
        )

    if not np.isfinite(scores).all():
        raise ValueError("Attack scores contain invalid numeric values.")

    if ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("Attack scores must be between 0 and 1.")

    source = raw_data.reset_index(drop=True)
    predicted_attack = scores >= DETECTION_THRESHOLD

    results = pd.DataFrame(
        {
            "event_index": np.arange(1, len(source) + 1),
        }
    )

    if "id" in source.columns:
        results["source_record_id"] = source["id"].to_numpy()
    else:
        results["source_record_id"] = results["event_index"]

    # Attach selected traffic metadata for defensive triage.
    # Metadata never changes the predicted class or score.
    for source_column, output_column in METADATA_COLUMNS.items():
        if source_column in source.columns:
            results[output_column] = source[
                source_column
            ].to_numpy()

    results["attack_score"] = np.round(scores, 6)

    results["predicted_label"] = predicted_attack.astype(int)

    results["predicted_class"] = np.where(
        predicted_attack,
        "ATTACK",
        "NORMAL",
    )

    results["severity"] = [
        severity_for_score(
            float(score),
            bool(is_attack),
        )
        for score, is_attack in zip(scores, predicted_attack)
    ]

    results["alert_reason"] = [
        (
            f"Model attack score {score:.3f} met or exceeded "
            f"the configured threshold of {DETECTION_THRESHOLD:.2f}."
        )
        if is_attack
        else "Model score was below the configured alert threshold."
        for score, is_attack in zip(scores, predicted_attack)
    ]

    results["recommended_action"] = [
        recommended_action(severity)
        for severity in results["severity"]
    ]

    return results


def save_detection_outputs(
    detection_results: pd.DataFrame,
) -> dict:
    """
    Write all detection results, generated alerts, and the summary.
    """
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = OUTPUT_DIR / "detection_results.csv"
    alerts_path = OUTPUT_DIR / "threat_alerts.csv"
    summary_path = OUTPUT_DIR / "detection_summary.json"

    detection_results.to_csv(
        results_path,
        index=False,
    )

    alerts = detection_results.loc[
        detection_results["predicted_label"] == 1
    ].copy()

    alerts = alerts.sort_values(
        by="attack_score",
        ascending=False,
    )

    alerts.to_csv(
        alerts_path,
        index=False,
    )

    severity_counts = {
        severity: int(
            (alerts["severity"] == severity).sum()
        )
        for severity in ("HIGH", "MEDIUM", "LOW")
    }

    normal_count = int(
        (detection_results["predicted_label"] == 0).sum()
    )

    alert_count = len(alerts)

    summary = {
        "project": "AI-Powered Cybersecurity Threat Detection System",
        "mode": "offline dataset replay",
        "source_dataset": "UNSW-NB15 prepared testing split",
        "model": "Random Forest",
        "records_processed": int(len(detection_results)),
        "predicted_normal_records": normal_count,
        "predicted_attack_records": int(alert_count),
        "alerts_generated": int(alert_count),
        "alerts_by_severity": severity_counts,
        "attack_score_threshold": DETECTION_THRESHOLD,
        "medium_severity_threshold": MEDIUM_SEVERITY_THRESHOLD,
        "high_severity_threshold": HIGH_SEVERITY_THRESHOLD,
        "ground_truth_labels_used_for_predictions": False,
        "score_interpretation": (
            "The score is the model's class-1 output. It is not a "
            "calibrated probability of a real compromise."
        ),
        "automated_response": False,
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    return {
        "results_path": results_path,
        "alerts_path": alerts_path,
        "summary_path": summary_path,
        "summary": summary,
        "alerts": alerts,
    }


def print_detection_summary(
    summary: dict,
    alerts: pd.DataFrame,
    results_path: Path,
    alerts_path: Path,
    summary_path: Path,
) -> None:
    """Print a compact SOC-style detection report."""
    print("\n" + "=" * 72)
    print("OFFLINE CYBERSECURITY DETECTION SUMMARY")
    print("=" * 72)

    print(f"Records processed: {summary['records_processed']:,}")
    print(
        f"Predicted normal records: "
        f"{summary['predicted_normal_records']:,}"
    )
    print(
        f"Predicted attack records: "
        f"{summary['predicted_attack_records']:,}"
    )
    print(f"Alerts generated: {summary['alerts_generated']:,}")

    print("\nAlert thresholds:")
    print(f"  Detection: {DETECTION_THRESHOLD:.2f} and above")
    print(f"  Medium:    {MEDIUM_SEVERITY_THRESHOLD:.2f} and above")
    print(f"  High:      {HIGH_SEVERITY_THRESHOLD:.2f} and above")

    print("\nAlerts by severity:")

    for severity in ("HIGH", "MEDIUM", "LOW"):
        print(
            f"  {severity}: "
            f"{summary['alerts_by_severity'][severity]:,}"
        )

    if not alerts.empty:
        print("\nHighest-scoring alerts (first 10):")

        preview_columns = [
            column
            for column in (
                "event_index",
                "source_record_id",
                "protocol",
                "service",
                "attack_score",
                "severity",
            )
            if column in alerts.columns
        ]

        print(
            alerts[preview_columns]
            .head(10)
            .to_string(index=False)
        )
    else:
        print("\nNo records met the configured detection threshold.")

    print("\nSaved output files:")
    print(f"  All detections: {results_path}")
    print(f"  Threat alerts:  {alerts_path}")
    print(f"  JSON summary:   {summary_path}")

    print("\nImportant:")
    print("  This is an offline dataset simulation.")
    print("  Ground-truth labels were not used to make predictions.")
    print("  No live traffic was monitored or blocked.")
    print("  Model scores are not calibrated compromise probabilities.")

    print("\nSUCCESS: Offline threat detection completed.")
    print("=" * 72)


def run_self_test() -> int:
    """
    Test alert thresholds and metadata handling without loading the
    trained models or the real dataset.
    """
    sample = pd.DataFrame(
        {
            "id": [101, 102, 103],
            "proto": ["tcp", "udp", "icmp"],
            "dur": [0.1, 1.5, 0.3],
            "label": [0, 1, 1],
            "attack_cat": ["Normal", "DoS", "Exploits"],
        }
    )

    scores = np.array([0.40, 0.70, 0.92])

    try:
        results = build_detection_results(
            raw_data=sample,
            attack_scores=scores,
        )

        assert results["predicted_label"].tolist() == [0, 1, 1]

        assert results["severity"].tolist() == [
            "NONE",
            "MEDIUM",
            "HIGH",
        ]

        assert len(results) == 3

        assert (
            results["source_record_id"].tolist()
            == [101, 102, 103]
        )

        # Ground-truth labels and categories must not be copied to
        # the model-generated detection output.
        assert "label" not in results.columns
        assert "attack_cat" not in results.columns

        assert severity_for_score(0.50, True) == "LOW"
        assert severity_for_score(0.65, True) == "MEDIUM"
        assert severity_for_score(0.85, True) == "HIGH"
        assert severity_for_score(0.99, False) == "NONE"

        assert recommended_action("HIGH")
        assert recommended_action("MEDIUM")
        assert recommended_action("LOW")
        assert recommended_action("NONE")

        print("Synthetic event count:", len(results))
        print(
            "Synthetic alerts:",
            int(results["predicted_label"].sum()),
        )
        print(
            "Synthetic severity results:",
            results["severity"].tolist(),
        )
        print("\nSUCCESS: Threat detection self-test passed.")
        print("No real dataset or saved model was required.")

        return 0

    except (AssertionError, TypeError, ValueError) as error:
        print(f"\nERROR: Threat detection self-test failed: {error}")
        return 1


def main() -> int:
    """Run the saved Random Forest model on the prepared testing dataset."""
    print("=" * 72)
    print("AI-POWERED CYBERSECURITY THREAT DETECTION SYSTEM")
    print("Step 10: Threat Detection and Alert Generation")
    print("=" * 72)

    required_artifacts = [
        PREPROCESSOR_PATH,
        RANDOM_FOREST_PATH,
    ]

    for artifact in required_artifacts:
        if not artifact.is_file():
            print(f"\nERROR: Required saved artifact not found: {artifact}")
            print("Complete Step 8 before running threat detection.")
            return 1

    try:
        # Load only trusted model artifacts generated by this project.
        transformer = joblib.load(PREPROCESSOR_PATH)
        model = joblib.load(RANDOM_FOREST_PATH)

        print(f"\nInput dataset: {TESTING_PATH}")
        raw_data = load_csv(TESTING_PATH)

        print(f"Loaded records: {len(raw_data):,}")

        # Label fields, if present in the source CSV, are removed before
        # transforming the data or generating predictions.
        features = prepare_detection_features(
            raw_data=raw_data,
            fitted_transformer=transformer,
        )

        print("\nApplying saved training transformations...")

        transformed_features = transformer.transform(features)

        check_transformed_values(
            transformed_features,
            "Threat detection input",
        )

        print(
            f"Transformed feature shape: "
            f"{transformed_features.shape}"
        )

        print("\nScoring network traffic with Random Forest...")

        attack_scores = get_attack_scores(
            model=model,
            transformed_features=transformed_features,
        )

        detection_results = build_detection_results(
            raw_data=raw_data,
            attack_scores=attack_scores,
        )

        output_info = save_detection_outputs(
            detection_results=detection_results,
        )

        print_detection_summary(
            summary=output_info["summary"],
            alerts=output_info["alerts"],
            results_path=output_info["results_path"],
            alerts_path=output_info["alerts_path"],
            summary_path=output_info["summary_path"],
        )

        return 0

    except (
        OSError,
        ValueError,
        TypeError,
        RuntimeError,
        MemoryError,
    ) as error:
        print("\nERROR: Threat detection could not be completed.")
        print(error)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run the offline cybersecurity threat detection pipeline."
        )
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Test alert logic using small synthetic records.",
    )

    arguments = parser.parse_args()

    if arguments.self_test:
        raise SystemExit(run_self_test())

    raise SystemExit(main())
