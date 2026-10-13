
"""
Visualization module for the AI-Powered Cybersecurity Threat Detection System.

Generates plots from actual model evaluation and offline threat detection
results. No model scores or alert counts are fabricated.

Generated visualizations:
1. Model performance comparison
2. Threat alert severity distribution
3. Attack-score distribution
4. Protocol distribution among predicted alerts
5. Random Forest feature importance
"""

from pathlib import Path

import joblib
import matplotlib

# Allow plots to be generated without opening interactive windows.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_COMPARISON_PATH = OUTPUT_DIR / "model_comparison.csv"
DETECTION_RESULTS_PATH = OUTPUT_DIR / "detection_results.csv"

PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
RANDOM_FOREST_PATH = MODEL_DIR / "random_forest.joblib"

SEVERITY_ORDER = ["HIGH", "MEDIUM", "LOW"]


def save_figure(figure, filename: str) -> Path:
    """Save a figure to the project's visualization output directory."""
    VISUALIZATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = VISUALIZATION_DIR / filename

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(f"Saved visualization: {output_path}")

    return output_path


def load_and_validate_results():
    """
    Load the actual model comparison and threat detection outputs.

    The required files are generated in Steps 9 and 10.
    """
    required_files = [
        MODEL_COMPARISON_PATH,
        DETECTION_RESULTS_PATH,
        PREPROCESSOR_PATH,
        RANDOM_FOREST_PATH,
    ]

    missing_files = [
        path
        for path in required_files
        if not path.is_file()
    ]

    if missing_files:
        missing_text = "\n".join(
            f"  - {path}" for path in missing_files
        )

        raise FileNotFoundError(
            "Required evaluation, detection, or model files are missing:\n"
            f"{missing_text}\n\n"
            "Complete the relevant earlier steps before generating "
            "visualizations."
        )

    model_comparison = pd.read_csv(MODEL_COMPARISON_PATH)
    detections = pd.read_csv(DETECTION_RESULTS_PATH)

    if model_comparison.empty:
        raise ValueError("The model comparison file contains no results.")

    if detections.empty:
        raise ValueError("The detection results file contains no records.")

    required_metrics = {
        "model",
        "accuracy",
        "precision_attack",
        "recall_attack",
        "f1_score_attack",
        "false_positive_rate",
    }

    missing_metrics = required_metrics - set(model_comparison.columns)

    if missing_metrics:
        raise ValueError(
            "Model comparison file is missing required columns: "
            f"{sorted(missing_metrics)}"
        )

    required_detection_columns = {
        "attack_score",
        "predicted_label",
        "predicted_class",
        "severity",
    }

    missing_detection_columns = (
        required_detection_columns - set(detections.columns)
    )

    if missing_detection_columns:
        raise ValueError(
            "Detection results are missing required columns: "
            f"{sorted(missing_detection_columns)}"
        )

    # Validate score fields rather than silently drawing misleading plots.
    for column in (
        "accuracy",
        "precision_attack",
        "recall_attack",
        "f1_score_attack",
        "false_positive_rate",
    ):
        model_comparison[column] = pd.to_numeric(
            model_comparison[column],
            errors="coerce",
        )

        if (
            model_comparison[column].isna().any()
            or not np.isfinite(model_comparison[column]).all()
        ):
            raise ValueError(
                f"The model comparison column '{column}' contains "
                "invalid metric values."
            )

        if (
            (model_comparison[column] < 0).any()
            or (model_comparison[column] > 1).any()
        ):
            raise ValueError(
                f"The model comparison column '{column}' must be "
                "between 0 and 1."
            )

    detections["attack_score"] = pd.to_numeric(
        detections["attack_score"],
        errors="coerce",
    )

    if (
        detections["attack_score"].isna().any()
        or not np.isfinite(detections["attack_score"]).all()
    ):
        raise ValueError(
            "Detection results contain invalid attack scores."
        )

    if (
        (detections["attack_score"] < 0).any()
        or (detections["attack_score"] > 1).any()
    ):
        raise ValueError(
            "Detection attack scores must be between 0 and 1."
        )

    detections["predicted_label"] = pd.to_numeric(
        detections["predicted_label"],
        errors="raise",
    ).astype(int)

    if not detections["predicted_label"].isin([0, 1]).all():
        raise ValueError(
            "Detection results contain unsupported predicted labels."
        )

    return model_comparison, detections


def plot_model_performance(
    model_comparison: pd.DataFrame,
) -> Path:
    """Compare classification performance for each evaluated model."""
    metric_columns = [
        "accuracy",
        "precision_attack",
        "recall_attack",
        "f1_score_attack",
    ]

    values = model_comparison.set_index("model")[metric_columns]

    figure, axis = plt.subplots(figsize=(10, 6))

    values.plot(
        kind="bar",
        ax=axis,
        width=0.78,
    )

    axis.set_title("Cybersecurity Model Performance Comparison")
    axis.set_xlabel("Machine-learning model")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1.05)
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    axis.legend(
        title="Metric",
        loc="best",
    )

    axis.set_xticklabels(
        axis.get_xticklabels(),
        rotation=15,
        ha="right",
    )

    return save_figure(
        figure,
        "model_metrics_comparison.png",
    )


def plot_alert_severity(
    detections: pd.DataFrame,
) -> Path:
    """Visualize the number of generated alerts at each severity."""
    alerts = detections.loc[
        detections["predicted_label"] == 1
    ]

    severity_counts = (
        alerts["severity"]
        .value_counts()
        .reindex(SEVERITY_ORDER, fill_value=0)
    )

    figure, axis = plt.subplots(figsize=(8, 5))

    bars = axis.bar(
        severity_counts.index,
        severity_counts.values,
    )

    axis.set_title("Generated Cybersecurity Alerts by Severity")
    axis.set_xlabel("Alert severity")
    axis.set_ylabel("Number of alerts")
    axis.grid(axis="y", linestyle="--", alpha=0.3)

    axis.bar_label(
        bars,
        fmt="%d",
        padding=3,
    )

    if alerts.empty:
        axis.text(
            0.5,
            0.5,
            "No records met the detection threshold",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )

    return save_figure(
        figure,
        "alert_severity_distribution.png",
    )


def plot_attack_score_distribution(
    detections: pd.DataFrame,
) -> Path:
    """Compare attack scores assigned to predicted normal and attack records."""
    predicted_normal = detections.loc[
        detections["predicted_label"] == 0,
        "attack_score",
    ]

    predicted_attack = detections.loc[
        detections["predicted_label"] == 1,
        "attack_score",
    ]

    figure, axis = plt.subplots(figsize=(9, 5))

    if not predicted_normal.empty:
        axis.hist(
            predicted_normal,
            bins=30,
            alpha=0.7,
            label=f"Predicted normal (n={len(predicted_normal):,})",
        )

    if not predicted_attack.empty:
        axis.hist(
            predicted_attack,
            bins=30,
            alpha=0.7,
            label=f"Predicted attack (n={len(predicted_attack):,})",
        )

    axis.axvline(
        0.50,
        linestyle="--",
        linewidth=1.5,
        label="Detection threshold (0.50)",
    )

    axis.set_title("Distribution of Model-Generated Attack Scores")
    axis.set_xlabel("Model attack score")
    axis.set_ylabel("Number of records")
    axis.set_xlim(0, 1)
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    axis.legend(loc="best")

    return save_figure(
        figure,
        "attack_score_distribution.png",
    )


def plot_protocol_alerts(
    detections: pd.DataFrame,
) -> Path:
    """Show the most common protocols among model-predicted attack records."""
    alerts = detections.loc[
        detections["predicted_label"] == 1
    ].copy()

    figure, axis = plt.subplots(figsize=(9, 6))

    if alerts.empty:
        axis.text(
            0.5,
            0.5,
            "No predicted attack records to summarize",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )

        axis.set_title("Protocols Among Predicted Attack Records")
        axis.set_axis_off()

    elif "protocol" not in alerts.columns:
        axis.text(
            0.5,
            0.5,
            "Protocol metadata is not available",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )

        axis.set_title("Protocols Among Predicted Attack Records")
        axis.set_axis_off()

    else:
        alerts["protocol"] = (
            alerts["protocol"]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )

        protocol_counts = (
            alerts["protocol"]
            .value_counts()
            .head(10)
            .sort_values()
        )

        bars = axis.barh(
            protocol_counts.index,
            protocol_counts.values,
        )

        axis.set_title("Protocols Among Predicted Attack Records")
        axis.set_xlabel("Number of predicted attack records")
        axis.set_ylabel("Network protocol")
        axis.grid(axis="x", linestyle="--", alpha=0.3)

        axis.bar_label(
            bars,
            fmt="%d",
            padding=3,
        )

    return save_figure(
        figure,
        "protocol_alert_counts.png",
    )


def plot_random_forest_feature_importance() -> Path:
    """
    Plot the most influential encoded features from the fitted Random Forest.

    Feature importance indicates model reliance, not proof of causation
    or the certainty that a feature represents an attack.
    """
    # Load only locally generated model artifacts that you trust.
    transformer = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(RANDOM_FOREST_PATH)

    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            "The saved Random Forest model does not expose feature importances."
        )

    feature_names = transformer.get_feature_names_out()
    importances = np.asarray(
        model.feature_importances_,
        dtype=float,
    )

    if len(feature_names) != len(importances):
        raise ValueError(
            "The number of transformed feature names does not match "
            "the Random Forest importance values."
        )

    if not np.isfinite(importances).all():
        raise ValueError(
            "Random Forest feature importances contain invalid values."
        )

    if not len(importances):
        raise ValueError(
            "The trained Random Forest has no feature importance values."
        )

    top_count = min(20, len(importances))

    top_indices = np.argsort(importances)[-top_count:]

    top_names = np.asarray(
        feature_names,
        dtype=str,
    )[top_indices]

    top_values = importances[top_indices]

    order = np.argsort(top_values)

    figure, axis = plt.subplots(
        figsize=(11, max(6, top_count * 0.35)),
    )

    bars = axis.barh(
        top_names[order],
        top_values[order],
    )

    axis.set_title("Top Random Forest Feature Importances")
    axis.set_xlabel("Feature importance")
    axis.set_ylabel("Encoded model feature")
    axis.grid(axis="x", linestyle="--", alpha=0.3)

    axis.bar_label(
        bars,
        fmt="%.4f",
        padding=3,
    )

    return save_figure(
        figure,
        "random_forest_feature_importance.png",
    )


def main() -> int:
    """Generate visualizations from existing evaluation and detection results."""
    print("=" * 72)
    print("AI-POWERED CYBERSECURITY THREAT DETECTION SYSTEM")
    print("Step 11: Security Visualization")
    print("=" * 72)

    try:
        VISUALIZATION_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_comparison, detections = load_and_validate_results()

        print(
            f"\nModel comparison records: {len(model_comparison):,}"
        )
        print(f"Detection records: {len(detections):,}")

        generated_files = []

        generated_files.append(
            plot_model_performance(model_comparison)
        )

        generated_files.append(
            plot_alert_severity(detections)
        )

        generated_files.append(
            plot_attack_score_distribution(detections)
        )

        generated_files.append(
            plot_protocol_alerts(detections)
        )

        generated_files.append(
            plot_random_forest_feature_importance()
        )

        print("\n" + "=" * 72)
        print("SUCCESS: VISUALIZATION GENERATION COMPLETED")
        print("=" * 72)

        print("\nGenerated files:")

        for path in generated_files:
            print(f"  {path.relative_to(PROJECT_ROOT)}")

        print("\nImportant:")
        print("  Charts were generated from existing model outputs.")
        print("  No new model training was performed.")
        print("  Feature importance does not establish causation.")
        print("  Predicted attacks are not necessarily confirmed incidents.")

        return 0

    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        MemoryError,
    ) as error:
        print("\nERROR: Visualization generation could not be completed.")
        print(error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

