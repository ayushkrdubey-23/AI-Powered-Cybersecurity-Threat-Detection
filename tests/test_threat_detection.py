
"""Automated tests for threat scoring, severity and alert reporting."""

import json

import numpy as np
import pandas as pd
import pytest

import src.threat_detection as threat_detection

from src.feature_engineering import build_preprocessor


class DummyClassifier:
    """Small classifier substitute for testing class-score selection."""

    classes_ = np.array([1, 0])

    def predict_proba(self, features):
        """
        Return scores in the order specified by classes_.

        Since class 1 is first, its score must be taken from column zero.
        """
        return np.tile(
            np.array([0.80, 0.20]),
            (features.shape[0], 1),
        )


def test_attack_score_selects_the_correct_class():
    """Attack scores must be selected using the model's class labels."""
    model = DummyClassifier()
    features = np.zeros((3, 2))

    scores = threat_detection.get_attack_scores(
        model,
        features,
    )

    np.testing.assert_allclose(
        scores,
        np.array([0.80, 0.80, 0.80]),
    )


def test_attack_score_rejects_wrong_prediction_count():
    """A malformed probability matrix must be rejected."""
    class IncorrectClassifier:
        classes_ = np.array([0, 1])

        def predict_proba(self, features):
            return np.array([[0.2, 0.8]])

    with pytest.raises(ValueError, match="number of scores"):
        threat_detection.get_attack_scores(
            IncorrectClassifier(),
            np.zeros((3, 2)),
        )


@pytest.mark.parametrize(
    ("score", "is_attack", "expected"),
    [
        (0.10, False, "NONE"),
        (0.49, False, "NONE"),
        (0.50, True, "LOW"),
        (0.64, True, "LOW"),
        (0.65, True, "MEDIUM"),
        (0.84, True, "MEDIUM"),
        (0.85, True, "HIGH"),
        (0.99, True, "HIGH"),
        (0.99, False, "NONE"),
    ],
)
def test_severity_thresholds(score, is_attack, expected):
    """Severity must follow the documented score thresholds."""
    assert (
        threat_detection.severity_for_score(score, is_attack)
        == expected
    )


def test_detection_results_exclude_ground_truth_labels():
    """Model-generated alerts must not include ground-truth labels."""
    raw = pd.DataFrame(
        {
            "id": [101, 102, 103],
            "proto": ["tcp", "udp", "icmp"],
            "dur": [0.1, 0.3, 0.5],
            "label": [0, 1, 1],
            "attack_cat": ["Normal", "DoS", "Exploits"],
        }
    )

    results = threat_detection.build_detection_results(
        raw_data=raw,
        attack_scores=np.array([0.20, 0.70, 0.90]),
    )

    assert results["predicted_label"].tolist() == [0, 1, 1]
    assert results["severity"].tolist() == ["NONE", "MEDIUM", "HIGH"]

    assert "label" not in results.columns
    assert "attack_cat" not in results.columns

    assert results["source_record_id"].tolist() == [101, 102, 103]


def test_detection_results_reject_score_count_mismatch():
    """Every event needs exactly one model score."""
    raw = pd.DataFrame({"id": [1, 2]})

    with pytest.raises(ValueError, match="does not match"):
        threat_detection.build_detection_results(
            raw_data=raw,
            attack_scores=np.array([0.8]),
        )


def test_detection_results_reject_invalid_scores():
    """Detection results must never accept NaN or out-of-range scores."""
    raw = pd.DataFrame({"id": [1, 2]})

    with pytest.raises(ValueError, match="invalid numeric values"):
        threat_detection.build_detection_results(
            raw_data=raw,
            attack_scores=np.array([0.3, np.nan]),
        )

    with pytest.raises(ValueError, match="between 0 and 1"):
        threat_detection.build_detection_results(
            raw_data=raw,
            attack_scores=np.array([0.3, 1.2]),
        )


def test_detection_feature_preparation_removes_target_fields():
    """Inference features must exclude label and attack category fields."""
    training_features = pd.DataFrame(
        {
            "dur": [0.1, 0.2, 0.3, 0.4],
            "proto": ["tcp", "udp", "tcp", "udp"],
        }
    )

    transformer, _, _ = build_preprocessor(training_features)
    transformer.fit(training_features)

    incoming = pd.DataFrame(
        {
            "id": [201, 202],
            "dur": [0.5, 0.6],
            "proto": ["tcp", "icmp"],
            "label": [0, 1],
            "attack_cat": ["Normal", "DoS"],
        }
    )

    features = threat_detection.prepare_detection_features(
        raw_data=incoming,
        fitted_transformer=transformer,
    )

    assert features.columns.tolist() == ["dur", "proto"]
    assert len(features) == len(incoming)


def test_save_detection_outputs_creates_reports(tmp_path, monkeypatch):
    """Detection reports and summaries should be saved successfully."""
    monkeypatch.setattr(
        threat_detection,
        "OUTPUT_DIR",
        tmp_path,
    )

    raw = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "proto": ["tcp", "udp", "tcp"],
            "dur": [0.1, 0.2, 0.3],
        }
    )

    results = threat_detection.build_detection_results(
        raw_data=raw,
        attack_scores=np.array([0.20, 0.70, 0.90]),
    )

    saved = threat_detection.save_detection_outputs(results)

    assert saved["results_path"].is_file()
    assert saved["alerts_path"].is_file()
    assert saved["summary_path"].is_file()

    summary = json.loads(
        saved["summary_path"].read_text(encoding="utf-8")
    )

    assert summary["records_processed"] == 3
    assert summary["alerts_generated"] == 2
    assert summary["automated_response"] is False

    assert len(pd.read_csv(saved["results_path"])) == 3
    assert len(pd.read_csv(saved["alerts_path"])) == 2
