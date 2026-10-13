
"""
Entry point for the AI-Powered Cybersecurity Threat Detection System.

Current workflow:
- Load the held-out testing dataset.
- Apply the saved preprocessing transformer.
- Evaluate previously trained classifiers.
- Save evaluation metrics, reports, confusion matrices, and predictions.
"""

from src.model_evaluation import main as run_model_evaluation


def main() -> int:
    """Start independent model evaluation."""
    return run_model_evaluation()


if __name__ == "__main__":
    raise SystemExit(main())
