
"""
Entry point for the AI-Powered Cybersecurity Threat Detection System.

Current workflow:
- Load and clean the training dataset.
- Engineer numerical and categorical features.
- Train Logistic Regression and Random Forest.
- Save the fitted preprocessing transformer and trained models.
"""

from src.model_training import main as run_model_training


def main() -> int:
    """Start model training."""
    return run_model_training()


if __name__ == "__main__":
    raise SystemExit(main())
