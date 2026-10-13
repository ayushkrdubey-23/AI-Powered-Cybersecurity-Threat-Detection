
"""
Entry point for the AI-Powered Cybersecurity Threat Detection System.

Current workflow:
- Load saved training transformations and the Random Forest classifier.
- Run offline dataset replay.
- Generate model predictions and threat alerts.
- Save detection records and an alert summary.
"""

from src.threat_detection import main as run_threat_detection


def main() -> int:
    """Start the offline threat detection workflow."""
    return run_threat_detection()


if __name__ == "__main__":
    raise SystemExit(main())
