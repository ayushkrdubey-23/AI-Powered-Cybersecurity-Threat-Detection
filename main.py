
"""
Entry point for the AI-Powered Cybersecurity Threat Detection System.

Step 5:
Load and inspect the prepared UNSW-NB15 training and testing datasets.
"""

from src.data_loader import main as run_dataset_inspection


def main() -> int:
    """Start the dataset inspection workflow."""
    return run_dataset_inspection()


if __name__ == "__main__":
    raise SystemExit(main())
