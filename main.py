
"""
Entry point for the AI-Powered Cybersecurity Threat Detection System.

Current workflow:
- Load evaluation and offline detection results.
- Generate security performance visualizations.
- Save the graphs to outputs/visualizations/.
"""

from src.visualization import main as run_visualization


def main() -> int:
    """Start the visualization workflow."""
    return run_visualization()


if __name__ == "__main__":
    raise SystemExit(main())
