
"""
Dataset loading and exploration utilities.

This module loads the prepared UNSW-NB15 training and testing CSV files
and generates readable summaries for initial dataset inspection.
"""

from pathlib import Path
from typing import Optional

import pandas as pd


# Resolve paths relative to the project, not to the terminal's location.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

TRAINING_FILE = "UNSW_NB15_training-set.csv"
TESTING_FILE = "UNSW_NB15_testing-set.csv"

TRAINING_PATH = DATA_DIR / TRAINING_FILE
TESTING_PATH = DATA_DIR / TESTING_FILE


def load_csv(file_path: Path) -> pd.DataFrame:
    """
    Load a CSV file into a Pandas DataFrame.

    Parameters
    ----------
    file_path:
        Absolute or project-relative path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        The loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the requested CSV file does not exist.
    ValueError
        If the file is empty or cannot be parsed correctly.
    """
    path = Path(file_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Place the required CSV file inside: {DATA_DIR}"
        )

    if path.stat().st_size == 0:
        raise ValueError(f"The dataset file is empty: {path}")

    try:
        # low_memory=False avoids inconsistent type inference across chunks.
        try:
            dataframe = pd.read_csv(path, low_memory=False)
        except UnicodeDecodeError:
            # Fallback for CSV files using a different character encoding.
            dataframe = pd.read_csv(
                path,
                low_memory=False,
                encoding="latin-1",
            )

    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"The dataset does not contain a readable CSV header: {path}"
        ) from exc

    except pd.errors.ParserError as exc:
        raise ValueError(
            f"Could not parse the CSV file: {path}\n"
            "Check whether the file is complete and correctly formatted."
        ) from exc

    if dataframe.empty:
        raise ValueError(
            f"The CSV contains column headers but no data rows: {path}"
        )

    # Remove accidental whitespace from column names without changing
    # the underlying data values.
    dataframe.columns = [
        str(column).strip() for column in dataframe.columns
    ]

    if dataframe.columns.duplicated().any():
        duplicate_columns = dataframe.columns[
            dataframe.columns.duplicated()
        ].tolist()

        raise ValueError(
            f"Duplicate column names were found in {path.name}: "
            f"{duplicate_columns}"
        )

    return dataframe


def print_dataset_summary(
    dataframe: pd.DataFrame,
    dataset_name: str,
    file_path: Optional[Path] = None,
) -> None:
    """
    Print an initial data-quality and structure summary.
    """
    print("\n" + "=" * 72)
    print(f"DATASET SUMMARY: {dataset_name}")
    print("=" * 72)

    if file_path is not None:
        print(f"File: {file_path}")

    row_count, column_count = dataframe.shape
    memory_mb = dataframe.memory_usage(deep=True).sum() / (1024 ** 2)

    print(f"Rows: {row_count:,}")
    print(f"Columns: {column_count:,}")
    print(f"Approximate DataFrame memory: {memory_mb:.2f} MB")

    print("\nColumn names:")
    for position, column in enumerate(dataframe.columns, start=1):
        print(f"  {position:02d}. {column}")

    numeric_columns = dataframe.select_dtypes(
        include="number"
    ).columns

    non_numeric_columns = dataframe.select_dtypes(
        exclude="number"
    ).columns

    print("\nData types:")
    print(f"  Numeric columns: {len(numeric_columns)}")
    print(f"  Non-numeric columns: {len(non_numeric_columns)}")

    print("\nMissing-value analysis:")
    missing_counts = dataframe.isna().sum()
    columns_with_missing = missing_counts[missing_counts > 0]

    total_missing = int(missing_counts.sum())
    rows_with_missing = int(dataframe.isna().any(axis=1).sum())

    print(f"  Total missing cells: {total_missing:,}")
    print(f"  Rows containing missing values: {rows_with_missing:,}")

    if columns_with_missing.empty:
        print("  No missing values detected.")
    else:
        print("  Columns containing missing values:")
        for column, count in columns_with_missing.sort_values(
            ascending=False
        ).head(10).items():
            print(f"    {column}: {count:,}")

        if len(columns_with_missing) > 10:
            print("    Additional columns omitted from this summary.")

    duplicate_count = int(dataframe.duplicated().sum())

    print("\nDuplicate-row analysis:")
    print(f"  Fully duplicated rows: {duplicate_count:,}")

    if "label" in dataframe.columns:
        print("\nBinary label distribution:")
        label_counts = dataframe["label"].value_counts(dropna=False)

        for label, count in label_counts.items():
            percentage = (count / row_count) * 100
            print(
                f"  Label {label!r}: {count:,} "
                f"records ({percentage:.2f}%)"
            )
    else:
        print(
            "\nNo 'label' column was found. "
            "Binary classification labels must be verified."
        )

    if "attack_cat" in dataframe.columns:
        print("\nAttack-category distribution:")
        attack_counts = dataframe["attack_cat"].value_counts(
            dropna=False
        )

        for category, count in attack_counts.items():
            percentage = (count / row_count) * 100
            print(
                f"  {category!r}: {count:,} "
                f"records ({percentage:.2f}%)"
            )

    preferred_columns = [
        "id",
        "dur",
        "proto",
        "service",
        "state",
        "attack_cat",
        "label",
    ]

    preview_columns = [
        column
        for column in preferred_columns
        if column in dataframe.columns
    ]

    if not preview_columns:
        preview_columns = dataframe.columns[:6].tolist()

    print("\nSample records:")
    print(
        dataframe[preview_columns]
        .head(5)
        .to_string(index=False)
    )

    print("\n" + "=" * 72)


def main() -> int:
    """
    Execute an initial inspection of the training and testing datasets.
    """
    print("AI-Powered Cybersecurity Threat Detection System")
    print("Dataset loading and initial exploration")
    print(f"Project root: {PROJECT_ROOT}")

    required_files = [
        ("Training dataset", TRAINING_PATH),
        ("Testing dataset", TESTING_PATH),
    ]

    missing_or_empty = []

    for dataset_name, path in required_files:
        if not path.is_file():
            missing_or_empty.append(
                (dataset_name, path, "file not found")
            )
        elif path.stat().st_size == 0:
            missing_or_empty.append(
                (dataset_name, path, "file is empty")
            )

    if missing_or_empty:
        print("\nERROR: Required dataset files are not ready.")

        for dataset_name, path, reason in missing_or_empty:
            print(f"\n{dataset_name}: {reason}")
            print(f"Expected path: {path}")

        print("\nDownload the prepared UNSW-NB15 CSV files and place")
        print("them in the project's data directory.")
        print("See docs/dataset_guide.md for instructions.")

        return 1

    for dataset_name, path in required_files:
        try:
            dataframe = load_csv(path)
            print_dataset_summary(
                dataframe=dataframe,
                dataset_name=dataset_name,
                file_path=path,
            )

            # Release the DataFrame before loading the next dataset.
            del dataframe

        except (OSError, ValueError) as error:
            print(f"\nERROR while loading {dataset_name}:")
            print(error)
            return 1

    print("\nSUCCESS: Both datasets were loaded and inspected.")
    print("No model has been trained in this step.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
