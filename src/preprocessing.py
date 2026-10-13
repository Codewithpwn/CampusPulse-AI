
from pathlib import Path
import pandas as pd


# ============================================================
# CampusPulse AI - Data Preprocessing
# Dataset: finalDataset0.2.xlsx
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_FILE = PROCESSED_DIR / "student_feedback_processed.csv"

# Numeric sentiment label column -> feedback text column.
# These names match the original Excel dataset exactly.
COLUMN_PAIRS = {
    "teaching": ("teaching", "teaching.1"),
    "coursecontent": ("coursecontent", "coursecontent.1"),
    "examination": ("examination", "Examination"),
    "labwork": ("labwork", "labwork.1"),
    "library": ("library_facilities", " library_facilities"),
    "extracurricular": (
        "extracurricular",
        "extracurricular.1",
    ),
}

# Verify this mapping against the dataset documentation.
LABEL_MAP = {
    -1: "negative",
    0: "neutral",
    1: "positive",
}


def find_dataset():
    """Find the original dataset, including in nested folders."""
    supported_extensions = {".xlsx", ".xls", ".csv"}

    files = [
        path
        for path in RAW_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in supported_extensions
        and not path.name.startswith("~$")
    ]

    if not files:
        raise FileNotFoundError(
            f"No Excel or CSV dataset found in: {RAW_DIR}"
        )

    # Prefer the exact file we inspected earlier.
    preferred_name = "finaldataset0.2.xlsx"

    for path in files:
        if path.name.lower() == preferred_name:
            return path

    if len(files) > 1:
        print("Multiple datasets found:")
        for path in files:
            print(" -", path.name)
        print("Using:", files[0].name)

    return files[0]


def load_dataset(file_path):
    """Load a CSV or Excel file without altering column names."""
    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    # Preserve leading spaces and capitalization in column names.
    df.columns = [str(column) for column in df.columns]

    return df


def prepare_dataset():
    """Convert six topic/label pairs into a tidy dataset."""
    file_path = find_dataset()
    df = load_dataset(file_path)

    print("=" * 65)
    print("CAMPUSPULSE AI - PREPROCESSING")
    print("=" * 65)
    print("Source file:", file_path.name)
    print("Original shape:", df.shape)

    # Detect missing or incorrectly named columns.
    required_columns = {
        column
        for pair in COLUMN_PAIRS.values()
        for column in pair
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "The dataset does not contain all expected columns.\n"
            f"Missing: {sorted(missing_columns)}\n"
            f"Available: {df.columns.tolist()}"
        )

    records = []

    # Process each feedback topic separately.
    for topic, (label_column, text_column) in COLUMN_PAIRS.items():

        # Select columns separately to avoid duplicate-name problems.
        part = pd.DataFrame({
            "label": df[label_column],
            "text": df[text_column],
        })

        # Keep the original Excel row index for grouped evaluation.
        part["source_row_id"] = df.index

        # Convert numeric labels safely.
        part["label"] = pd.to_numeric(
            part["label"], errors="coerce"
        )

        # Clean feedback text without deleting meaningful words.
        part["text"] = (
            part["text"]
            .fillna("")
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        # Remove missing labels and empty feedback.
        part = part.dropna(subset=["label"])
        part = part[part["text"].str.len() > 0]

        # Retain only labels documented in LABEL_MAP.
        part = part[part["label"].isin(LABEL_MAP)]

        part["sentiment"] = part["label"].map(LABEL_MAP)
        part["topic"] = topic

        records.append(
            part[
                ["text", "topic", "sentiment", "source_row_id"]
            ]
        )

    if not records:
        raise ValueError("No valid feedback records were found.")

    result = pd.concat(records, ignore_index=True)

    # Normalize text for duplicate/conflict checks only.
    result["normalized_text"] = (
        result["text"]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Identify identical comments in the same topic with
    # contradictory sentiment labels.
    conflict_counts = (
        result.groupby(
            ["topic", "normalized_text"]
        )["sentiment"].nunique()
    )

    conflicting_keys = set(
        conflict_counts[conflict_counts > 1].index
    )

    if conflicting_keys:
        print(
            "\nWarning: Excluding",
            len(conflicting_keys),
            "topic/text combinations with conflicting labels."
        )

        keep_rows = [
            (topic, text) not in conflicting_keys
            for topic, text in zip(
                result["topic"],
                result["normalized_text"],
            )
        ]

        result = result.loc[keep_rows].copy()

    # Remove repeated text within the same topic.
    # Keep the original source-row identifier for splitting.
    result = result.drop_duplicates(
        subset=["topic", "normalized_text"],
        keep="first",
    )

    result = result.drop(columns=["normalized_text"])
    result = result.reset_index(drop=True)

    # Validate the final dataset.
    if result.empty:
        raise ValueError(
            "No records remain after preprocessing."
        )

    if result[["text", "topic", "sentiment", "source_row_id"]].isnull().any().any():
        raise ValueError(
            "Unexpected missing values remain in the processed data."
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    result.to_csv(OUTPUT_FILE, index=False)

    print("\nProcessed records:", len(result))
    print(
        "Unique original rows:",
        result["source_row_id"].nunique(),
    )

    print("\nSentiment distribution:")
    print(result["sentiment"].value_counts().to_string())

    print("\nTopic distribution:")
    print(result["topic"].value_counts().to_string())

    print("\nOutput columns:")
    print(result.columns.tolist())

    print("\nFirst five processed records:")
    print(result.head().to_string(index=False))

    print("\nSaved processed dataset to:")
    print(OUTPUT_FILE)

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    prepare_dataset()
