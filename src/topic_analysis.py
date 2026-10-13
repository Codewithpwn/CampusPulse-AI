
"""Topic-wise analysis of student feedback."""

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "student_feedback_processed.csv"
)


def analyze_topics(data):
    """Return topic-level sentiment counts and negative-feedback rates."""
    required = {"topic", "sentiment"}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing))}"
        )

    frame = data.copy()
    frame["topic"] = frame["topic"].fillna("").astype(str).str.strip()
    frame["sentiment"] = (
        frame["sentiment"].fillna("").astype(str).str.strip().str.lower()
    )

    frame = frame[
        frame["topic"].ne("")
        & frame["sentiment"].isin({"positive", "neutral", "negative"})
    ]

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "topic",
                "total_feedback",
                "positive",
                "neutral",
                "negative",
                "negative_percentage",
            ]
        )

    counts = pd.crosstab(frame["topic"], frame["sentiment"])

    for label in ["positive", "neutral", "negative"]:
        if label not in counts.columns:
            counts[label] = 0

    counts = counts[["positive", "neutral", "negative"]]
    counts["total_feedback"] = counts.sum(axis=1)

    counts["negative_percentage"] = (
        counts["negative"] / counts["total_feedback"] * 100
    ).round(1)

    result = counts.reset_index()
    result = result[
        [
            "topic",
            "total_feedback",
            "positive",
            "neutral",
            "negative",
            "negative_percentage",
        ]
    ]

    return result.sort_values(
        "negative_percentage", ascending=False
    ).reset_index(drop=True)


def analyze_file(path=DEFAULT_DATA_PATH):
    """Load a processed CSV and return its topic analysis."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path)
    return analyze_topics(data)


if __name__ == "__main__":
    summary = analyze_file()

    if summary.empty:
        print("No valid topic feedback was found.")
    else:
        print(summary.to_string(index=False))
