
from collections import Counter
import re

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


EXTRA_STOP_WORDS = {
    "student", "students", "college", "university",
    "good", "bad", "nice", "very", "much", "also",
}

STOP_WORDS = set(ENGLISH_STOP_WORDS) | EXTRA_STOP_WORDS


def extract_common_terms(comments, top_n=10):
    """Find frequent informative words in negative feedback."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    words = []

    for comment in comments:
        if not isinstance(comment, str):
            continue

        tokens = re.findall(r"[a-zA-Z]{3,}", comment.lower())
        words.extend(
            token for token in tokens
            if token not in STOP_WORDS
        )

    return Counter(words).most_common(top_n)


def analyze_negative_feedback(data, top_n=10):
    """Summarize frequent terms in comments labelled negative."""
    required = {"text", "sentiment", "topic"}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing))}"
        )

    negative = data[
        data["sentiment"].astype(str).str.lower().eq("negative")
    ].copy()

    if negative.empty:
        return negative, []

    negative["text"] = negative["text"].fillna("").astype(str)
    negative = negative[negative["text"].str.strip() != ""]

    common_terms = extract_common_terms(
        negative["text"].tolist(),
        top_n=top_n,
    )

    return negative, common_terms


if __name__ == "__main__":
    from pathlib import Path

    project_dir = Path(__file__).resolve().parent.parent
    data_path = (
        project_dir
        / "data"
        / "processed"
        / "student_feedback_processed.csv"
    )

    data = pd.read_csv(data_path)
    negative, terms = analyze_negative_feedback(data)

    print(f"Negative feedback records: {len(negative)}")
    print("\nMost common terms in negative feedback:")

    for term, count in terms:
        print(f"{term}: {count}")

    
    
    print("\nTopic-wise negative feedback analysis:")

    all_topic_counts = data.groupby("topic").size().rename("total_feedback")
    negative_topic_counts = (
        negative.groupby("topic").size().rename("negative_feedback")
    )

    summary = pd.concat(
        [all_topic_counts, negative_topic_counts], axis=1
    ).fillna(0)

    summary["negative_feedback"] = summary["negative_feedback"].astype(int)
    summary["negative_percentage"] = (
        summary["negative_feedback"] / summary["total_feedback"] * 100
    ).round(1)

    summary = summary.sort_values(
        "negative_percentage", ascending=False
    )

    print(summary.to_string())

