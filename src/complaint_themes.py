
from pathlib import Path
import re

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "student_feedback_processed.csv"
)
OUTPUT_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "complaint_themes.csv"
)


# Transparent keyword rules: a match suggests a theme, not a verified cause.

THEME_RULES = {
    "Lab equipment and practical learning": [
        "lab", "laboratory", "practical", "hands-on",
        "equipment", "latest technologies", "experiments",
    ],
    "Library books and availability": [
        "books", "book availability", "not available",
        "less number", "library resources", "shortage of books",
    ],
    "Library systems and rules": [
        "issue the book", "issuing", "system in library",
        "library system", "rules in library", "library rules",
    ],
    "Course material and content": [
        "course material", "course content", "course materials",
        "study material", "syllabus", "rote learning",
        "cramming", "subject content",
    ],
    "Assessment and grading fairness": [
        "marks", "evaluation", "grading", "remarks",
        "assessment", "ca ", "unfair marking", "exam evaluation",
    ],
    "Teaching quality and subject knowledge": [
        "lecture delivery", "lecture delivery is",
        "not teach", "don't teach", "dont teach",
        "subject knowledge", "grip on subject",
        "experienced", "not experienced", "not effective",
        "not satisfactory", "not good at teaching",
    ],
    "Teacher interaction and student support": [
        "interact", "interaction", "listen our problems",
        "student problems", "cooperate", "behave",
        "behaviour", "communication", "punctuality",
    ],
    "General dissatisfaction": [
        "not good", "not so good", "not satisfactory",
        "not up to the mark", "not upto the mark",
        "needs improvement", "need improvement",
        "poor", "bad", "unsatisfactory",
    ],
}




def identify_themes(comment):
    """Return all matching themes; unmatched comments need human review."""
    if not isinstance(comment, str) or not comment.strip():
        return ["Other / Review"]

    text = re.sub(r"\s+", " ", comment.lower()).strip()
    matches = []

    for theme, keywords in THEME_RULES.items():
        for keyword in keywords:
            if keyword in text:
                matches.append(theme)
                break

    return matches or ["Other / Review"]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run the preprocessing script first."
        )

    data = pd.read_csv(DATA_PATH)

    required = {"text", "topic", "sentiment"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing))}"
        )

    negative = data[
        data["sentiment"].astype(str).str.lower().eq("negative")
    ].copy()

    if negative.empty:
        print("No negative feedback records were found.")
        return

    negative["matched_themes"] = negative["text"].apply(identify_themes)
    negative["theme"] = negative["matched_themes"].apply(
        lambda themes: "; ".join(themes)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    negative.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"Negative comments analyzed: {len(negative)}")
    print(f"Saved evidence and themes to: {OUTPUT_PATH}")
    print("\nTheme matches (a comment can match multiple themes):")

    theme_counts = (
        negative.explode("matched_themes")["matched_themes"]
        .value_counts()
    )
    print(theme_counts.to_string())

    print("\nSample evidence:")
    for _, row in negative.head(8).iterrows():
        print(f"\nTopic: {row['topic']}")
        print(f"Theme: {row['theme']}")
        print(f"Comment: {row['text']}")


if __name__ == "__main__":
    main()
