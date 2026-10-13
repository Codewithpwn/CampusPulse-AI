
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = (
    PROJECT_DIR / "data" / "processed"
    / "student_feedback_processed.csv"
)
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

LABELS = ["negative", "neutral", "positive"]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Processed data not found. Run "
            "python src/preprocessing.py first."
        )

    df = pd.read_csv(DATA_PATH)

    required = {"text", "topic", "sentiment", "source_row_id"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}. "
            "Update preprocessing.py and rerun it first."
        )

    df = df.dropna(
        subset=["text", "sentiment", "source_row_id"]
    ).copy()

    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df = df[df["sentiment"].isin(LABELS)]

    print("Records:", len(df))
    print("Original source rows:", df["source_row_id"].nunique())

    print("\nSentiment counts:")
    print(df["sentiment"].value_counts())

    # Keep every topic from an original row in the same split.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(
            df["text"],
            df["sentiment"],
            groups=df["source_row_id"],
        )
    )

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    print("\nTraining records:", len(train_df))
    print("Test records:", len(test_df))

    train_groups = set(train_df["source_row_id"])
    test_groups = set(test_df["source_row_id"])

    assert train_groups.isdisjoint(test_groups), (
        "Data leakage detected: source rows overlap."
    )

    print("Source-row leakage check: PASSED")

    print("\nTraining sentiment counts:")
    print(train_df["sentiment"].value_counts())

    print("\nTest sentiment counts:")
    print(test_df["sentiment"].value_counts())

    if train_df["sentiment"].nunique() < 2:
        raise ValueError(
            "Training split has fewer than two sentiment classes."
        )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=10000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    print("\nTraining CampusPulse AI sentiment model...")

    model.fit(train_df["text"], train_df["sentiment"])

    predictions = model.predict(test_df["text"])
    actual = test_df["sentiment"]

    accuracy = accuracy_score(actual, predictions)
    macro_f1 = f1_score(
        actual,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    print("\nAccuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))

    print("\nClassification report:")
    report_text = classification_report(
        actual,
        predictions,
        labels=LABELS,
        zero_division=0,
    )
    print(report_text)

    print("\nConfusion matrix")
    print("Rows = actual; columns = predicted")
    print("Class order:", LABELS)
    matrix = confusion_matrix(
        actual,
        predictions,
        labels=LABELS,
    )
    print(matrix)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "sentiment_pipeline.joblib"
    joblib.dump(model, model_path)

    metrics = {
        "records": int(len(df)),
        "train_records": int(len(train_df)),
        "test_records": int(len(test_df)),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "labels": LABELS,
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            actual,
            predictions,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "split_method": "GroupShuffleSplit by source_row_id",
        "random_state": 42,
    }

    metrics_path = REPORT_DIR / "model_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\nModel saved:", model_path)
    print("Metrics saved:", metrics_path)
    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
