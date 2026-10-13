
"""Evaluate the saved sentiment model on labeled feedback."""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "sentiment_pipeline.joblib"
DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "student_feedback_processed.csv"
)
REPORT_PATH = PROJECT_DIR / "reports" / "model_evaluation.json"

VALID_LABELS = {"negative", "neutral", "positive"}


def evaluate_model(model, data):
    """Evaluate model predictions against labeled sentiment values."""
    required = {"text", "sentiment"}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing))}"
        )

    frame = data.copy()
    frame["text"] = frame["text"].fillna("").astype(str).str.strip()
    frame["sentiment"] = (
        frame["sentiment"].fillna("").astype(str).str.strip().str.lower()
    )

    frame = frame[
        frame["text"].ne("")
        & frame["sentiment"].isin(VALID_LABELS)
    ]

    if frame.empty:
        raise ValueError("No valid labeled feedback is available for evaluation.")

    y_true = frame["sentiment"].tolist()
    y_pred = [
        str(label).lower()
        for label in model.predict(frame["text"].tolist())
    ]

    labels = ["negative", "neutral", "positive"]

    return {
        "records_evaluated": len(frame),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist(),
        "class_order": labels,
        "evaluation_data": str(DATA_PATH),
        "warning": (
            "Evaluate on a held-out test set for a trustworthy generalization "
            "estimate. Evaluating on training data can overstate performance."
        ),
    }


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Run python src/train.py first."
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    model = joblib.load(MODEL_PATH)
    data = pd.read_csv(DATA_PATH)
    results = evaluate_model(model, data)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(f"Records evaluated: {results['records_evaluated']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(f"Report saved to: {REPORT_PATH}")
    print(
        "\nImportant: this script evaluates on the entire processed dataset. "
        "Its scores are not a held-out test result and should not replace the "
        "86.01% accuracy and 0.8105 macro F1 reported by the existing test split."
    )


if __name__ == "__main__":
    main()
