
from pathlib import Path
import sys

import pytest

# Make the project's src directory importable
PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict import load_model, predict_sentiment


@pytest.fixture(scope="module")
def model():
    """Load the trained model once for this test module."""
    return load_model()


def test_model_file_exists():
    model_path = PROJECT_DIR / "models" / "sentiment_pipeline.joblib"
    assert model_path.exists(), f"Missing trained model: {model_path}"


@pytest.mark.parametrize(
    "comment",
    [
        "The teacher explains concepts clearly.",
        "The laboratory equipment is broken.",
        "The course content is average.",
    ],
)
def test_prediction_returns_valid_sentiment(model, comment):
    result = predict_sentiment(comment, model)

    assert result["sentiment"] in {"positive", "negative", "neutral"}
    assert 0.0 <= result["confidence"] <= 1.0


def test_prediction_contains_all_class_probabilities(model):
    result = predict_sentiment(
        "The teacher explains concepts clearly.",
        model,
    )

    assert set(result["probabilities"].keys()) == {
        "positive",
        "negative",
        "neutral",
    }
    assert all(
        0.0 <= probability <= 1.0
        for probability in result["probabilities"].values()
    )
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)


@pytest.mark.parametrize("comment", ["", "   ", "\n\t"])
def test_empty_feedback_is_rejected(model, comment):
    with pytest.raises(ValueError):
        predict_sentiment(comment, model)

from complaint_themes import identify_themes


def test_lab_comment_matches_lab_theme():
    themes = identify_themes(
        "The laboratory equipment is outdated."
    )
    assert "Lab equipment and practical learning" in themes


def test_library_comment_matches_library_theme():
    themes = identify_themes(
        "All the books are not available in the library."
    )
    assert "Library books and availability" in themes


def test_course_comment_matches_course_theme():
    themes = identify_themes(
        "Course material needs improvement."
    )
    assert "Course material and content" in themes


def test_unmatched_comment_is_flagged_for_review():
    themes = identify_themes(
        "The overall experience was disappointing."
    )
    assert themes == ["Other / Review"]


def test_empty_comment_is_flagged_for_review():
    themes = identify_themes("")
    assert themes == ["Other / Review"]
