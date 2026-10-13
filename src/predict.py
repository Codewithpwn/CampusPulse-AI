
from pathlib import Path
import joblib

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "sentiment_pipeline.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run: python src/train.py"
        )
    return joblib.load(MODEL_PATH)


def predict_sentiment(text, model=None):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter non-empty feedback text.")

    if model is None:
        model = load_model()

    prediction = model.predict([text.strip()])[0]

    probabilities = model.predict_proba([text.strip()])[0]
    classes = model.named_steps["classifier"].classes_

    confidence = float(max(probabilities))

    scores = {
        str(label): float(probability)
        for label, probability in zip(classes, probabilities)
    }

    return {
        "text": text.strip(),
        "sentiment": str(prediction),
        "confidence": confidence,
        "probabilities": scores,
    }


if __name__ == "__main__":
    model = load_model()

    print("CampusPulse AI - Sentiment Prediction")
    print("Type 'exit' to quit.")

    while True:
        text = input("\nEnter feedback: ").strip()

        if text.lower() == "exit":
            break

        try:
            result = predict_sentiment(text, model)
            print("Predicted sentiment:", result["sentiment"])
            print(
                "Model confidence:",
                f"{result['confidence']:.1%}",
            )
            print("Class probabilities:", result["probabilities"])
        except ValueError as error:
            print(error)
