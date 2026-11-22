"""Evaluate the saved model on a held-out test split."""

import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.config import MODEL_PATH, RANDOM_STATE, TEST_SIZE
from src.preprocess import preprocess_text
from src.train import build_dataset, load_intents


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python -m src.train` first.")

    pipeline = joblib.load(MODEL_PATH)
    intents = load_intents()["intents"]
    texts, labels = build_dataset(intents)
    cleaned = [preprocess_text(text) for text in texts]

    _, X_test, _, y_test = train_test_split(
        cleaned,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Evaluation on {len(X_test)} held-out samples")
    print(f"Accuracy: {accuracy:.4f}\n")
    print(
        classification_report(
            y_test, y_pred, labels=sorted(set(labels)), digits=3, zero_division=0
        )
    )


if __name__ == "__main__":
    main()
