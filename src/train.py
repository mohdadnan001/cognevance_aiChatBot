"""Train the intent classifier and persist model artifacts."""

import json
import time
from datetime import datetime, timezone

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    CONFUSION_MATRIX_PATH,
    DOCS_DIR,
    INTENTS_PATH,
    METADATA_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    REPORT_PATH,
    TEST_SIZE,
)
from src.preprocess import ensure_nltk_data, preprocess_text


def load_intents(path=INTENTS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_dataset(intents: list[dict]) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for intent in intents:
        for pattern in intent["patterns"]:
            texts.append(pattern)
            labels.append(intent["tag"])
    return texts, labels


def save_confusion_matrix(y_test, y_pred, labels) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(22, 20))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=labels, xticks_rotation=90, ax=ax, colorbar=False
    )
    ax.set_title("Intent classification confusion matrix")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(fig)


def main() -> None:
    ensure_nltk_data()
    started = time.perf_counter()

    payload = load_intents()
    intents = payload["intents"]
    texts, labels = build_dataset(intents)
    print(f"Loaded {len(texts)} patterns across {len(intents)} intents")

    cleaned = [preprocess_text(text) for text in texts]
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    C=10.0,
                    tol=1e-3,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    print("Training TF-IDF + Logistic Regression pipeline...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    labels_sorted = sorted(set(labels))
    report = classification_report(
        y_test, y_pred, labels=labels_sorted, digits=3, zero_division=0
    )

    print(f"\nTest accuracy: {accuracy:.4f}\n")
    print(report)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    metadata = {
        "accuracy": float(accuracy),
        "num_intents": len(intents),
        "num_samples": len(texts),
        "num_train": len(X_train),
        "num_test": len(X_test),
        "intents": sorted(intent["tag"] for intent in intents),
        "responses": {intent["tag"]: intent["responses"] for intent in intents},
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_seconds": round(time.perf_counter() - started, 2),
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    save_confusion_matrix(y_test, y_pred, labels_sorted)

    print(f"Saved model      -> {MODEL_PATH}")
    print(f"Saved metadata   -> {METADATA_PATH}")
    print(f"Saved report     -> {REPORT_PATH}")
    print(f"Saved matrix     -> {CONFUSION_MATRIX_PATH}")
    print(f"Done in {metadata['training_seconds']}s")


if __name__ == "__main__":
    main()
