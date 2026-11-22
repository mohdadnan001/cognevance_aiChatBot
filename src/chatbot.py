"""Chatbot inference: intent recognition and response generation."""

import json
import random
from datetime import datetime
from pathlib import Path

import joblib

from src.config import (
    CONFIDENCE_THRESHOLD,
    FALLBACK_RESPONSES,
    METADATA_PATH,
    MODEL_PATH,
)
from src.preprocess import preprocess_text


class Chatbot:
    """Loads the trained pipeline and turns user text into a reply."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        metadata_path: Path = METADATA_PATH,
        threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        if not model_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                "Model artifacts not found. Run `python -m src.train` first."
            )
        self.pipeline = joblib.load(model_path)
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.responses = self.metadata["responses"]
        self.threshold = threshold
        self.labels = list(self.pipeline.classes_)

    def predict(self, text: str) -> dict:
        cleaned = preprocess_text(text)
        if not cleaned:
            return self._fallback()

        probabilities = self.pipeline.predict_proba([cleaned])[0]
        ranked = sorted(
            zip(self.labels, probabilities), key=lambda item: item[1], reverse=True
        )
        intent, confidence = ranked[0]
        top_intents = [
            {"intent": label, "confidence": round(float(score), 4)}
            for label, score in ranked[:3]
        ]

        if confidence < self.threshold:
            result = self._fallback(confidence)
            result["top_intents"] = top_intents
            return result

        return {
            "intent": intent,
            "confidence": round(float(confidence), 4),
            "response": self._generate_response(intent),
            "top_intents": top_intents,
        }

    def _generate_response(self, intent: str) -> str:
        if intent == "time_query":
            return f"The current time is {datetime.now():%H:%M}."
        responses = self.responses.get(intent) or FALLBACK_RESPONSES
        return random.choice(responses)

    def _fallback(self, confidence: float = 0.0) -> dict:
        return {
            "intent": None,
            "confidence": round(float(confidence), 4),
            "response": random.choice(FALLBACK_RESPONSES),
            "top_intents": [],
        }

    def chat(self, text: str) -> str:
        return self.predict(text)["response"]
