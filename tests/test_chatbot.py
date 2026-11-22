"""Tests for preprocessing and chatbot inference."""

import pytest

from src.config import MODEL_PATH
from src.preprocess import preprocess_text

requires_model = pytest.mark.skipif(
    not MODEL_PATH.exists(), reason="Run `python -m src.train` first"
)


def test_preprocess_removes_stopwords_and_punctuation():
    tokens = preprocess_text("Hello, how are YOU doing today?!").split()
    assert "hello" in tokens
    assert "you" not in tokens
    assert "," not in " ".join(tokens)


def test_preprocess_lemmatizes():
    tokens = preprocess_text("cancelling orders").split()
    assert "cancel" in tokens


def test_preprocess_removes_dataset_placeholders():
    result = preprocess_text("cancel order {{Order Number}}")
    assert "{{" not in result
    assert "cancel" in result


@pytest.fixture(scope="module")
def bot():
    from src.chatbot import Chatbot

    return Chatbot()


@requires_model
def test_greeting_intent(bot):
    result = bot.predict("hello there")
    assert result["intent"] == "greeting"
    assert result["confidence"] > 0.30
    assert result["response"]


@requires_model
def test_track_order_intent(bot):
    result = bot.predict("where is my order?")
    assert result["intent"] == "track_order"


@requires_model
def test_unknown_input_falls_back(bot):
    result = bot.predict("zxqv blorf quux")
    assert result["intent"] is None
    assert result["response"]
