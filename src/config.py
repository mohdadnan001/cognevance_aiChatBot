"""Central configuration for the Cognevance AI Chatbot."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DOCS_DIR = BASE_DIR / "docs"
MODELS_DIR = BASE_DIR / "models"

GENISYS_PATH = RAW_DATA_DIR / "genisys_intent.json"
BITEXT_PATH = RAW_DATA_DIR / "bitext_customer_support_27k.csv"
INTENTS_PATH = DATA_DIR / "intents.json"

MODEL_PATH = MODELS_DIR / "intent_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
REPORT_PATH = DOCS_DIR / "classification_report.txt"
CONFUSION_MATRIX_PATH = DOCS_DIR / "confusion_matrix.png"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CONFIDENCE_THRESHOLD = 0.30

FALLBACK_RESPONSES = [
    "I'm not sure I understood that. Could you rephrase your question?",
    "Sorry, I didn't catch that. Could you try asking in a different way?",
    "I'm still learning and I'm not confident about that one. Could you give me more detail?",
]
