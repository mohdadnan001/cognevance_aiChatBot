"""Streamlit chat interface for the Cognevance AI Chatbot."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.chatbot import Chatbot

WELCOME = {
    "role": "assistant",
    "content": (
        "Hello! I'm Cognevance, an NLP-powered assistant. Ask me about "
        "orders, refunds, invoices, accounts or just say hi."
    ),
    "meta": None,
}


st.set_page_config(page_title="Cognevance AI Chatbot", layout="centered")


@st.cache_resource
def load_chatbot() -> Chatbot:
    return Chatbot()


try:
    bot = load_chatbot()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

st.title("Cognevance AI Chatbot")
st.caption("Intent recognition and response generation with NLTK + scikit-learn")

with st.sidebar:
    st.subheader("Model")
    st.metric("Test accuracy", f"{bot.metadata['accuracy'] * 100:.1f}%")
    left, right = st.columns(2)
    left.metric("Intents", bot.metadata["num_intents"])
    right.metric("Samples", f"{bot.metadata['num_samples']:,}")
    st.caption(
        "Trained: "
        + bot.metadata["trained_at"][:19].replace("T", " ")
        + " UTC"
    )
    st.divider()
    bot.threshold = st.slider(
        "Confidence threshold", 0.05, 0.95, bot.threshold, 0.05
    )
    if st.button("Clear chat"):
        st.session_state.messages = [WELCOME]
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [WELCOME]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("meta"):
            st.caption(message["meta"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    result = bot.predict(prompt)
    if result["intent"]:
        meta = (
            f"intent: `{result['intent']}` | "
            f"confidence: {result['confidence']:.2f}"
        )
    else:
        meta = "fallback: intent not recognized"

    with st.chat_message("assistant"):
        st.markdown(result["response"])
        st.caption(meta)

    st.session_state.messages.append(
        {"role": "assistant", "content": result["response"], "meta": meta}
    )
