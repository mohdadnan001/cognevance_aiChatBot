"""Merge the raw Kaggle datasets into a single unified intents file.

Sources:
  * Bitext Gen AI Chatbot Customer Support Dataset (27 intents, ~27k samples)
  * Chatbots: Intent Recognition Dataset by elvinagammed (conversational intents)

The output schema matches the common chatbot `intents.json` format:
    {"intents": [{"tag": str, "patterns": [str], "responses": [str]}]}
"""

import json
from collections import Counter

import pandas as pd

from src.config import BITEXT_PATH, GENISYS_PATH, INTENTS_PATH

BITEXT_SOURCE = (
    "https://www.kaggle.com/datasets/bitext/"
    "bitext-gen-ai-chatbot-customer-support-dataset"
)
GENISYS_SOURCE = (
    "https://www.kaggle.com/datasets/elvinagammed/"
    "chatbots-intent-recognition-dataset"
)

GENISYS_INTENTS = {
    "Greeting": "greeting",
    "GreetingResponse": "greeting_response",
    "CourtesyGreeting": "courtesy_greeting",
    "CourtesyGreetingResponse": "courtesy_greeting_response",
    "CurrentHumanQuery": "current_human_query",
    "NameQuery": "name_query",
    "RealNameQuery": "real_name_query",
    "TimeQuery": "time_query",
    "Thanks": "thanks",
    "GoodBye": "goodbye",
    "CourtesyGoodBye": "courtesy_goodbye",
    "Jokes": "jokes",
    "Clever": "clever",
    "UnderstandQuery": "understand_query",
    "WhoAmI": "who_am_i",
    "SelfAware": "self_aware",
}

GENISYS_RESPONSE_OVERRIDES = {
    "greeting": [
        "Hello! How can I help you today?",
        "Hi there! What can I do for you?",
    ],
    "greeting_response": [
        "Great to meet you! How can I help?",
        "Nice to meet you! What can I do for you?",
    ],
    "courtesy_greeting": [
        "I'm doing well, thanks for asking! How can I help you?",
        "All good here! What can I help you with?",
    ],
    "courtesy_greeting_response": [
        "Glad to hear it! How can I help you today?",
        "That's good to hear. What can I do for you?",
    ],
    "current_human_query": [
        "I can't access your personal details, but I'm happy to help.",
        "I can't see personal information, but feel free to tell me your name.",
    ],
    "name_query": [
        "I'm Cognevance, your AI assistant.",
        "You can call me Cognevance.",
    ],
    "real_name_query": [
        "My full name is Cognevance AI Chatbot.",
        "I'm Cognevance, an NLP-powered assistant.",
    ],
    "time_query": [
        "I don't have a live clock, but I can tell you the time if you ask again in the app.",
        "Sorry, I can't check the time directly, but your device clock can.",
    ],
    "thanks": [
        "You're welcome!",
        "No problem at all, happy to help!",
    ],
    "goodbye": [
        "Goodbye! Have a great day.",
        "See you later!",
    ],
    "courtesy_goodbye": [
        "Thanks for chatting, goodbye!",
        "Take care and goodbye!",
    ],
    "clever": [
        "Thank you! I was trained with plenty of NLP data.",
        "That's kind of you to say!",
    ],
    "understand_query": [
        "Yes, I use NLP to understand the intent behind your message.",
        "I do my best to understand you using intent recognition.",
    ],
    "who_am_i": [
        "I can't see you, but I'm always happy to chat.",
        "I can't access a camera, but I'm here to help.",
    ],
    "self_aware": [
        "That's a deep question! I'm aware enough to recognize your intent.",
        "I'll leave the philosophy to you!",
    ],
}

BITEXT_RESPONSE_OVERRIDES = {
    "cancel_order": [
        "I can help you cancel an order. Please share your order number.",
        "Sure, let's cancel that order. Could you provide the order number?",
    ],
    "change_order": [
        "I can help you change your order. Please tell me your order number and what you'd like to change.",
        "What would you like to change about your order? Please include the order number.",
    ],
    "change_shipping_address": [
        "I can update your shipping address. Please provide the order number and the new address.",
        "Let's change the shipping address. What's the order number?",
    ],
    "check_cancellation_fee": [
        "I can check whether a cancellation fee applies. Please share your order number.",
        "Sure, provide your order details and I'll check any cancellation fees.",
    ],
    "check_invoice": [
        "I can help you check your invoice. Could you share the order or invoice number?",
        "Sure, please provide your invoice or order number and I'll look it up.",
    ],
    "check_payment_methods": [
        "We accept major credit cards, debit cards and common online payment options.",
        "You can pay by credit card, debit card or one of our supported online methods.",
    ],
    "check_refund_policy": [
        "Our refund policy allows returns within the return window if the item meets our conditions.",
        "I can explain our refund policy. Items are generally refundable within the return window.",
    ],
    "complaint": [
        "I'm sorry to hear that. Please tell me what went wrong and I'll do my best to help.",
        "I understand your frustration. Could you share more details so we can resolve this?",
    ],
    "contact_customer_service": [
        "You can reach our customer service team via live chat, email or phone. Which would you prefer?",
        "I can connect you with customer service. Please tell me what you need help with.",
    ],
    "contact_human_agent": [
        "I can connect you to a human agent. Please hold on while I transfer you.",
        "Sure, let me forward you to a human agent who can help further.",
    ],
    "create_account": [
        "I can help you create an account. Please provide the email you'd like to use.",
        "Let's set up your account. What email address should we use?",
    ],
    "delete_account": [
        "I can help you delete your account. Please confirm the email associated with it.",
        "Are you sure you want to delete your account? Please confirm your email.",
    ],
    "delivery_options": [
        "We offer standard and express delivery options. Which would you prefer?",
        "You can choose standard or express shipping at checkout. Would you like details?",
    ],
    "delivery_period": [
        "Delivery usually takes a few business days depending on your location. Do you have an order number?",
        "I can check the estimated delivery period for you. Please share your order number.",
    ],
    "edit_account": [
        "I can help you update your account details. What would you like to change?",
        "Sure, let's edit your account. Which information needs updating?",
    ],
    "get_invoice": [
        "I can send you your invoice. Please provide the order number or purchase email.",
        "Sure, share your order number and I'll retrieve the invoice for you.",
    ],
    "get_refund": [
        "I can help you request a refund. Please share your order number and the reason.",
        "Sure, let's start your refund. What's the order number?",
    ],
    "newsletter_subscription": [
        "I can help you subscribe to or unsubscribe from our newsletter. What would you like to do?",
        "Would you like to subscribe or unsubscribe from our newsletter?",
    ],
    "payment_issue": [
        "I'm sorry you're having a payment issue. Please tell me what happened and I'll help.",
        "Let's resolve your payment issue. Can you share the error you see?",
    ],
    "place_order": [
        "I can help you place an order. What would you like to buy?",
        "Let's get your order placed. Tell me which item you'd like.",
    ],
    "recover_password": [
        "I can help you recover your password. Please provide the email linked to your account.",
        "Let's reset your password. What email did you register with?",
    ],
    "registration_problems": [
        "I can help with registration problems. What happens when you try to register?",
        "Sorry for the trouble signing up. Please describe the issue and I'll guide you.",
    ],
    "review": [
        "Thank you for your feedback! Would you like to leave a review?",
        "We'd love your feedback. Please tell us about your experience.",
    ],
    "set_up_shipping_address": [
        "I can set up a shipping address for you. Please provide the full address.",
        "Let's add your shipping address. Please share the street, city and postal code.",
    ],
    "switch_account": [
        "I can help you switch accounts. Please tell me which account you'd like to use.",
        "Sure, which account would you like to switch to?",
    ],
    "track_order": [
        "I can track your order. Please share your order number.",
        "Sure, give me your order number and I'll check its status.",
    ],
    "track_refund": [
        "I can check your refund status. Please share the order or refund reference number.",
        "Sure, provide your order number and I'll track the refund for you.",
    ],
}


# Hand-written seed phrases that make the assistant robust for common queries.
EXTRA_PATTERNS = {
    "greeting": [
        "hey",
        "hey there",
        "good morning",
        "good afternoon",
        "good evening",
        "howdy",
    ],
    "goodbye": [
        "bye",
        "bye bye",
        "goodbye",
        "see you later",
        "talk to you later",
        "farewell",
        "catch you later",
    ],
    "thanks": [
        "thank you",
        "thanks a lot",
        "many thanks",
        "much appreciated",
        "appreciate it",
    ],
    "courtesy_greeting": [
        "how are you doing",
        "how is it going",
        "what is up",
        "how do you do",
    ],
    "name_query": ["who are you", "what are you called"],
    "time_query": ["what time is it", "current time", "do you know the time"],
    "jokes": ["make me laugh", "say something funny", "know any good jokes"],
    "recover_password": [
        "i cannot log in",
        "i can't log in",
        "i can't sign in",
        "unable to login",
        "login problem",
        "i forgot my login details",
        "i can't access my account",
    ],
    "track_order": [
        "where is my order",
        "where is my package",
        "has my order shipped",
        "when will my order arrive",
    ],
    "cancel_order": ["i want to cancel my order", "cancel my purchase"],
    "get_refund": [
        "i want a refund",
        "i need a refund",
        "refund my order",
        "i want my money back",
        "i would like a refund",
        "give me my money back",
    ],
    "payment_issue": [
        "my payment failed",
        "my card was declined",
        "payment did not go through",
    ],
    "contact_human_agent": [
        "let me talk to a human",
        "i need a real person",
        "connect me to an agent",
    ],
    "delivery_period": [
        "how long will delivery take",
        "when will it be delivered",
    ],
    "change_shipping_address": [
        "i need to change my address",
        "update my delivery address",
    ],
    "capabilities": [
        "what can you do",
        "what are your features",
        "how can you help me",
        "what do you do",
        "show me what you can do",
        "what are your capabilities",
    ],
}

EXTRA_RESPONSES = {
    "capabilities": [
        "I can help with orders, refunds, invoices, payments, accounts, "
        "shipping and general chat. Try asking 'where is my order?' or "
        "'I want a refund'."
    ],
}


def clean_text(value: str) -> str:
    return str(value).replace("\ufffd", "'").strip()


def load_genisys() -> list[dict]:
    payload = json.loads(GENISYS_PATH.read_text(encoding="utf-8"))
    intents = []
    for item in payload["intents"]:
        tag = GENISYS_INTENTS.get(item["intent"])
        if not tag:
            continue
        patterns = [
            clean_text(pattern)
            for pattern in item["text"]
            if len(clean_text(pattern)) > 1
        ]
        responses = GENISYS_RESPONSE_OVERRIDES.get(tag)
        if not responses:
            responses = [
                clean_text(response)
                for response in item["responses"]
                if len(clean_text(response)) > 1
            ]
        intents.append(
            {"tag": tag, "patterns": sorted(set(patterns)), "responses": responses}
        )
    return intents


def load_bitext() -> list[dict]:
    df = pd.read_csv(BITEXT_PATH)
    df = df.dropna(subset=["instruction", "intent"])
    intents = []
    for intent, group in df.groupby("intent", sort=True):
        patterns = sorted(
            {
                clean_text(pattern)
                for pattern in group["instruction"].unique()
                if len(clean_text(pattern)) > 1
            }
        )
        responses = BITEXT_RESPONSE_OVERRIDES.get(intent)
        if not responses:
            responses = [
                clean_text(response)
                for response in group["response"].dropna().unique()[:3]
            ]
        intents.append(
            {"tag": intent, "patterns": patterns, "responses": responses}
        )
    return intents


def apply_extras(intents: list[dict]) -> list[dict]:
    by_tag = {intent["tag"]: intent for intent in intents}
    for tag, patterns in EXTRA_PATTERNS.items():
        intent = by_tag.get(tag)
        if intent is None:
            intent = {
                "tag": tag,
                "patterns": [],
                "responses": EXTRA_RESPONSES.get(tag, []),
            }
            intents.append(intent)
            by_tag[tag] = intent
        intent["patterns"] = sorted(set(intent["patterns"]) | set(patterns))
    return intents


def main() -> None:
    intents = apply_extras(load_genisys() + load_bitext())

    seen = set()
    for intent in intents:
        if intent["tag"] in seen:
            raise ValueError(f"Duplicate intent tag: {intent['tag']}")
        seen.add(intent["tag"])

    pattern_counts = Counter()
    for intent in intents:
        pattern_counts[intent["tag"]] = len(intent["patterns"])

    payload = {
        "metadata": {
            "description": "Unified intents dataset for the Cognevance AI Chatbot",
            "sources": [
                {"name": "Bitext customer support", "url": BITEXT_SOURCE},
                {"name": "Chatbots intent recognition", "url": GENISYS_SOURCE},
            ],
            "num_intents": len(intents),
            "num_patterns": sum(pattern_counts.values()),
        },
        "intents": intents,
    }

    INTENTS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Wrote {INTENTS_PATH}")
    print(f"Intents: {len(intents)}")
    print(f"Patterns: {sum(pattern_counts.values())}")
    print(f"Smallest intents: {pattern_counts.most_common()[-5:]}")


if __name__ == "__main__":
    main()
