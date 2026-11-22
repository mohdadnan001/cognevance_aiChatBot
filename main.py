"""Interactive command-line chat with the Cognevance AI Chatbot."""

from src.chatbot import Chatbot


def main() -> None:
    bot = Chatbot()
    print("=" * 60)
    print("Cognevance AI Chatbot")
    print(f"Intents: {bot.metadata['num_intents']} | "
          f"Test accuracy: {bot.metadata['accuracy'] * 100:.1f}%")
    print("Type 'quit', 'exit' or 'bye' to stop.")
    print("=" * 60)

    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not text:
            continue
        if text.lower() in {"quit", "exit", "bye"}:
            print("Bot: Goodbye! Have a great day.")
            break

        result = bot.predict(text)
        intent = result["intent"] or "fallback"
        print(f"Bot ({intent}, {result['confidence']:.2f}): {result['response']}")


if __name__ == "__main__":
    main()
