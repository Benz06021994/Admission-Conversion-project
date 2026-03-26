from .chatbot_nlp import get_chatbot_result

print("Chatbot test started. Type 'exit' to stop.\n")

while True:
    user_message = input("You: ").strip()
    if user_message.lower() in ["exit", "quit"]:
        print("Chatbot test ended.")
        break

    result = get_chatbot_result(user_message)
    print("Bot:", result["response"])
    print("Intent:", result["intent"])
    print("Confidence:", round(result["confidence"], 4))
    print()