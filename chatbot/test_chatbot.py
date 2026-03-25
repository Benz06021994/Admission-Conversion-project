from .chatbot_nlp import get_chatbot_response

print("Chatbot test started. Type 'exit' to stop.\n")

while True:
    user_message = input("You: ").strip()
    if user_message.lower() in ["exit", "quit"]:
        print("Chatbot test ended.")
        break

    response = get_chatbot_response(user_message)
    print("Bot:", response)