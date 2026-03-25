import os
import joblib
from chatbot.chatbot_responses import RESPONSES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "chatbot_intent_model.pkl")

model = joblib.load(MODEL_PATH)

def predict_intent(user_message: str):
    predicted_intent = model.predict([user_message])[0]

    confidence = 1.0
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([user_message])[0]
        confidence = max(probs)

    return predicted_intent, confidence

def get_chatbot_response(user_message: str, threshold: float = 0.10) -> str:
    if not user_message or not user_message.strip():
        return "Please type a message."

    intent, confidence = predict_intent(user_message)

    if confidence < threshold:
        return RESPONSES["fallback"]

    return RESPONSES.get(intent, RESPONSES["fallback"])