import os
import sys
import random
import re
import joblib
import pandas as pd
import __main__

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Make project root importable so crm_model.py can be found
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# IMPORTANT:
# Your saved admission pipeline depends on CRMFeatureBuilder.
from crm_model import CRMFeatureBuilder

# Expose under __main__ because the saved pipeline expects it there
__main__.CRMFeatureBuilder = CRMFeatureBuilder

CHATBOT_MODEL_PATH = os.path.join(BASE_DIR, "chatbot", "chatbot_intent_model.pkl")
CHATBOT_VECTORIZER_PATH = os.path.join(BASE_DIR, "chatbot", "chatbot_vectorizer.pkl")
ADMISSION_MODEL_PATH = os.path.join(BASE_DIR, "crm_output_v2", "artifacts", "best_pipeline.pkl")

# =========================
# Load models
# =========================
intent_model = joblib.load(CHATBOT_MODEL_PATH)
vectorizer = joblib.load(CHATBOT_VECTORIZER_PATH)
admission_model = joblib.load(ADMISSION_MODEL_PATH)

# =========================
# Static responses
# =========================
RESPONSES = {
    "greeting": [
        "Hello! How can I help you with admission conversion today?",
        "Hi! I can help with lead queries and conversion prediction."
    ],
    "goodbye": [
        "Goodbye!",
        "See you later!"
    ],
    "thanks": [
        "You're welcome!",
        "Glad to help."
    ],
    "project_overview": [
        "This system predicts admission conversion probability and also supports chatbot-based assistance for placement officers."
    ],
    "required_inputs": [
        "The model requires these inputs: Contact Owner, Track Interested, District, Source of lead, Course, Specialization, and Gender."
    ],
    "how_to_predict": [
        "You can ask me to predict a lead, and I will collect the required details one by one."
    ],
    "conversion_probability": [
        "The conversion probability indicates how likely a lead is to convert based on historical patterns."
    ],
    "model_used": [
        "The chatbot uses TF-IDF with Logistic Regression for intent classification, and the admission model predicts conversion probability."
    ],
    "database_info": [
        "Prediction and chatbot activity can be logged into PostgreSQL for analytics and future improvements."
    ],
    "dashboard_help": [
        "The dashboard can show lead trends, conversion rates, and performance breakdowns using stored prediction data."
    ],
    "followup_strategy": [
        "High-probability leads can be prioritized for immediate follow-up, while medium-probability leads may need nurturing."
    ],
    "lead_priority": [
        "Leads with higher conversion probability should be prioritized first for calls and follow-up."
    ],
    "model_trust": [
        "The model is trained on historical CRM data and provides probability-based decision support."
    ],
    "score_interpretation": [
        "A higher score means a higher estimated chance of admission conversion."
    ],
    "source_performance": [
        "Lead source performance can be analyzed using dashboard reports and historical conversion data."
    ],
    "system_usage": [
        "You can use this chatbot to ask questions, understand the system, and also predict lead conversion."
    ],
    "ask_lead_details": [
        "Please provide the lead details so I can predict the admission conversion probability."
    ],
    "prediction_cancelled": [
        "Prediction process cancelled. You can start again anytime."
    ],
    "fallback": [
        "Sorry, I could not understand that clearly. Please try again or ask about prediction, lead priority, follow-up strategy, or model usage."
    ]
}

# =========================
# Prediction intent aliases
# =========================
PREDICTION_INTENTS = {
    "predict_conversion",
    "lead_prediction",
    "check_conversion",
    "predict_lead"
}

# =========================
# Required model fields
# =========================
REQUIRED_FIELDS = [
    "Contact Owner",
    "Track Interested",
    "District",
    "Source of lead",
    "Course",
    "Specialization",
    "Gender"
]

# =========================
# In-memory user sessions
# =========================
USER_SESSIONS = {}

# =========================
# Helpers
# =========================
def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_lead_data(lead_data: dict) -> dict:
    cleaned = {}
    for field in REQUIRED_FIELDS:
        value = lead_data.get(field, "Unspecified") if lead_data else "Unspecified"
        if value is None or str(value).strip() == "":
            value = "Unspecified"
        cleaned[field] = str(value).strip()
    return cleaned


def predict_intent(user_input: str):
    cleaned = clean_text(user_input)
    X = vectorizer.transform([cleaned])

    pred = intent_model.predict(X)[0]
    probs = intent_model.predict_proba(X)[0]

    class_index = list(intent_model.classes_).index(pred)
    confidence = float(probs[class_index])

    return pred, confidence


def get_random_response(intent: str) -> str:
    if intent in RESPONSES:
        return random.choice(RESPONSES[intent])
    return random.choice(RESPONSES["fallback"])


def interpret_probability(prob: float) -> str:
    if prob >= 0.75:
        return "This lead has a high chance of conversion and can be prioritized immediately."
    elif prob >= 0.45:
        return "This lead has a moderate chance of conversion and may benefit from timely follow-up."
    else:
        return "This lead currently has a lower chance of conversion and may require nurturing or a different follow-up strategy."


def predict_admission_conversion(lead_data: dict):
    normalized = normalize_lead_data(lead_data)
    df = pd.DataFrame([normalized])
    prob = float(admission_model.predict_proba(df)[0][1])

    return {
        "probability": round(prob, 4),
        "percentage": f"{prob * 100:.2f}%",
        "interpretation": interpret_probability(prob),
        "lead_data_used": normalized
    }


# =========================
# Session helpers
# =========================
def start_prediction_session(user_id: str):
    USER_SESSIONS[user_id] = {
        "mode": "prediction",
        "current_index": 0,
        "lead_data": {}
    }


def cancel_prediction_session(user_id: str):
    if user_id in USER_SESSIONS:
        del USER_SESSIONS[user_id]


def is_prediction_session_active(user_id: str) -> bool:
    return user_id in USER_SESSIONS and USER_SESSIONS[user_id].get("mode") == "prediction"


def handle_prediction_session(user_id: str, user_input: str):
    session = USER_SESSIONS[user_id]
    current_index = session["current_index"]

    # Store the current answer in the correct field
    if current_index < len(REQUIRED_FIELDS):
        field = REQUIRED_FIELDS[current_index]
        session["lead_data"][field] = user_input.strip()
        session["current_index"] += 1

    # Ask the next field if more are pending
    if session["current_index"] < len(REQUIRED_FIELDS):
        next_field = REQUIRED_FIELDS[session["current_index"]]

        return {
            "response": f"Please provide {next_field}.",
            "intent": "predict_conversion",
            "confidence": 1.0,
            "requires_lead_data": True,
            "current_field": next_field,
            "collected_data": session["lead_data"]
        }

    # All fields collected -> predict
    pred_result = predict_admission_conversion(session["lead_data"])

    # End session
    cancel_prediction_session(user_id)

    return {
        "response": f"Predicted admission conversion probability is {pred_result['percentage']}. {pred_result['interpretation']}",
        "intent": "predict_conversion",
        "confidence": 1.0,
        "requires_lead_data": False,
        "prediction": pred_result
    }


# =========================
# Main chatbot function
# =========================
def get_chatbot_response(user_input: str, user_id: str = "default_user"):
    message = clean_text(user_input)

    # Cancel/reset during active prediction flow
    if message in {"cancel", "stop", "reset", "exit"} and is_prediction_session_active(user_id):
        cancel_prediction_session(user_id)
        return {
            "response": get_random_response("prediction_cancelled"),
            "intent": "cancel_prediction",
            "confidence": 1.0
        }

    # Continue existing prediction session
    if is_prediction_session_active(user_id):
        return handle_prediction_session(user_id, user_input)

    # Normal intent classification
    intent, confidence = predict_intent(user_input)

    # Lowered fallback threshold
    if confidence < 0.15:
        return {
            "response": get_random_response("fallback"),
            "intent": "fallback",
            "confidence": round(confidence, 4)
        }

    # Start step-by-step prediction flow
    if intent in PREDICTION_INTENTS:
        start_prediction_session(user_id)
        return {
            "response": f"Please provide {REQUIRED_FIELDS[0]}.",
            "intent": "predict_conversion",
            "confidence": round(confidence, 4),
            "requires_lead_data": True,
            "current_field": REQUIRED_FIELDS[0],
            "collected_data": {}
        }

    # Standard chatbot response
    return {
        "response": get_random_response(intent),
        "intent": intent,
        "confidence": round(confidence, 4)
    }