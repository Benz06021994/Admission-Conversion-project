import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "intents.csv")
MODEL_PATH = os.path.join(BASE_DIR, "chatbot_intent_model.pkl")

# =========================
# Load data
# =========================
df = pd.read_csv(DATA_PATH)

if "text" not in df.columns or "intent" not in df.columns:
    raise ValueError("intents.csv must contain 'text' and 'intent' columns")

X = df["text"].astype(str)
y = df["intent"].astype(str)

if len(df) < 10:
    raise ValueError("Training data is too small. Add more intent examples.")

# =========================
# Models to compare
# Naive Bayes is kept second intentionally.
# If both models get same accuracy, Naive Bayes will be selected.
# =========================
models = [
    ("Logistic Regression", LogisticRegression(max_iter=1000)),
    ("Multinomial Naive Bayes", MultinomialNB())
]

best_model_name = None
best_pipeline = None
best_accuracy = -1.0

print("Training and evaluating models on full dataset...\n")

for model_name, clf in models:
    print("=" * 60)
    print(f"Training: {model_name}")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )),
        ("clf", clf)
    ])

    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)

    acc = accuracy_score(y, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y, y_pred, zero_division=0))

    # Select better model
    # If tie, prefer Naive Bayes
    if acc > best_accuracy:
        best_accuracy = acc
        best_model_name = model_name
        best_pipeline = pipeline

    elif acc == best_accuracy and model_name == "Multinomial Naive Bayes":
        best_accuracy = acc
        best_model_name = model_name
        best_pipeline = pipeline

print("=" * 60)
print(f"Final selected model: {best_model_name}")
print(f"Best accuracy: {best_accuracy:.4f}")

joblib.dump(best_pipeline, MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")