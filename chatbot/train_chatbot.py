import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "intents.csv")
MODEL_PATH = os.path.join(BASE_DIR, "chatbot_intent_model.pkl")

df = pd.read_csv(DATA_PATH)

if "text" not in df.columns or "intent" not in df.columns:
    raise ValueError("intents.csv must contain 'text' and 'intent' columns")

X = df["text"].astype(str)
y = df["intent"].astype(str)

if len(df) < 10:
    raise ValueError("Training data is too small. Add more intent examples.")

num_classes = y.nunique()
min_class_count = y.value_counts().min()

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000))
])

# If dataset is too small for stratified split, train on full data
if len(df) < 50 or min_class_count < 2:
    print("Dataset is small, so training on full data without train/test split.")
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
else:
    test_size = max(num_classes, int(0.2 * len(df)))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")