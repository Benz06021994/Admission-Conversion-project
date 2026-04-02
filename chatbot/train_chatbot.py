import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "chatbot", "intents.csv")
MODEL_PATH = os.path.join(BASE_DIR, "chatbot", "chatbot_intent_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "chatbot", "chatbot_vectorizer.pkl")

def main():
    df = pd.read_csv(DATA_PATH)

    text_col = "text"
    intent_col = "intent"

    df[text_col] = df[text_col].astype(str).str.strip()
    df[intent_col] = df[intent_col].astype(str).str.strip()

    X = df[text_col]
    y = df[intent_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 3),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode"
    )

    model = LogisticRegression(
        max_iter=3000,
        C=2.0,
        class_weight="balanced",
        solver="liblinear"
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)

    print("Holdout Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds, zero_division=0))

    cv_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode"
        )),
        ("clf", LogisticRegression(
            max_iter=3000,
            C=2.0,
            class_weight="balanced",
            solver="liblinear"
        ))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(cv_pipeline, X, y, cv=cv, scoring="accuracy")

    print("\n5-Fold CV Accuracy Scores:", cv_scores)
    print("Mean CV Accuracy:", cv_scores.mean())

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("\nSaved:")
    print(MODEL_PATH)
    print(VECTORIZER_PATH)

if __name__ == "__main__":
    main()