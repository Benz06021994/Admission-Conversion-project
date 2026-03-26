# NLP Chatbot Module

## Overview
This module provides an NLP-based chatbot for the Admission Conversion Prediction System. It uses TF-IDF vectorization and Logistic Regression for intent classification.

---

## Features
- Intent classification using machine learning
- Handles both system-level and business-level queries
- Supports placement officer decision-making
- Returns response, intent, and confidence

---

## Model Details
- TF-IDF Vectorizer
- Logistic Regression (selected after comparison with Naive Bayes)
- Trained on custom intent dataset

---

## How to Use

### Simple Usage

```python
from chatbot.chatbot_nlp import get_chatbot_response

response = get_chatbot_response(user_input)