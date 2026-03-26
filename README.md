# Admission Conversion Prediction System

## 📌 Overview
This project predicts the probability of converting a lead into an admission using Machine Learning. It also includes an NLP-based chatbot to assist users in understanding the system and interacting with it.

---

## 🎯 Business Problem
Educational institutes receive a large number of leads, but not all convert into admissions.  
This system helps:
- Identify high-potential leads
- Prioritize follow-ups
- Improve conversion efficiency
- Supports business decision-making for placement officers

---

## ⚙️ Features

### 1. Admission Prediction Model
- Predicts conversion probability (0–100%)
- Uses structured CRM data
- Supports decision-making for sales teams

### 2. NLP Chatbot
- Answers user queries intelligently
- Built using:
  - TF-IDF Vectorizer
  - Logistic Regression (selected after comparison with Naive Bayes)
- Handles both:
  - System-level queries (inputs, prediction, dashboard)
  - Business-level queries (lead priority, follow-up strategy, score interpretation)
- Returns:
  - Response (for UI)
  - Intent and confidence (for backend/database logging)

---

## 🧠 How It Works

### Prediction Model
1. User provides input features:
   - Contact Owner
   - Track Interested
   - District
   - Source of Lead
   - Course
   - Specialization
   - Gender

2. Model processes inputs and outputs:
   👉 Conversion Probability

---

### Chatbot Flow
User Query → TF-IDF Vectorization → Logistic Regression → Intent → Response + Confidence

---

## 🛠️ Tech Stack
- Python
- Scikit-learn
- Pandas, NumPy
- Flask (for integration)
- PostgreSQL (for storage)

---

## 📁 Project Structure
```
chatbot/
├── chatbot_nlp.py
├── chatbot_responses.py
├── intents.csv
├── train_chatbot.py
git├── chatbot_intent_model.pkl

crm_model.py
requirements.txt
```

---

## ▶️ How to Run Chatbot

```bash
python -m chatbot.test_chatbot