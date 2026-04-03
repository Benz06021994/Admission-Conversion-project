# 🎓 Admission Conversion Prediction System with NLP Chatbot

---

## 📌 Overview

This project predicts **admission conversion probability** and provides an **NLP-based chatbot interface** for placement officers.

It enables:
- 📊 Data-driven decision making  
- 🎯 Lead prioritization  
- 🤖 Conversational interaction  

---

## 🧠 Key Features

### 🔹 Machine Learning Model
- Predicts **conversion probability (not binary output)**
- Uses custom feature engineering via `CRMFeatureBuilder`
- Handles **class imbalance effectively**
- Outputs probability for better business decisions

---

### 🔹 NLP Chatbot
- Built using **TF-IDF + Logistic Regression**
- Supports **multiple business intents**
- Returns:
  - Response
  - Intent
  - Confidence score

---

### 🔹 Step-by-Step Prediction (Conversational AI)

The chatbot collects lead details **one by one**:

1. Contact Owner  
2. Track Interested  
3. District  
4. Source of lead  
5. Course  
6. Specialization  
7. Gender  

👉 After collecting all inputs, it predicts conversion probability.

---

### 🔹 Session Handling
- Maintains conversation state using session memory
- Supports:
  - Multi-step interaction  
  - Cancel / Reset (`cancel`, `stop`, `reset`)

---

## 📂 Project Structure

```text
chatbot/
├── chatbot_nlp.py          # Chatbot logic
├── chatbot_intent_model.pkl
├── chatbot_vectorizer.pkl
├── intents.csv

crm_output_v2/artifacts/
└── best_pipeline.pkl       # ML model

crm_model.py                # Feature engineering class
```

---

## ⚙️ How to Use

### 🔹 Function

```python
get_chatbot_response(message, user_id)
```

---

### 🔹 Input

- `message`: User input text  
- `user_id`: Unique session identifier (**important for step-by-step flow**)

---

### 🔹 Output

Returns:

- response  
- intent  
- confidence  
- requires_lead_data  
- prediction (after all inputs collected)

---

## 🔁 Example Flow

User: predict this lead  
Bot: Please provide Contact Owner  

User: Sales Person 1  
Bot: Please provide Track Interested  

...  

Bot: Predicted conversion probability: 78%  

---

## 🗄️ Database Integration (Planned)

### Tables:
- chatbot_logs  
- predictions  

Used for:
- Analytics  
- Dashboard  
- Performance tracking  

---

## 🚀 Deployment

Recommended platform:

👉 **Render (PaaS)**  
- Easy Flask deployment  
- PostgreSQL support  
- Free tier available for starters

---

## 📊 Model Performance

- 🎯 Accuracy: ~90%  
- 🔁 Cross-validation: ~85%  

---

## 🎯 Business Value

- Helps **prioritize leads**  
- Improves **conversion rate**  
- Enables **data-driven decisions**  

---

## 🔮 Future Improvements

- Store sessions in database  
- Improve UI (progress tracking)  
- Continuous model retraining  

---
