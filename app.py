import io
import os
import sqlite3
from datetime import datetime

import sklearn.compose._column_transformer as ct

class _RemainderColsList(list):
    pass

ct._RemainderColsList = _RemainderColsList

import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, send_file, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash

from crm_model import CRMFeatureBuilder
from chatbot.chatbot_nlp import get_chatbot_result
from flask import jsonify, request

app = Flask(__name__)
app.secret_key = "crm_secret_key"


PIPELINE_PATH = os.path.join("crm_output_v2", "artifacts", "best_pipeline.pkl")
pipeline = joblib.load(PIPELINE_PATH)


# sklearn compatibility fix
try:
    for step in pipeline.named_steps.values():
        if not hasattr(step, "_name_to_fitted_passthrough"):
            setattr(step, "_name_to_fitted_passthrough", {})
except:
    pass


DB_PATH = "predictions.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        userid TEXT UNIQUE,
        username TEXT,
        email TEXT UNIQUE,
        password TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        lead_id TEXT,   
        userid TEXT,
        contact_owner TEXT,
        track_interested TEXT,
        district TEXT,
        source_of_lead TEXT,
        course TEXT,
        specialization TEXT,
        gender TEXT,
        conversion_probability REAL,
        score REAL,
        converted INTEGER DEFAULT 0,
        predicted_at TEXT
    )
    """)

    conn.commit()
    conn.close()


def generate_userid(username):

    prefix = username[:3].lower()

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM users WHERE username=?", (username,))
    count = cursor.fetchone()[0] + 1

    conn.close()

    return f"{prefix}0{count}"


def save_prediction(userid, data, prob, score):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO predictions(
    lead_id, userid, contact_owner, track_interested, district,
    source_of_lead, course, specialization, gender,
    conversion_probability, score, converted, predicted_at)
    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        data["Lead ID"],  # ✅ NEW
        userid,
        data["Contact Owner"],
        data["Track Interested"],
        data["District"],
        data["Source of lead"],
        data["Course"],
        data["Specialization"],
        data["Gender"],
        prob,
        score,
        0,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


# ✅ FINAL BULLETPROOF DROPDOWN LOADER
try:
    dropdown_path = "Unique values to make front end drop down menu.xlsx"

    df_dropdown = pd.read_excel(dropdown_path)

    # Clean column names
    df_dropdown.columns = df_dropdown.columns.str.strip()

    print("✅ Dropdown Columns:", df_dropdown.columns.tolist())

    # 🔥 DEBUG FIRST ROW
    print("Sample Data:\n", df_dropdown.head())

    # SAFE ACCESS (no crash)
    def get_values(col):
        if col in df_dropdown.columns:
            return sorted(df_dropdown[col].dropna().astype(str).unique())
        else:
            print(f"❌ Missing column: {col}")
            return []

    contact_owners = get_values("Contact Owner")
    track_interested_list = get_values("Track Interested")
    districts = get_values("District")
    source_of_lead_list = get_values("Source of lead")
    courses = get_values("Course")
    specializations = get_values("Specialization")
    genders = get_values("Gender")

    print("✅ Loaded dropdown counts:",
          len(contact_owners),
          len(track_interested_list),
          len(districts),
          len(source_of_lead_list),
          len(courses),
          len(specializations),
          len(genders)
    )

except Exception as e:
    print("❌ Dropdown load error:", e)

    contact_owners = []
    track_interested_list = []
    districts = []
    source_of_lead_list = []
    courses = []
    specializations = []
    genders = []


@app.route("/login", methods=["GET", "POST"])
def login():

    message = None

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]
        role = request.form["role"]

        if role == "admin" and username == "admin" and password == "admin123":
            session["role"] = "admin"
            return redirect(url_for("home"))

        if role == "officer":

            conn = get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM users WHERE userid=?", (username,))
            user = cursor.fetchone()
            conn.close()

            if user and check_password_hash(user["password"], password):
                session["role"] = "officer"
                session["user"] = user["userid"]
                return redirect(url_for("home"))

            message = "Invalid UserID or Password"

    return render_template("login.html", message=message)


@app.route("/register", methods=["GET", "POST"])
def register():

    message = None

    if request.method == "POST":

        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        userid = generate_userid(username)
        hashed_password = generate_password_hash(password)

        conn = get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
            INSERT INTO users(userid,username,email,password)
            VALUES(?,?,?,?)
            """, (userid, username, email, hashed_password))

            conn.commit()

            message = f"Account created! Your UserID: {userid}"

        except:
            message = "User already exists"

        conn.close()

    return render_template("register.html", message=message)


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():

    message = None

    if request.method == "POST":

        email = request.form["email"]
        new_password = generate_password_hash(request.form["password"])

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("UPDATE users SET password=? WHERE email=?", (new_password, email))

        conn.commit()
        conn.close()

        message = "Password updated"

    return render_template("forgot_password.html", message=message)


@app.route("/")
def home():

    if "role" not in session:
        return redirect(url_for("login"))

    return render_template("home.html")


@app.route("/logout")
def logout():

    session.clear()
    return redirect(url_for("login"))


# ✅ FINAL FIXED PREDICT ROUTE
@app.route("/predict", methods=["GET", "POST"])
def predict():

    if "role" not in session:
        return redirect(url_for("login"))

    if session["role"] == "admin":
        return redirect(url_for("dashboard"))

    prediction = None

    if request.method == "POST":

        data = {
            "Lead ID": request.form.get("Lead ID"),  # ✅ NEW FIELD
            "Contact Owner": request.form.get("Contact Owner"),
            "Track Interested": request.form.get("Track Interested"),
            "District": request.form.get("District"),
            "Source of lead": request.form.get("Source of lead"),
            "Course": request.form.get("Course"),
            "Specialization": request.form.get("Specialization"),
            "Gender": request.form.get("Gender")
        }

        # ❗ IMPORTANT: exclude Lead ID from model input
        input_df = pd.DataFrame([{
            k: v for k, v in data.items() if k != "Lead ID"
        }])

        try:
            prob = float(pipeline.predict_proba(input_df)[0][1])
        except Exception as e:
            return f"Prediction Error: {str(e)}"

        score = round(prob * 100, 2)

        # ✅ SMART HINTS (unchanged logic)
        if prob >= 0.7:
            label = "High Chance"
            positive = "Strong lead. High probability of conversion."
            negative = "Ensure quick follow-up to close."
        elif prob >= 0.4:
            label = "Medium Chance"
            positive = "Potential lead. Needs nurturing."
            negative = "May drop without proper follow-up."
        else:
            label = "Low Chance"
            positive = "Low interest currently."
            negative = "Requires strong engagement strategy."

        # ✅ SAVE ONLY ONCE (FIXED duplicate)
        save_prediction(session["user"], data, prob, score)

        prediction = {
            "prob": score,
            "score": score,
            "label": label,      # ✅ added (optional for UI)
            "positive": positive,
            "negative": negative
        }

    return render_template(
        "predict.html",
        prediction=prediction,
        contact_owners=contact_owners,
        track_interested_list=track_interested_list,
        districts=districts,
        source_of_lead_list=source_of_lead_list,
        courses=courses,
        specializations=specializations,
        genders=genders
    )


@app.route("/toggle-conversion/<int:lead_id>", methods=["POST"])
def toggle_conversion(lead_id):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    UPDATE predictions
    SET converted = CASE WHEN converted=1 THEN 0 ELSE 1 END
    WHERE id=?
    """, (lead_id,))

    conn.commit()
    conn.close()

    return redirect(url_for("history"))


@app.route("/history")
def history():

    if "role" not in session:
        return redirect(url_for("login"))

    conn = get_connection()

    officer_filter = request.args.get("officer")
    lead_id_filter = request.args.get("lead_id")

    query = "SELECT * FROM predictions WHERE 1=1"
    params = []

    if session["role"] == "admin":

        if officer_filter:
            query += " AND userid=?"
            params.append(officer_filter)

    else:
        query += " AND userid=?"
        params.append(session["user"])

    if lead_id_filter:
        query += " AND lead_id LIKE ?"
        params.append(f"%{lead_id_filter}%")

    query += " ORDER BY id DESC"

    rows = conn.execute(query, params).fetchall()

    officers = None
    if session["role"] == "admin":
        officers = conn.execute(
            "SELECT DISTINCT userid FROM predictions"
        ).fetchall()

    conn.close()

    return render_template(
        "history.html",
        rows=rows,
        officers=officers
    )


@app.route("/dashboard")
def dashboard():

    if "role" not in session:
        return redirect(url_for("login"))

    conn = get_connection()

    officer_filter = request.args.get("officer")
    date_filter = request.args.get("date")
    month_filter = request.args.get("month")
    year_filter = request.args.get("year")

    query = "SELECT * FROM predictions WHERE 1=1"
    params = []

    # ======================
    # ROLE FILTER
    # ======================
    if session["role"] == "admin":
        if officer_filter:
            query += " AND userid=?"
            params.append(officer_filter)

        officers = conn.execute(
            "SELECT DISTINCT userid FROM predictions"
        ).fetchall()
    else:
        query += " AND userid=?"
        params.append(session["user"])
        officers = None

    # ======================
    # DATE FILTERS
    # ======================
    if date_filter:
        query += " AND DATE(predicted_at)=?"
        params.append(date_filter)

    if month_filter:
        query += " AND strftime('%Y-%m', predicted_at)=?"
        params.append(month_filter)

    if year_filter:
        query += " AND strftime('%Y', predicted_at)=?"
        params.append(year_filter)

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    # ======================
    # SAFE HANDLING
    # ======================
    if df.empty:
        total = converted = not_converted = avg_score = conversion_rate = 0
        high = medium = low = 0
        best_officer = "-"
        insight = "No data available"
    else:
        total = len(df)
        converted = int(df["converted"].sum())
        not_converted = total - converted
        avg_score = round(df["score"].mean(), 2)
        conversion_rate = round((converted / total) * 100, 2)

        # Lead quality
        high = len(df[df["score"] >= 70])
        medium = len(df[(df["score"] >= 40) & (df["score"] < 70)])
        low = len(df[df["score"] < 40])

        # Best officer
        best = df.groupby("userid")["converted"].sum().sort_values(ascending=False)
        best_officer = best.index[0] if not best.empty else "-"

        # AI Insight
        if conversion_rate > 60:
            insight = "🔥 Excellent conversion performance"
        elif conversion_rate > 30:
            insight = "⚡ Moderate performance, improve follow-ups"
        else:
            insight = "❄️ Low conversions, need strong strategy"

    return render_template(
        "dashboard.html",
        total=total,
        converted=converted,
        not_converted=not_converted,
        avg_score=avg_score,
        conversion_rate=conversion_rate,
        best_officer=best_officer,
        high=high,
        medium=medium,
        low=low,
        officers=officers,
        insight=insight
    )


@app.route("/dashboard-data")
def dashboard_data():

    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()

    # ======================
    # EXISTING LOGIC (KEEPED)
    # ======================

    # Monthly trend
    df["month"] = df["predicted_at"].str[:7]
    monthly = df.groupby("month")["score"].mean().reset_index()

    # Daily conversion rate
    df["date"] = df["predicted_at"].str[:10]
    daily = df.groupby("date")["converted"].mean().reset_index()

    # ======================
    # NEW ADDITIONS (CHARTS)
    # ======================

    # High / Medium / Low
    high = len(df[df["score"] >= 70])
    medium = len(df[(df["score"] >= 40) & (df["score"] < 70)])
    low = len(df[df["score"] < 40])

    # Course distribution
    if "Course" in df.columns:
        courses = df["Course"].value_counts().to_dict()
    else:
        courses = {}

    # Source distribution
    if "Source of lead" in df.columns:
        sources = df["Source of lead"].value_counts().to_dict()
    else:
        sources = {}

    # ======================
    # RETURN ALL DATA
    # ======================

    return jsonify({
        # OLD (keep working)
        "labels": df["predicted_at"].tolist(),
        "scores": df["score"].tolist(),
        "monthly_labels": monthly["month"].tolist(),
        "monthly_scores": monthly["score"].tolist(),
        "daily_labels": daily["date"].tolist(),
        "daily_conversion": (daily["converted"] * 100).tolist(),

        # NEW (for charts)
        "high": high,
        "medium": medium,
        "low": low,
        "courses": courses,
        "sources": sources
    })


@app.route("/export/csv")
def export_csv():

    conn = get_connection()
    officer_filter = request.args.get("officer")

    if officer_filter:
        df = pd.read_sql_query(
            "SELECT * FROM predictions WHERE userid=?",
            conn,
            params=(officer_filter,)
        )
    else:
        df = pd.read_sql_query("SELECT * FROM predictions", conn)

    conn.close()

    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="prediction_history.csv"
    )

@app.route("/edit/<int:id>", methods=["GET", "POST"])
def edit_lead(id):

    if "role" not in session:
        return redirect(url_for("login"))

    conn = get_connection()
    cursor = conn.cursor()

    if request.method == "POST":

        cursor.execute("""
        UPDATE predictions SET
            contact_owner=?,
            track_interested=?,
            district=?,
            source_of_lead=?,
            course=?,
            specialization=?,
            gender=?
        WHERE id=?
        """, (
            request.form.get("Contact Owner"),
            request.form.get("Track Interested"),
            request.form.get("District"),
            request.form.get("Source of lead"),
            request.form.get("Course"),
            request.form.get("Specialization"),
            request.form.get("Gender"),
            id
        ))

        conn.commit()
        conn.close()

        return redirect(url_for("history"))

    row = cursor.execute(
        "SELECT * FROM predictions WHERE id=?",
        (id,)
    ).fetchone()

    conn.close()

    # ✅ PASS DROPDOWNS
    return render_template(
        "edit.html",
        row=row,
        contact_owners=contact_owners,
        track_interested_list=track_interested_list,
        districts=districts,
        source_of_lead_list=source_of_lead_list,
        courses=courses,
        specializations=specializations,
        genders=genders
    )

@app.route("/delete/<int:id>")
def delete_lead(id):

    if "role" not in session:
        return redirect(url_for("login"))

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM predictions WHERE id=?", (id,))
    conn.commit()
    conn.close()

    return redirect(url_for("history"))


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    result = get_chatbot_result(user_message)

    return jsonify({
        "reply": result["response"]
    })

init_db()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)