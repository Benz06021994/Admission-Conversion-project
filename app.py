import io
import os
from datetime import datetime

import joblib
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from crm_model import CRMFeatureBuilder  # kept because you already had it

load_dotenv()

app = Flask(__name__)

# -----------------------------
# Load model / pipeline
# -----------------------------
PIPELINE_PATH = "crm_output_v2/artifacts/best_pipeline.pkl"
pipeline = joblib.load(PIPELINE_PATH)

# -----------------------------
# Database config
# -----------------------------
# If .env has DATABASE_URL, it will use that.
# Otherwise it falls back to local SQLite for your machine.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///admission_app.db")

engine = create_engine(DATABASE_URL, future=True)
DB_DIALECT = engine.dialect.name  # 'sqlite', 'postgresql', etc.


def check_db_connection():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"DB connection failed: {e}")
        return False


# -----------------------------
# Dropdown values
# -----------------------------
CONTACT_OWNERS = [f"Sales Person-{i}" for i in range(1, 31)]

TRACK_INTERESTED = [
    "Data Science",
    "Data Analytics",
    "MERN Stack Developer",
    "Python Django",
    "Flutter",
    "Digital Marketing",
    "UI/UX Design"
]

DISTRICTS = [
    "Thiruvananthapuram", "Kollam", "Pathanamthitta", "Alappuzha", "Kottayam",
    "Idukki", "Ernakulam", "Thrissur", "Palakkad", "Malappuram",
    "Kozhikode", "Wayanad", "Kannur", "Kasaragod"
]

SOURCE_OF_LEAD = [
    "Organic/Website Traffic",
    "Job Portal",
    "Direct Enquiry / Referral",
    "Internal / Existing",
    "Social Media",
    "Walk-in"
]

COURSES = [
    "BE/BTECH",
    "BCA",
    "BCS",
    "BA",
    "BE"
]

SPECIALIZATIONS = [
    "AI And Data Science",
    "Applied Statistics And Data Analytics",
    "Aviation",
    "Arts",
    "Business Analytics"
]

GENDERS = ["Male", "Female"]

# -----------------------------
# Database functions
# -----------------------------
def init_db():
    if DB_DIALECT == "sqlite":
        create_table_query = """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lead_id TEXT,
            contact_owner TEXT,
            track_interested TEXT,
            district TEXT,
            source_of_lead TEXT,
            course TEXT,
            specialization TEXT,
            gender TEXT,
            conversion_probability REAL,
            score REAL,
            predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            actual_outcome INTEGER DEFAULT 0
        );
        """
    else:
        create_table_query = """
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            lead_id TEXT,
            contact_owner TEXT,
            track_interested TEXT,
            district TEXT,
            source_of_lead TEXT,
            course TEXT,
            specialization TEXT,
            gender TEXT,
            conversion_probability DOUBLE PRECISION,
            score DOUBLE PRECISION,
            predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            actual_outcome INTEGER DEFAULT 0
        );
        """

    with engine.begin() as conn:
        conn.execute(text(create_table_query))


def save_prediction(
    lead_id,
    contact_owner,
    track_interested,
    district,
    source_of_lead,
    course,
    specialization,
    gender,
    conversion_probability,
    score
):
    insert_query = text("""
        INSERT INTO predictions (
            lead_id,
            contact_owner,
            track_interested,
            district,
            source_of_lead,
            course,
            specialization,
            gender,
            conversion_probability,
            score,
            predicted_at,
            actual_outcome
        )
        VALUES (
            :lead_id,
            :contact_owner,
            :track_interested,
            :district,
            :source_of_lead,
            :course,
            :specialization,
            :gender,
            :conversion_probability,
            :score,
            :predicted_at,
            :actual_outcome
        )
    """)

    with engine.begin() as conn:
        conn.execute(insert_query, {
            "lead_id": lead_id,
            "contact_owner": contact_owner,
            "track_interested": track_interested,
            "district": district,
            "source_of_lead": source_of_lead,
            "course": course,
            "specialization": specialization,
            "gender": gender,
            "conversion_probability": float(conversion_probability),
            "score": float(score),
            "predicted_at": datetime.now(),
            "actual_outcome": 0
        })


def get_all_predictions():
    query = text("""
        SELECT *
        FROM predictions
        ORDER BY id DESC
    """)

    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()
    return rows


def update_actual_outcome(record_id, outcome_value):
    query = text("""
        UPDATE predictions
        SET actual_outcome = :outcome_value
        WHERE id = :record_id
    """)

    with engine.begin() as conn:
        conn.execute(query, {
            "outcome_value": outcome_value,
            "record_id": record_id
        })


def get_actual_outcome(record_id):
    query = text("""
        SELECT actual_outcome
        FROM predictions
        WHERE id = :record_id
    """)

    with engine.connect() as conn:
        row = conn.execute(query, {"record_id": record_id}).mappings().first()

    if row:
        return row["actual_outcome"]
    return 0


def get_predictions_dataframe():
    query = """
        SELECT
            id,
            lead_id,
            contact_owner,
            track_interested,
            district,
            source_of_lead,
            course,
            specialization,
            gender,
            conversion_probability,
            score,
            predicted_at,
            actual_outcome
        FROM predictions
        ORDER BY id DESC
    """

    df = pd.read_sql(query, engine)

    if not df.empty:
        df["conversion_probability"] = (df["conversion_probability"] * 100).round(2)
        df["actual_outcome"] = df["actual_outcome"].fillna(0).astype(int)

    return df


def get_dashboard_data(period="yearly", value=""):
    query = """
        SELECT predicted_at, conversion_probability, score
        FROM predictions
        ORDER BY predicted_at
    """

    df = pd.read_sql(query, engine)

    if df.empty:
        return {
            "labels": [],
            "lead_counts": [],
            "avg_probability": [],
            "avg_score": []
        }

    df["predicted_at"] = pd.to_datetime(df["predicted_at"], errors="coerce")
    df = df.dropna(subset=["predicted_at"])

    if df.empty:
        return {
            "labels": [],
            "lead_counts": [],
            "avg_probability": [],
            "avg_score": []
        }

    if period == "monthly" and value:
        year, month = value.split("-")
        filtered = df[
            (df["predicted_at"].dt.year == int(year)) &
            (df["predicted_at"].dt.month == int(month))
        ].copy()
        filtered["label"] = filtered["predicted_at"].dt.day.astype(str)

    elif period == "quarterly" and value:
        year, quarter = value.split("-")
        quarter_num = int(quarter.replace("Q", ""))
        filtered = df[
            (df["predicted_at"].dt.year == int(year)) &
            (df["predicted_at"].dt.quarter == quarter_num)
        ].copy()
        filtered["label"] = filtered["predicted_at"].dt.month_name().str[:3]

    elif period == "halfyearly" and value:
        year, half = value.split("-")

        if half == "H1":
            filtered = df[
                (df["predicted_at"].dt.year == int(year)) &
                (df["predicted_at"].dt.month.between(1, 6))
            ].copy()
        else:
            filtered = df[
                (df["predicted_at"].dt.year == int(year)) &
                (df["predicted_at"].dt.month.between(7, 12))
            ].copy()

        filtered["label"] = filtered["predicted_at"].dt.month_name().str[:3]

    elif period == "yearly" and value:
        filtered = df[df["predicted_at"].dt.year == int(value)].copy()
        filtered["label"] = filtered["predicted_at"].dt.month_name().str[:3]

    else:
        filtered = df.copy()
        filtered["label"] = filtered["predicted_at"].dt.year.astype(str)

    if filtered.empty:
        return {
            "labels": [],
            "lead_counts": [],
            "avg_probability": [],
            "avg_score": []
        }

    grouped = filtered.groupby("label").agg(
        lead_count=("predicted_at", "count"),
        avg_probability=("conversion_probability", "mean"),
        avg_score=("score", "mean")
    ).reset_index()

    grouped["avg_probability"] = (grouped["avg_probability"] * 100).round(2)
    grouped["avg_score"] = grouped["avg_score"].round(2)

    return {
        "labels": grouped["label"].tolist(),
        "lead_counts": grouped["lead_count"].tolist(),
        "avg_probability": grouped["avg_probability"].tolist(),
        "avg_score": grouped["avg_score"].tolist()
    }


def get_history_rows(search_lead=""):
    if search_lead:
        # Cross-database case-insensitive search for both PostgreSQL and SQLite
        query = text("""
            SELECT *
            FROM predictions
            WHERE LOWER(COALESCE(lead_id, '')) LIKE LOWER(:search_text)
            ORDER BY id DESC
        """)
        params = {"search_text": f"%{search_lead}%"}
    else:
        query = text("""
            SELECT *
            FROM predictions
            ORDER BY id DESC
        """)
        params = {}

    with engine.connect() as conn:
        rows = conn.execute(query, params).mappings().all()

    return rows


def delete_prediction_by_id(record_id):
    query = text("""
        DELETE FROM predictions
        WHERE id = :record_id
    """)

    with engine.begin() as conn:
        conn.execute(query, {"record_id": record_id})


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction_result = None
    error_message = None

    if request.method == "POST":
        try:
            lead_id = request.form.get("Lead ID", "").strip()

            contact_owner = request.form.get("Contact Owner", "Unspecified")
            track_interested = request.form.get("Track Interested", "Unspecified")
            district = request.form.get("District", "Unspecified")
            source_of_lead = request.form.get("Source of lead", "Unspecified")
            course = request.form.get("Course", "Unspecified")
            specialization = request.form.get("Specialization", "Unspecified")
            gender = request.form.get("Gender", "Unspecified")

            input_df = pd.DataFrame([{
                "Contact Owner": contact_owner,
                "Track Interested": track_interested,
                "District": district,
                "Source of lead": source_of_lead,
                "Course": course,
                "Specialization": specialization,
                "Gender": gender
            }])

            prob = float(pipeline.predict_proba(input_df)[0][1])
            score = round(prob * 100, 2)

            save_prediction(
                lead_id,
                contact_owner,
                track_interested,
                district,
                source_of_lead,
                course,
                specialization,
                gender,
                prob,
                score
            )

            prediction_result = {
                "conversion_probability": f"{prob * 100:.2f}%",
                "score": score
            }

        except SQLAlchemyError as e:
            error_message = f"Database error: {str(e)}"
        except Exception as e:
            error_message = f"Prediction error: {str(e)}"

    return render_template(
        "predict.html",
        contact_owners=CONTACT_OWNERS,
        track_interested_list=TRACK_INTERESTED,
        districts=DISTRICTS,
        source_of_lead_list=SOURCE_OF_LEAD,
        courses=COURSES,
        specializations=SPECIALIZATIONS,
        genders=GENDERS,
        prediction_result=prediction_result,
        error_message=error_message
    )


@app.route("/history")
def history():
    try:
        search_lead = request.args.get("lead_id", "").strip()
        rows = get_history_rows(search_lead)
        return render_template("history.html", rows=rows, search_lead=search_lead)
    except Exception as e:
        return render_template("history.html", rows=[], search_lead="", error_message=str(e))


@app.route("/toggle_conversion/<int:record_id>", methods=["POST"])
def toggle_conversion(record_id):
    current_value = get_actual_outcome(record_id)

    if current_value == 1:
        update_actual_outcome(record_id, 0)
    else:
        update_actual_outcome(record_id, 1)

    return redirect(url_for("history"))


@app.route("/delete_prediction/<int:record_id>", methods=["POST"])
def delete_prediction(record_id):
    delete_prediction_by_id(record_id)
    return redirect(url_for("history"))


@app.route("/dashboard")
def dashboard():
    rows = get_all_predictions()

    total_predictions = len(rows)
    converted_count = sum(1 for row in rows if row["actual_outcome"] == 1)
    not_converted_count = sum(1 for row in rows if row["actual_outcome"] == 0)

    avg_probability = round(
        sum(row["conversion_probability"] for row in rows) / total_predictions * 100, 2
    ) if total_predictions > 0 else 0

    avg_score = round(
        sum(row["score"] for row in rows) / total_predictions, 2
    ) if total_predictions > 0 else 0

    years = list(range(2024, 2051))

    return render_template(
        "dashboard.html",
        total_predictions=total_predictions,
        converted_count=converted_count,
        not_converted_count=not_converted_count,
        avg_probability=avg_probability,
        avg_score=avg_score,
        years=years
    )


@app.route("/dashboard-data")
def dashboard_data():
    period = request.args.get("period", "yearly")
    value = request.args.get("value", "")
    return jsonify(get_dashboard_data(period, value))


@app.route("/export/csv")
def export_csv():
    df = get_predictions_dataframe()

    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="prediction_history.csv"
    )


@app.route("/export/excel")
def export_excel():
    try:
        df = get_predictions_dataframe()

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Prediction History")

        output.seek(0)

        return send_file(
            output,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name="prediction_history.xlsx"
        )
    except Exception as e:
        return f"Excel export error: {str(e)}", 500


# -----------------------------
# Start app
# -----------------------------
if __name__ == "__main__":
    if check_db_connection():
        try:
            init_db()
            print(f"Database initialized successfully using: {DATABASE_URL}")
        except Exception as e:
            print(f"Database initialization failed: {e}")
    else:
        print("Running app without database initialization.")

    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)