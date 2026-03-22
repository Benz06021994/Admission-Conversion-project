import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

SQLITE_DB_PATH = "predictions.db"
DATABASE_URL = os.getenv("DATABASE_URL")

postgres_engine = create_engine(DATABASE_URL, future=True)

def create_postgres_table():
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
        predicted_at TIMESTAMP,
        actual_outcome INTEGER DEFAULT 0
    );
    """

    with postgres_engine.begin() as conn:
        conn.execute(text(create_table_query))


def migrate_data():
    sqlite_conn = sqlite3.connect(SQLITE_DB_PATH)

    df = pd.read_sql_query("SELECT * FROM predictions", sqlite_conn)
    sqlite_conn.close()

    if df.empty:
        print("No data found in SQLite.")
        return

    df["actual_outcome"] = df["actual_outcome"].fillna(0).astype(int)
    df["predicted_at"] = pd.to_datetime(df["predicted_at"], errors="coerce")
    df = df.dropna(subset=["predicted_at"])

    df.to_sql("predictions", postgres_engine, if_exists="append", index=False)

    print(f"✅ Migrated {len(df)} rows successfully!")


if __name__ == "__main__":
    create_postgres_table()
    migrate_data()