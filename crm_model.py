# ==========================================================
# crm_model.py  (PROBABILITY-FOCUSED / FAST VERSION)
# ==========================================================
# CRM Lead Conversion Probability Modeling
# Scope: ONLY 2024 & 2025
#
# Frontend Inputs (7):
#   - Contact Owner
#   - Track Interested
#   - District
#   - Source of lead
#   - Course
#   - Specialization
#   - Gender
#
# Internal engineered features:
#   - SourceGroup
#   - OwnerRate
#   - CourseRate
#   - TrackRate
#   - DistrictRate
#   - SpecializationRate
#   - SourceGroupRate
#
# Goal:
#   Predict conversion probability only
#
# Models:
#   - LogisticRegression
#   - RandomForest
#   - XGBoost (optional)
#   - LightGBM (optional)
#
# Split:
#   - Time-based 70/15/15 by Created Time
#
# Selection metric:
#   - Validation PR-AUC
#
# Outputs:
#   - crm_output_v2/artifacts/best_pipeline.pkl
#   - crm_output_v2/artifacts/best_artifact.pkl
#   - crm_output_v2/artifacts/model_card.json
#   - crm_output_v2/artifacts/prediction_schema.json
#   - crm_output_v2/reports/model_report.xlsx
#   - crm_output_v2/charts/*.png
#
# Run:
#   python crm_model.py --data crm_conversion_model.csv
# ==========================================================

import os
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import ParameterGrid

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    brier_score_loss
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


# =========================
# Settings
# =========================
TARGET = "Converted"
DATE_COL = "Created Time"
YEAR_FILTER = [2024, 2025]

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42
MISSING_LABEL = "Unspecified"

FRONTEND_FEATURES = [
    "Contact Owner",
    "Track Interested",
    "District",
    "Source of lead",
    "Course",
    "Specialization",
    "Gender"
]

RAW_REQUIRED = [DATE_COL] + FRONTEND_FEATURES + [TARGET]

CATEGORICAL_MODEL_FEATURES = [
    "Contact Owner",
    "Track Interested",
    "District",
    "Course",
    "Specialization",
    "Gender",
    "SourceGroup"
]

NUMERIC_MODEL_FEATURES = [
    "OwnerRate",
    "CourseRate",
    "TrackRate",
    "DistrictRate",
    "SpecializationRate",
    "SourceGroupRate"
]

ALL_MODEL_FEATURES = CATEGORICAL_MODEL_FEATURES + NUMERIC_MODEL_FEATURES


# =========================
# Helpers
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def safe_to_datetime(s):
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

def normalize_text_series(s: pd.Series) -> pd.Series:
    s = s.astype("object")
    s = s.replace(["nan", "NaN", "None", "NONE", ""], np.nan)
    s = s.fillna(MISSING_LABEL).astype(str)
    s = s.str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    return s

def normalize_course(s: pd.Series) -> pd.Series:
    s = normalize_text_series(s)
    return s.str.upper()

def normalize_gender(s: pd.Series) -> pd.Series:
    return normalize_text_series(s).str.title()

def conversion_summary_table(df_in: pd.DataFrame, col: str):
    return (
        df_in.groupby(col)[TARGET]
        .agg(count="size", conversions="sum", conversion_rate="mean")
        .reset_index()
        .sort_values(["count", "conversion_rate"], ascending=[False, False])
    )

def save_line_chart(series: pd.Series, title: str, ylabel: str, out_path: str, show: bool = False):
    plt.figure(figsize=(10, 4))
    series.sort_index().plot()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()

def save_bar_chart(table: pd.DataFrame, category_col: str, value_col: str, title: str, out_path: str, top_k: int = 10, show: bool = False):
    t = table.copy()
    if t.empty:
        return
    t = t.sort_values(value_col, ascending=False).head(top_k)
    t = t.sort_values(value_col, ascending=True)

    plt.figure(figsize=(10, 4))
    plt.barh(t[category_col].astype(str), t[value_col])
    plt.title(title)
    plt.xlabel(value_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()

def save_pr_curve(y_true, probs, title: str, out_path: str, show: bool = False):
    prec, rec, _ = precision_recall_curve(y_true, probs)
    plt.figure(figsize=(6, 4))
    plt.plot(rec, prec)
    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()


# =========================
# Source grouping buckets
# =========================
ORGANIC = {
    "INFOPARK WEBSITE", "WEBSITE ENQUIRY", "GOOGLE SOURCED LEADS", "SEO",
    "SOCIAL MEDIA", "FACEBOOK", "LINKEDIN"
}
PAID = {
    "DIGITAL MARKETING", "WHATSAPP AD CAMPAIGN", "FREE INTERNSHIP CAMPAIGN",
    "PLACEMENT CAMPAIGN", "BULK LEAD"
}
JOB_PORTAL = {"INDEED", "NAUKRI", "OLX"}
EVENTS = {
    "SEMINAR", "WORKSHOP", "WEBINAR", "JOB FAIR", "TECHFEST", "BOOT CAMP",
    "INDUSTRIAL VISIT", "CAMPUS OUTREACH", "REFERALS FROM COLLEGES",
    "REFERRALS FROM COLLEGES", "KKEM"
}
DIRECT = {"ENQUIRY", "INCOMING CONTACT", "REFERENCE"}
INTERNAL = {"CARRYOVER", "CARRY OVER"}

def group_source_value(x: str) -> str:
    if x is None:
        return "Misc"
    v = str(x).strip()
    v = " ".join(v.split())
    v_up = v.upper()

    if v_up in ORGANIC:
        return "Organic"
    if v_up in PAID:
        return "Paid Marketing"
    if v_up in JOB_PORTAL:
        return "Job Portal"
    if v_up in EVENTS:
        return "Events / Offline"
    if v_up in DIRECT:
        return "Direct Enquiry / Referral"
    if v_up in INTERNAL:
        return "Internal / Existing"
    if v_up in {"OTHERS", "OTHER", "UNKNOWN", "UNSPECIFIED"}:
        return "Misc"
    return "Misc"


# =========================
# Feature Builder
# =========================
class CRMFeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=20.0):
        self.smoothing = smoothing

    def _normalize_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["Contact Owner"] = normalize_text_series(X["Contact Owner"])
        X["Track Interested"] = normalize_text_series(X["Track Interested"])
        X["District"] = normalize_text_series(X["District"])
        X["Source of lead"] = normalize_text_series(X["Source of lead"])
        X["Course"] = normalize_course(X["Course"])
        X["Specialization"] = normalize_text_series(X["Specialization"])
        X["Gender"] = normalize_gender(X["Gender"])
        X["SourceGroup"] = X["Source of lead"].apply(group_source_value)
        return X

    def _smoothed_rate_map(self, series: pd.Series, y: pd.Series):
        tmp = pd.DataFrame({"col": series, "y": y})
        grp = tmp.groupby("col")["y"].agg(["mean", "count"]).reset_index()
        grp["smoothed"] = (
            (grp["count"] * grp["mean"] + self.smoothing * self.global_rate_) /
            (grp["count"] + self.smoothing)
        )
        return dict(zip(grp["col"], grp["smoothed"]))

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("CRMFeatureBuilder.fit requires y.")

        X = pd.DataFrame(X).copy()
        X = self._normalize_frame(X)
        y = pd.Series(y).astype(int)

        self.global_rate_ = float(y.mean())
        self.owner_rate_map_ = self._smoothed_rate_map(X["Contact Owner"], y)
        self.course_rate_map_ = self._smoothed_rate_map(X["Course"], y)
        self.track_rate_map_ = self._smoothed_rate_map(X["Track Interested"], y)
        self.district_rate_map_ = self._smoothed_rate_map(X["District"], y)
        self.specialization_rate_map_ = self._smoothed_rate_map(X["Specialization"], y)
        self.sourcegroup_rate_map_ = self._smoothed_rate_map(X["SourceGroup"], y)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X = self._normalize_frame(X)

        X["OwnerRate"] = X["Contact Owner"].map(self.owner_rate_map_).fillna(self.global_rate_)
        X["CourseRate"] = X["Course"].map(self.course_rate_map_).fillna(self.global_rate_)
        X["TrackRate"] = X["Track Interested"].map(self.track_rate_map_).fillna(self.global_rate_)
        X["DistrictRate"] = X["District"].map(self.district_rate_map_).fillna(self.global_rate_)
        X["SpecializationRate"] = X["Specialization"].map(self.specialization_rate_map_).fillna(self.global_rate_)
        X["SourceGroupRate"] = X["SourceGroup"].map(self.sourcegroup_rate_map_).fillna(self.global_rate_)

        return X[ALL_MODEL_FEATURES].copy()


# =========================
# Data loading
# =========================
def load_and_prepare_data(data_path: str, encoding: str = "latin1") -> pd.DataFrame:
    df_raw = pd.read_csv(data_path, encoding=encoding, low_memory=False)
    df_raw.columns = [c.strip() for c in df_raw.columns]

    missing = [c for c in RAW_REQUIRED if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nAvailable columns: {df_raw.columns.tolist()}")

    print("Raw shape:", df_raw.shape)

    df = df_raw[RAW_REQUIRED].copy()
    df[DATE_COL] = safe_to_datetime(df[DATE_COL])

    print("\nYear distribution after parsing Created Time:")
    print(df[DATE_COL].dt.year.value_counts(dropna=False).sort_index())
    print("Missing Created Time after parsing:", int(df[DATE_COL].isna().sum()))

    df = df.dropna(subset=[DATE_COL]).copy()
    df = df[df[DATE_COL].dt.year.isin(YEAR_FILTER)].copy()

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int).clip(0, 1)

    df["Contact Owner"] = normalize_text_series(df["Contact Owner"])
    df["Track Interested"] = normalize_text_series(df["Track Interested"])
    df["District"] = normalize_text_series(df["District"])
    df["Source of lead"] = normalize_text_series(df["Source of lead"])
    df["Course"] = normalize_course(df["Course"])
    df["Specialization"] = normalize_text_series(df["Specialization"])
    df["Gender"] = normalize_gender(df["Gender"])

    df = df.drop_duplicates(subset=FRONTEND_FEATURES + [TARGET, DATE_COL]).reset_index(drop=True)

    print("\nAfter year filter shape:", df.shape)
    print("Target counts:\n", df[TARGET].value_counts(dropna=False))
    print("Conversion rate:", round(df[TARGET].mean(), 4))

    return df


def time_split_70_15_15(df: pd.DataFrame):
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    valid_end = int(n * (TRAIN_RATIO + VALID_RATIO))

    train_df = df.iloc[:train_end].copy()
    valid_df = df.iloc[train_end:valid_end].copy()
    test_df = df.iloc[valid_end:].copy()

    return train_df, valid_df, test_df


# =========================
# Model builders
# =========================
def make_preprocessor():
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe)
    ])

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    return ColumnTransformer(
        transformers=[
            ("cat", categorical_pipe, CATEGORICAL_MODEL_FEATURES),
            ("num", numeric_pipe, NUMERIC_MODEL_FEATURES)
        ],
        remainder="drop"
    )


def make_pipeline(model):
    return Pipeline(steps=[
        ("feature_builder", CRMFeatureBuilder(smoothing=20.0)),
        ("preprocessor", make_preprocessor()),
        ("model", model)
    ])


def get_model_search_space(pos_count: int, neg_count: int):
    scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0

    search_space = {
        "LogisticRegression": {
            "model_class": LogisticRegression,
            "param_grid": {
                "C": [0.5, 1.0],
                "solver": ["liblinear"],
                "class_weight": ["balanced"],
                "max_iter": [2000],
                "random_state": [RANDOM_STATE]
            }
        },
        "RandomForest": {
            "model_class": RandomForestClassifier,
            "param_grid": {
                "n_estimators": [300],
                "max_depth": [8, 12],
                "min_samples_split": [10],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt"],
                "class_weight": ["balanced"],
                "random_state": [RANDOM_STATE],
                "n_jobs": [-1]
            }
        }
    }

    if HAS_XGB:
        search_space["XGBoost"] = {
            "model_class": XGBClassifier,
            "param_grid": {
                "n_estimators": [300],
                "max_depth": [6],
                "learning_rate": [0.03, 0.05],
                "subsample": [0.9],
                "colsample_bytree": [0.9],
                "min_child_weight": [1, 3],
                "reg_lambda": [1.0, 2.0],
                "scale_pos_weight": [scale_pos_weight],
                "random_state": [RANDOM_STATE],
                "n_jobs": [-1],
                "eval_metric": ["logloss"],
                "tree_method": ["hist"]
            }
        }
    else:
        print("\n[INFO] xgboost not installed -> skipping XGBoost")

    if HAS_LGBM:
        search_space["LightGBM"] = {
            "model_class": LGBMClassifier,
            "param_grid": {
                "n_estimators": [300],
                "learning_rate": [0.03, 0.05],
                "num_leaves": [31],
                "max_depth": [8],
                "min_child_samples": [20],
                "subsample": [0.8, 0.9],
                "colsample_bytree": [0.8, 0.9],
                "reg_lambda": [1.0],
                "class_weight": ["balanced"],
                "random_state": [RANDOM_STATE],
                "n_jobs": [-1],
                "verbosity": [-1]
            }
        }
    else:
        print("\n[INFO] lightgbm not installed -> skipping LightGBM")

    return search_space


# =========================
# Evaluation
# =========================
def evaluate_probabilities(y_true, probs):
    roc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else np.nan
    pr = average_precision_score(y_true, probs) if len(np.unique(y_true)) > 1 else np.nan
    brier = brier_score_loss(y_true, probs)
    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "brier_score": brier
    }


def train_and_tune_models(train_df: pd.DataFrame, valid_df: pd.DataFrame):
    X_train = train_df[FRONTEND_FEATURES].copy()
    y_train = train_df[TARGET].copy()

    X_valid = valid_df[FRONTEND_FEATURES].copy()
    y_valid = valid_df[TARGET].copy()

    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())

    print("\nTrain size:", X_train.shape)
    print("Valid size:", X_valid.shape)
    print("Train positive:", pos, "| Train negative:", neg)

    search_space = get_model_search_space(pos, neg)

    tuning_rows = []
    best_objects = {}

    for model_name, cfg in search_space.items():
        print(f"\n=== Tuning {model_name} ===")

        best_pr = -1.0
        best_pipe = None
        best_params = None
        best_valid_probs = None

        for params in ParameterGrid(cfg["param_grid"]):
            model = cfg["model_class"](**params)
            pipe = make_pipeline(model)

            pipe.fit(X_train, y_train)
            valid_probs = pipe.predict_proba(X_valid)[:, 1]

            valid_metrics = evaluate_probabilities(y_valid, valid_probs)

            tuning_rows.append({
                "model": model_name,
                "params": json.dumps(params),
                "valid_pr_auc": valid_metrics["pr_auc"],
                "valid_roc_auc": valid_metrics["roc_auc"],
                "valid_brier_score": valid_metrics["brier_score"]
            })

            if valid_metrics["pr_auc"] > best_pr:
                best_pr = valid_metrics["pr_auc"]
                best_pipe = pipe
                best_params = params
                best_valid_probs = valid_probs.copy()

        best_objects[model_name] = {
            "pipeline": best_pipe,
            "params": best_params,
            "valid_probs": best_valid_probs
        }

        print("Best validation PR-AUC:", round(best_pr, 6))
        print("Best params:", best_params)

    tuning_df = pd.DataFrame(tuning_rows).sort_values(
        ["valid_pr_auc", "valid_roc_auc"],
        ascending=False
    ).reset_index(drop=True)

    best_model_name = tuning_df.iloc[0]["model"]
    best_info = best_objects[best_model_name]

    return tuning_df, best_model_name, best_info


def final_evaluation(best_model_name, best_info, train_df, valid_df, test_df, charts_dir, show_charts=False):
    train_valid_df = pd.concat([train_df, valid_df], axis=0).reset_index(drop=True)

    X_tv = train_valid_df[FRONTEND_FEATURES].copy()
    y_tv = train_valid_df[TARGET].copy()

    X_test = test_df[FRONTEND_FEATURES].copy()
    y_test = test_df[TARGET].copy()

    model = best_info["pipeline"].named_steps["model"].__class__(**best_info["params"])
    final_pipe = make_pipeline(model)
    final_pipe.fit(X_tv, y_tv)

    test_probs = final_pipe.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_probabilities(y_test, test_probs)

    print("\n=== Final Test Evaluation ===")
    print("Best model:", best_model_name)
    print("Test metrics:", {k: round(float(v), 4) if pd.notnull(v) else v for k, v in test_metrics.items()})

    save_pr_curve(
        y_test,
        test_probs,
        title=f"Precision-Recall Curve ({best_model_name}) - Test",
        out_path=os.path.join(charts_dir, f"pr_curve_test_{best_model_name}.png"),
        show=show_charts
    )

    return final_pipe, test_probs, test_metrics


# =========================
# Main
# =========================
def run_training(data_path: str, out_dir: str, encoding: str = "latin1", show_charts: bool = False):
    artifacts_dir = os.path.join(out_dir, "artifacts")
    reports_dir = os.path.join(out_dir, "reports")
    charts_dir = os.path.join(out_dir, "charts")

    ensure_dir(out_dir)
    ensure_dir(artifacts_dir)
    ensure_dir(reports_dir)
    ensure_dir(charts_dir)

    df = load_and_prepare_data(data_path, encoding=encoding)
    df["SourceGroup"] = df["Source of lead"].apply(group_source_value)

    conv_counts = df[TARGET].value_counts().sort_index()
    plt.figure(figsize=(5, 4))
    plt.bar(conv_counts.index.astype(str), conv_counts.values)
    plt.title("Converted (0/1) Count — 2024–2025")
    plt.xlabel("Converted")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "converted_distribution.png"), dpi=150)
    if show_charts:
        plt.show()
    plt.close()

    tmp = df.copy()
    tmp["month_start"] = tmp[DATE_COL].dt.to_period("M").dt.to_timestamp()
    monthly_volume = tmp.groupby("month_start").size()
    monthly_conv = tmp.groupby("month_start")[TARGET].mean()

    save_line_chart(
        monthly_volume,
        "Monthly Lead Volume Trend (2024–2025)",
        "Lead Count",
        os.path.join(charts_dir, "monthly_lead_volume.png"),
        show=show_charts
    )
    save_line_chart(
        monthly_conv,
        "Monthly Conversion Rate Trend (2024–2025)",
        "Conversion Rate",
        os.path.join(charts_dir, "monthly_conversion_rate.png"),
        show=show_charts
    )

    sg_tbl = conversion_summary_table(df, "SourceGroup")
    save_bar_chart(
        sg_tbl,
        "SourceGroup",
        "count",
        "Lead Volume by Source Group",
        os.path.join(charts_dir, "sourcegroup_volume.png"),
        top_k=10,
        show=show_charts
    )
    save_bar_chart(
        sg_tbl,
        "SourceGroup",
        "conversion_rate",
        "Conversion Rate by Source Group",
        os.path.join(charts_dir, "sourcegroup_conversion_rate.png"),
        top_k=10,
        show=show_charts
    )

    train_df, valid_df, test_df = time_split_70_15_15(df)

    print("\nSplit sizes:")
    print("Train:", train_df.shape)
    print("Valid:", valid_df.shape)
    print("Test :", test_df.shape)

    tuning_df, best_model_name, best_info = train_and_tune_models(train_df, valid_df)

    print("\n=== Tuning Summary (Top 10) ===")
    print(tuning_df.head(10).to_string(index=False))

    final_pipe, test_probs, test_metrics = final_evaluation(
        best_model_name=best_model_name,
        best_info=best_info,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        charts_dir=charts_dir,
        show_charts=show_charts
    )

    X_all = df[FRONTEND_FEATURES].copy()
    y_all = df[TARGET].copy()

    final_model_for_all = best_info["pipeline"].named_steps["model"].__class__(**best_info["params"])
    best_pipeline_all = make_pipeline(final_model_for_all)
    best_pipeline_all.fit(X_all, y_all)

    pipeline_path = os.path.join(artifacts_dir, "best_pipeline.pkl")
    joblib.dump(best_pipeline_all, pipeline_path)

    fb = best_pipeline_all.named_steps["feature_builder"]

    artifact = {
        "best_model_name": best_model_name,
        "best_model_params": best_info["params"],
        "year_filter": YEAR_FILTER,
        "missing_label_used": MISSING_LABEL,
        "frontend_fields": FRONTEND_FEATURES,
        "categorical_model_features": CATEGORICAL_MODEL_FEATURES,
        "numeric_model_features": NUMERIC_MODEL_FEATURES,
        "notes": (
            "This model is for conversion probability prediction. "
            "Raw frontend inputs are normalized inside pipeline. "
            "Source of lead is mapped to SourceGroup. "
            "Leakage-safe smoothed rate features are learned in fit(). "
            "Production output should use predict_proba() only."
        ),
        "global_conversion_rate": fb.global_rate_,
        "source_grouping": {
            "Organic": sorted(list(ORGANIC)),
            "Paid Marketing": sorted(list(PAID)),
            "Job Portal": sorted(list(JOB_PORTAL)),
            "Events / Offline": sorted(list(EVENTS)),
            "Direct Enquiry / Referral": sorted(list(DIRECT)),
            "Internal / Existing": sorted(list(INTERNAL)),
            "Misc": ["Others/Unspecified/Unknown/Any not mapped"]
        },
        "learned_rate_maps_preview": {
            "owner_rate_top10": dict(list(fb.owner_rate_map_.items())[:10]),
            "course_rate_top10": dict(list(fb.course_rate_map_.items())[:10]),
            "track_rate_top10": dict(list(fb.track_rate_map_.items())[:10]),
            "district_rate_top10": dict(list(fb.district_rate_map_.items())[:10]),
            "specialization_rate_top10": dict(list(fb.specialization_rate_map_.items())[:10]),
            "sourcegroup_rate_all": fb.sourcegroup_rate_map_
        },
        "tuning_top10": tuning_df.head(10).to_dict(orient="records"),
        "test_metrics": test_metrics
    }

    artifact_path = os.path.join(artifacts_dir, "best_artifact.pkl")
    joblib.dump(artifact, artifact_path)

    model_card = {
        "model_name": best_model_name,
        "model_params": best_info["params"],
        "training_years": YEAR_FILTER,
        "frontend_inputs": FRONTEND_FEATURES,
        "goal": "conversion_probability_prediction",
        "pipeline_contains_feature_engineering": True,
        "feature_engineering_inside_pipeline": {
            "source_grouping": True,
            "smoothed_target_rates": [
                "OwnerRate",
                "CourseRate",
                "TrackRate",
                "DistrictRate",
                "SpecializationRate",
                "SourceGroupRate"
            ]
        },
        "missing_label": MISSING_LABEL,
        "paths": {
            "pipeline": pipeline_path,
            "artifact": artifact_path
        },
        "test_metrics": test_metrics
    }

    model_card_path = os.path.join(artifacts_dir, "model_card.json")
    with open(model_card_path, "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=4)

    prediction_schema = {
        "input_schema": {
            "Contact Owner": "string",
            "Track Interested": "string",
            "District": "string",
            "Source of lead": "string",
            "Course": "string",
            "Specialization": "string",
            "Gender": "string"
        },
        "processing_rules": {
            "inside_pipeline": True,
            "text_normalization": "strip + collapse spaces; missing -> Unspecified",
            "course_normalization": "uppercase",
            "gender_normalization": "title case",
            "source_grouping": "Source of lead mapped to SourceGroup buckets",
            "rate_features": [
                "OwnerRate",
                "CourseRate",
                "TrackRate",
                "DistrictRate",
                "SpecializationRate",
                "SourceGroupRate"
            ],
            "unseen_category_handling": "falls back to global conversion rate for rate features; one-hot uses handle_unknown=ignore"
        },
        "output_schema": {
            "conversion_probability": "float between 0 and 1"
        }
    }

    schema_path = os.path.join(artifacts_dir, "prediction_schema.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(prediction_schema, f, indent=4)

    full_probs = best_pipeline_all.predict_proba(X_all)[:, 1]

    pred_out = df[[DATE_COL] + FRONTEND_FEATURES].copy()
    pred_out["SourceGroup"] = df["SourceGroup"].values
    pred_out["Converted_actual"] = y_all.values
    pred_out["conversion_probability"] = full_probs

    owner_tbl = conversion_summary_table(df, "Contact Owner")
    track_tbl = conversion_summary_table(df, "Track Interested")
    dist_tbl = conversion_summary_table(df, "District")
    course_tbl = conversion_summary_table(df, "Course")
    spec_tbl = conversion_summary_table(df, "Specialization")
    gender_tbl = conversion_summary_table(df, "Gender")

    excel_path = os.path.join(reports_dir, "model_report.xlsx")

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        tuning_df.to_excel(writer, sheet_name="tuning_results", index=False)

        owner_tbl.to_excel(writer, sheet_name="owner_summary", index=False)
        track_tbl.to_excel(writer, sheet_name="track_summary", index=False)
        dist_tbl.to_excel(writer, sheet_name="district_summary", index=False)
        course_tbl.to_excel(writer, sheet_name="course_summary", index=False)
        spec_tbl.to_excel(writer, sheet_name="specialization_summary", index=False)
        gender_tbl.to_excel(writer, sheet_name="gender_summary", index=False)
        sg_tbl.to_excel(writer, sheet_name="sourcegroup_summary", index=False)

        pd.DataFrame([test_metrics]).to_excel(writer, sheet_name="final_test_metrics", index=False)
        pred_out.to_excel(writer, sheet_name="predictions", index=False)

    print("\nSaved:")
    print("✅ Pipeline:", pipeline_path)
    print("✅ Artifact:", artifact_path)
    print("✅ Model card:", model_card_path)
    print("✅ Schema:", schema_path)
    print("✅ Excel report:", excel_path)
    print("📌 Charts saved in:", charts_dir)
    print("\n✅ DONE | Rows:", len(df), "| Best model:", best_model_name)


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="CRM Conversion Probability Model (2024-2025) with SourceGroup + rate features + reduced-grid tuning."
    )
    parser.add_argument("--data", required=True, help="Path to CRM CSV file")
    parser.add_argument("--out", default="crm_output_v2", help="Output directory")
    parser.add_argument("--encoding", default="latin1", help="CSV encoding (default: latin1)")
    parser.add_argument("--show_charts", action="store_true", help="Show chart popups")
    args = parser.parse_args()

    run_training(
        data_path=args.data,
        out_dir=args.out,
        encoding=args.encoding,
        show_charts=args.show_charts
    )

if __name__ == "__main__":
    main()