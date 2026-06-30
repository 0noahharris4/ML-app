from __future__ import annotations
import os, io, csv
import requests
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template, Response

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

# ── Constants ─────────────────────────────────────────────────────────────────
DATASET_URL = (
    "https://raw.githubusercontent.com/0noahharris4/Machine-Learning-Repo"
    "/refs/heads/main/diabetes_prediction_dataset.csv"
)
MODEL_PATH       = "diabetes_model.pkl"
CATEGORICAL_COLS = ["gender", "smoking_history"]
NUMERIC_COLS     = ["age", "hypertension", "heart_disease", "bmi",
                    "HbA1c_level", "blood_glucose_level"]

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)


# ── Model loading / training ──────────────────────────────────────────────────
def train_model() -> Pipeline:
    """Download dataset from GitHub and train the model."""
    print("Downloading dataset from GitHub…")
    resp = requests.get(DATASET_URL, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))

    X = df.drop("diabetes", axis=1)
    Y = df["diabetes"]

    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ("num", StandardScaler(), NUMERIC_COLS),
    ])

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(Y), y=Y
    )
    cw = {i: w for i, w in enumerate(class_weights)}

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight=cw)),
    ])

    X_train, _, Y_train, _ = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, Y_train)
    joblib.dump(pipeline, MODEL_PATH)
    print("Model trained and saved.")
    return pipeline


def load_model() -> Pipeline:
    if os.path.exists(MODEL_PATH):
        print("Loading existing model…")
        return joblib.load(MODEL_PATH)
    return train_model()


# Train / load at startup
_model: Pipeline = load_model()


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_input_df(data: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "gender":              str(data["gender"]),
        "age":                 float(data["age"]),
        "hypertension":        int(data["hypertension"]),
        "heart_disease":       int(data["heart_disease"]),
        "smoking_history":     str(data["smoking_history"]),
        "bmi":                 float(data["bmi"]),
        "HbA1c_level":         float(data["HbA1c_level"]),
        "blood_glucose_level": int(data["blood_glucose_level"]),
    }])


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    try:
        df          = build_input_df(data)
        prediction  = int(_model.predict(df)[0])
        probability = float(_model.predict_proba(df)[0][1])
        return jsonify({
            "prediction":  prediction,
            "probability": round(probability, 4),
            "risk_label":  "High Risk" if prediction == 1 else "Low Risk",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/report", methods=["POST"])
def report():
    """Stream a CSV report of the prediction back to the browser."""
    data = request.get_json() or {}
    try:
        df          = build_input_df(data)
        prediction  = int(_model.predict(df)[0])
        probability = float(_model.predict_proba(df)[0][1])

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "gender", "age", "hypertension", "heart_disease",
            "smoking_history", "bmi", "HbA1c_level", "blood_glucose_level",
            "predicted_diabetes", "predicted_probability",
        ])
        writer.writerow([
            data["gender"], data["age"], data["hypertension"],
            data["heart_disease"], data["smoking_history"], data["bmi"],
            data["HbA1c_level"], data["blood_glucose_level"],
            prediction, round(probability, 4),
        ])

        return Response(
            buf.getvalue().encode("utf-8"),
            mimetype="text/csv",
            headers={"Content-Disposition":
                     "attachment; filename=diabetes_prediction_report.csv"},
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
