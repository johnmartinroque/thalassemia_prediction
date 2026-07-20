"""
test.py
-------
Loads the best model saved by train.py and lets you type in a patient's
values to get a predicted 'thal' result.

Run:
    python test.py
(must be run after train.py, in the same folder)
"""
import os
import numpy as np
import pandas as pd
import joblib


MODEL_DIR = "models"
# Load the best model, its matching scaler, and the feature order it expects
model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
all_models = joblib.load(os.path.join(MODEL_DIR, "all_models.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
features = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))

# Human-readable labels for the thal classes.
# Adjust this if your data uses a different encoding.
THAL_LABELS = {
    0: "normal / unknown",
    1: "normal",
    2: "fixed defect",
    3: "reversable defect",
}

FIELD_HINTS = {
    "age": "age in years, e.g. 55",
    "sex": "1 = male, 0 = female",
    "cp": "chest pain type, 0-3",
    "trestbps": "resting blood pressure, e.g. 130",
    "chol": "serum cholesterol mg/dl, e.g. 246",
    "fbs": "fasting blood sugar > 120 mg/dl? 1 = yes, 0 = no",
    "restecg": "resting ECG results, 0/1/2",
    "thalach": "max heart rate achieved, e.g. 150",
    "exang": "exercise induced angina, 1 = yes, 0 = no",
    "oldpeak": "ST depression induced by exercise, e.g. 1.0",
    "slope": "slope of peak exercise ST segment, 0/1/2",
    "ca": "number of major vessels colored by flourosopy, 0-3",
    "target": "heart disease diagnosis, 1 = present, 0 = absent",
}


def prompt_float(field_name):
    hint = FIELD_HINTS.get(field_name, "")
    while True:
        raw = input(f"{field_name} ({hint}): ").strip()
        try:
            return float(raw)
        except ValueError:
            print("  Please enter a number.")


def main():
    print("Enter patient values to predict the 'thal' result.\n")

    # Get user input
    values = [prompt_float(f) for f in features]

    # Create DataFrame and scale it
    X_input = pd.DataFrame([values], columns=features)
    X_scaled = scaler.transform(X_input)

    print("\nPredictions by each model:")
    print("-" * 70)

    predictions = {}

    for name, mdl in all_models.items():
        if name == "Decision Tree":  # skip decision tree
            continue
        pred = mdl.predict(X_scaled)[0]
        label = THAL_LABELS.get(pred, str(pred))
        predictions[name] = pred

        if hasattr(mdl, "predict_proba"):
            confidence = max(mdl.predict_proba(X_scaled)[0]) * 100
            print(f"{name:22s}: {label:20s} ({confidence:.2f}% confidence)")
        else:
            print(f"{name:22s}: {label}")

    print("\nBest model prediction:")
    best_pred = model.predict(X_scaled)[0]
    best_label = THAL_LABELS.get(best_pred, str(best_pred))
    print(f"{best_label}")

if __name__ == "__main__":
    main()