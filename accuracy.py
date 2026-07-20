"""
accuracy.py
-----------
Loads the models trained by train.py and reports each one's accuracy
on the SAME held-out test split that was used during training.

Run:
    python accuracy.py
(must be run after train.py, in the same folder)
"""
import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report


MODEL_DIR = "models"
# Load saved artifacts
models = joblib.load(os.path.join(MODEL_DIR, "all_models.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

X_test = pd.read_csv(os.path.join(MODEL_DIR, "X_test.csv"))
y_test = pd.read_csv(os.path.join(MODEL_DIR, "y_test.csv")).squeeze("columns")

X_test_scaled = scaler.transform(X_test)

print(f"Test set size: {len(X_test)} rows\n")
print(f"{'Model':22s} {'Accuracy':>10s}")
print("-" * 34)

scored = []
for name, model in models.items():
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    scored.append((name, acc, preds))
    print(f"{name:22s} {acc:10.3f}")

scored.sort(key=lambda t: t[1], reverse=True)
best_name, best_acc, best_preds = scored[0]

print(f"\nBest model: {best_name} ({best_acc:.3f})")
print(f"\nDetailed report for best model ({best_name}) — Thalassemia Status classification:")
print(classification_report(y_test, best_preds, zero_division=0))