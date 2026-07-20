"""
train.py
--------
Trains several ML models to predict Thalassemia Status:
    0 = normal
    1 = fixed defect
    2 = reversable defect
(NOTE: your original data used 0/1/2/3 for thal — see the cleaning step below.)

Inputs used to predict Thalassemia Status:
    age, sex, cp, trestbps, chol, fbs, restecg, thalach,
    exang, oldpeak, slope, ca, target

Run:
    python train.py
"""

import pandas as pd
import numpy as np
import joblib
import os


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

DATA_PATH = "heart.csv"
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "target"]
TARGET_COL = "thal"

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# Drop exact duplicate rows so the same patient record can't end up in
# both train and test — this is what was inflating accuracy to ~98%.
df = df.drop_duplicates()

# No row limiting/sampling happens here — every row in heart.csv is used.
# This print confirms the full row count made it into the DataFrame.
print(f"Loaded {len(df)} rows from '{DATA_PATH}' after removing duplicates")

# Your description says Thalassemia Status should be 0/1/2 (normal/fixed/
# reversable), but the classic UCI heart.csv actually stores it as 0,1,2,3
# (with 0 meaning "unknown/missing" in the original source and 1/2/3
# meaning normal/fixed/reversable). Some rows may have thal=3 and thal=0.
# We keep whatever values are in your file and just report which
# classes exist, so nothing gets silently mislabeled.
print("Classes found in 'Thalassemia Status' column:", sorted(df[TARGET_COL].unique()))
print("Class counts:\n", df[TARGET_COL].value_counts().sort_index())

X = df[FEATURES]
y = df[TARGET_COL]

# ---------------------------------------------------------------------
# 2. Train / test split
# ---------------------------------------------------------------------
# With 1026 rows this is a reasonably sized split. stratify=y keeps the
# class proportions (including any rare Thalassemia Status classes)
# consistent between train and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---------------------------------------------------------------------
# 3. Scale features (helps Logistic Regression, SVM, KNN)
# ---------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------
# 4. Define candidate models
# ---------------------------------------------------------------------
models = {
    "logistic_regression": LogisticRegression(max_iter=2000),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "svm": SVC(kernel="rbf", probability=True, random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=3),
}

# ---------------------------------------------------------------------
# 5. Train + evaluate each model, track the best one
# ---------------------------------------------------------------------
results = {}
best_name, best_model, best_acc = None, None, -1

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name:22s} accuracy: {acc:.3f}")

    if acc > best_acc:
        best_name, best_model, best_acc = name, model, acc

print(f"\nBest model: {best_name} (accuracy={best_acc:.3f})")

# ---------------------------------------------------------------------
# 6. Save everything accuracy.py and test.py will need
# ---------------------------------------------------------------------
joblib.dump(models, os.path.join(MODEL_DIR, "all_models.pkl"))
joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(FEATURES, os.path.join(MODEL_DIR, "features.pkl"))

# Save the exact train/test split
X_test.to_csv(os.path.join(MODEL_DIR, "X_test.csv"), index=False)
y_test.to_csv(os.path.join(MODEL_DIR, "y_test.csv"), index=False)

print(f"\nSaved model files in '{MODEL_DIR}/'")