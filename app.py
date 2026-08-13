"""
app.py
------
Streamlit UI for the model trained by train.py.

Run:
    streamlit run app.py
(must be run after train.py, with the "models" folder present in the same directory)
"""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = "models"
VISUALS_DIR = "visuals"

st.set_page_config(page_title="Thalassemia Status Predictor", page_icon="🫀", layout="wide")

# filename -> caption shown under the chart
VISUAL_CAPTIONS = {
    "01_thal_class_distribution.png": "Distribution of Thalassemia Status classes in the dataset",
    "02_correlation_heatmap.png": "Correlation heatmap of all numeric features",
    "03_age_distribution.png": "Age distribution, split by Thalassemia Status",
    "04_thalach_by_thal.png": "Max heart rate achieved (thalach), by Thalassemia Status",
    "05_chol_by_thal.png": "Serum cholesterol, by Thalassemia Status",
    "06_target_vs_thal.png": "Heart disease diagnosis vs Thalassemia Status",
    "07_pairwise_scatter.png": "Pairwise relationships between key numeric features",
}

# ---------------------------------------------------------------------------
# Load model artifacts (cached so this only runs once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    all_models = joblib.load(os.path.join(MODEL_DIR, "all_models.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))
    return model, all_models, scaler, features


THALASSEMIA_LABELS = {
    0: "normal / unknown",
    1: "normal",
    2: "fixed defect",
    3: "reversable defect",
}

# field -> (label, help text, widget type, kwargs)
FIELD_CONFIG = {
    "age": ("Age", "Age in years", "number", dict(min_value=1, max_value=120, value=55, step=1)),
    "sex": ("Sex", "", "select", dict(options=[("Male", 1), ("Female", 0)])),
    "cp": ("Chest pain type", "0-3", "select", dict(options=[("0", 0), ("1", 1), ("2", 2), ("3", 3)])),
    "trestbps": ("Resting blood pressure", "mm Hg, e.g. 130", "number", dict(min_value=50, max_value=250, value=130, step=1)),
    "chol": ("Serum cholesterol", "mg/dl, e.g. 246", "number", dict(min_value=50, max_value=700, value=246, step=1)),
    "fbs": ("Fasting blood sugar > 120 mg/dl", "", "select", dict(options=[("Yes", 1), ("No", 0)])),
    "restecg": ("Resting ECG results", "0/1/2", "select", dict(options=[("0", 0), ("1", 1), ("2", 2)])),
    "thalach": ("Max heart rate achieved", "e.g. 150", "number", dict(min_value=50, max_value=250, value=150, step=1)),
    "exang": ("Exercise induced angina", "", "select", dict(options=[("Yes", 1), ("No", 0)])),
    "oldpeak": ("ST depression (exercise)", "e.g. 1.0", "float", dict(min_value=0.0, max_value=10.0, value=1.0, step=0.1)),
    "slope": ("Slope of peak exercise ST segment", "0/1/2", "select", dict(options=[("0", 0), ("1", 1), ("2", 2)])),
    "ca": ("Major vessels colored by flourosopy", "0-3", "select", dict(options=[("0", 0), ("1", 1), ("2", 2), ("3", 3)])),
    "target": ("Heart disease diagnosis present", "", "select", dict(options=[("Yes", 1), ("No", 0)])),
}


def render_field(field_name):
    """Render the right widget for a feature and return its numeric value."""
    if field_name not in FIELD_CONFIG:
        # Fallback for any feature not in the config
        return st.number_input(field_name, value=0.0)

    label, help_text, kind, kwargs = FIELD_CONFIG[field_name]

    if kind == "select":
        options = kwargs["options"]
        choice = st.selectbox(label, options, format_func=lambda o: o[0], help=help_text or None)
        return float(choice[1])
    elif kind == "float":
        return float(st.number_input(label, help=help_text or None, **kwargs))
    else:  # "number"
        return float(st.number_input(label, help=help_text or None, **kwargs))


def render_predict_tab(model, all_models, scaler, features):
    st.caption("Enter patient values below, then click Predict.")

    with st.form("patient_form"):
        cols = st.columns(2)
        values = {}
        for i, f in enumerate(features):
            with cols[i % 2]:
                values[f] = render_field(f)
        submitted = st.form_submit_button("Predict", use_container_width=True)

    if submitted:
        X_input = pd.DataFrame([[values[f] for f in features]], columns=features)
        X_scaled = scaler.transform(X_input)

        rows = []
        for name, mdl in all_models.items():
            pred = mdl.predict(X_scaled)[0]
            label = THALASSEMIA_LABELS.get(pred, str(pred))
            if hasattr(mdl, "predict_proba"):
                confidence = max(mdl.predict_proba(X_scaled)[0]) * 100
                rows.append({"Model": name, "Prediction": label, "Confidence": f"{confidence:.2f}%"})
            else:
                rows.append({"Model": name, "Prediction": label, "Confidence": "—"})

        st.subheader("Predictions by each model")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        best_pred = model.predict(X_scaled)[0]
        best_label = THALASSEMIA_LABELS.get(best_pred, str(best_pred))
        st.subheader("Best model prediction")
        st.success(best_label)


def render_insights_tab():
    st.caption("Charts generated by visualize.py from heart.csv.")

    if not os.path.isdir(VISUALS_DIR):
        st.info(
            f"No `{VISUALS_DIR}/` folder found yet. Run `python visualize.py` "
            "first, in the same directory as this app, then refresh the page."
        )
        return

    available = [f for f in VISUAL_CAPTIONS if os.path.exists(os.path.join(VISUALS_DIR, f))]
    if not available:
        st.info(
            f"`{VISUALS_DIR}/` exists but no expected chart files were found. "
            "Run `python visualize.py` to regenerate them."
        )
        return

    # Two charts per row
    for i in range(0, len(available), 2):
        row = available[i:i + 2]
        cols = st.columns(2)
        for col, fname in zip(cols, row):
            with col:
                st.image(
                    os.path.join(VISUALS_DIR, fname),
                    caption=VISUAL_CAPTIONS[fname],
                    use_container_width=True,
                )


def main():
    st.title("🫀 Thalassemia Status Predictor")

    try:
        model, all_models, scaler, features = load_artifacts()
    except FileNotFoundError:
        st.error(
            f"Couldn't find model files in `{MODEL_DIR}/`. "
            "Run `python train.py` first, in the same folder as this app."
        )
        return

    predict_tab, insights_tab = st.tabs(["🔮 Predict", "📊 Data Insights"])

    with predict_tab:
        render_predict_tab(model, all_models, scaler, features)

    with insights_tab:
        render_insights_tab()


if __name__ == "__main__":
    main()