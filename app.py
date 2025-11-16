import streamlit as st
import pandas as pd
import joblib
import numpy as np

log_reg = joblib.load("logistic_regression_model.pkl")
rf = joblib.load("random_forest_classifier_model.pkl")
scaler = joblib.load("scaler.pkl")

feature_names = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach",
                 "exang","oldpeak","slope","ca","thal"]

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("🫀 Heart Disease Prediction App")

st.header("Enter Patient Details")
input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(feature, min_value=0.0, step=1.0, format="%.2f")

input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)

log_pred_proba = log_reg.predict_proba(input_scaled)[0][1]
rf_pred_proba = rf.predict_proba(input_scaled)[0][1]

log_pred = "Heart Disease" if log_pred_proba > 0.5 else "No Heart Disease"
rf_pred = "Heart Disease" if rf_pred_proba > 0.5 else "No Heart Disease"

st.subheader("Predictions")
st.write(f"**Logistic Regression:** {log_pred} (Probability: {log_pred_proba:.2f})")
st.write(f"**Random Forest:** {rf_pred} (Probability: {rf_pred_proba:.2f})")

st.subheader("Feature Contribution (Logistic Regression)")
coef = log_reg.coef_[0]
contribution = input_scaled[0] * coef

contribution_df = pd.DataFrame({
    "Feature": feature_names,
    "Contribution": contribution
}).sort_values("Contribution", ascending=False)

st.bar_chart(contribution_df.set_index("Feature"))
