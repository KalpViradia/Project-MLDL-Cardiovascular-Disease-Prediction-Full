import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load Model & Scaler
model = joblib.load("../models/best_model.pkl")
scaler = joblib.load("../models/scaler.pkl")
try:
    threshold = joblib.load("../models/threshold.pkl")
except FileNotFoundError:
    threshold = 0.7

# Page Configuration
st.set_page_config(
    page_title="Cardio Risk Prediction",
    layout="centered"
)

# Header
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>💓 Cardiovascular Disease Risk Prediction</h1>
    <p style='text-align: center;'>Enter patient details to predict the risk of cardiovascular disease.</p>
""", unsafe_allow_html=True)

# Input Form
st.header("📝 Patient Health Details")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=45)
    height = st.number_input("Height (cm)", min_value=100, max_value=220, value=165)
    weight = st.number_input("Weight (kg)", min_value=20, max_value=200, value=70)

with col2:
    ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=80, max_value=200, value=120)
    ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=50, max_value=150, value=80)
    cholesterol = st.selectbox("Cholesterol Level (1=Normal, 2=Above Normal, 3=Well Above)", [1, 2, 3])
    gluc = st.selectbox("Glucose Level (1=Normal, 2=Above Normal, 3=Well Above)", [1, 2, 3])

col3, col4 = st.columns(2)
with col3:
    smoke = st.selectbox("Smoking?", [0, 1])
    alco = st.selectbox("Alcohol Intake?", [0, 1])

with col4:
    active = st.selectbox("Physically Active?", [0, 1])
    gender = st.selectbox("Gender (1 = Female, 2 = Male)", [1, 2])

# BMI Calculation
bmi = weight / ((height / 100) ** 2)
st.write(f"📌 **BMI:** {bmi:.2f}")

# Prediction
if st.button("Predict Risk"):

    # Determine whether the trained model expects an 'id' feature
    feature_names = getattr(model, "feature_names_in_", None)

    if feature_names is not None and "id" in feature_names:
        # Model trained with 'id' column present
        row = {
            "id": 0,  # dummy id
            "gender": gender,
            "height": height,
            "weight": weight,
            "ap_hi": ap_hi,
            "ap_lo": ap_lo,
            "cholesterol": cholesterol,
            "gluc": gluc,
            "smoke": smoke,
            "alco": alco,
            "active": active,
            "age_years": age,
            "bmi": bmi,
        }
    else:
        # Model trained without 'id' column
        row = {
            "gender": gender,
            "height": height,
            "weight": weight,
            "ap_hi": ap_hi,
            "ap_lo": ap_lo,
            "cholesterol": cholesterol,
            "gluc": gluc,
            "smoke": smoke,
            "alco": alco,
            "active": active,
            "age_years": age,
            "bmi": bmi,
        }

    # Build input DataFrame
    input_df = pd.DataFrame([row])

    num_cols = ["height", "weight", "ap_hi", "ap_lo", "bmi", "age_years"]
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    proba = model.predict_proba(input_df)[0, 1]
    prediction = int(proba >= threshold)

    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.error("⚠ High Risk of Cardiovascular Disease")
    else:
        st.success("✔ Low Risk of Cardiovascular Disease")
    st.write(f"Predicted risk probability: {proba:.2f} (threshold: {threshold:.2f})")

# Footer
st.markdown("""
    <hr>
    <p style='text-align:center; font-size: 13px;'>
        Built with ❤️ using Machine Learning & Streamlit
    </p>
""", unsafe_allow_html=True)