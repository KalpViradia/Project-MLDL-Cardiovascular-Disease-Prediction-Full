import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import xgboost  # Necessary for loading XGBoost model
import lightgbm # Necessary for loading LightGBM model

from utils import setup_page

# Setup Page & Theme
setup_page("Cardio Risk Prediction", "💓", layout="wide")

# Load Model & Scaler
@st.cache_resource
def load_models():
    # Load model and scaler (updated to load LightGBM model)
    # Get the directory where this script is located
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct paths relative to this script
    model_path = os.path.join(curr_dir, "../models/best_model.pkl")
    scaler_path = os.path.join(curr_dir, "../models/scaler.pkl")
    thresh_path = os.path.join(curr_dir, "../models/threshold.pkl")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    try:
        threshold = joblib.load(thresh_path)
    except FileNotFoundError:
        threshold = 0.7
    return model, scaler, threshold

model, scaler, threshold = load_models()

# Header
st.markdown('<h1 class="main-header">💓 Cardiovascular Disease Risk Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Enter your details below to assess your heart health risk.</p>', unsafe_allow_html=True)

# Main Container with Theme Card Style
# st.markdown('<div class="theme-card">', unsafe_allow_html=True)

# Inputs
st.subheader("👤 Personal & Physical Details")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("👴 Age (years)", 1, 120, 45)
    gender = st.selectbox("⚥ Gender", [1,2], index=1, format_func=lambda x: "Female" if x==1 else "Male")

with col2:
    height = st.number_input("📏 Height (cm)", 100, 220, 165)
    weight = st.number_input("⚖️ Weight (kg)", 20, 200, 70)

with col3:
    bmi = weight / ((height / 100) ** 2)

    # BMI Category
    if bmi < 18.5:
        bmi_cat = "Underweight"
        bmi_color = "blue"
    elif bmi < 25:
        bmi_cat = "Normal"
        bmi_color = "green"
    elif bmi < 30:
        bmi_cat = "Overweight"
        bmi_color = "orange"
    else:
        bmi_cat = "Obese"
        bmi_color = "red"

    st.markdown(
        f"""
        <div style="margin-top:5px; font-size:1rem;">📌 BMI</div>
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="font-size:1.5rem; font-weight:bold;">{bmi:.1f}</div>
            <div style="color:{bmi_color}; font-weight:bold; font-size:1.5rem;">{bmi_cat}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.subheader("🩺 Vital Signs")
col4, col5 = st.columns(2)

with col4:
    ap_hi = st.number_input("💉 Systolic BP", 80, 200, 120)
    ap_lo = st.number_input("💉 Diastolic BP", 50, 150, 80)

with col5:
    cholesterol = st.selectbox("🧪 Cholesterol", ["Low", "Medium", "High"])
    gluc = st.selectbox("🧪 Glucose", ["Low", "Medium", "High"])

st.subheader("🚬 Lifestyle Factors")
col6, col7, col8 = st.columns(3)

with col6:
    smoke = st.selectbox("🚬 Smoker?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
with col7:
    alco = st.selectbox("🍷 Alcohol?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
with col8:
    active = st.selectbox("🏃 Active?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")

# Predict Button
if st.button("🔮 Predict My Risk"):
    # prepare row
    row = {
        "gender": gender,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": {"Low":1, "Medium":2, "High":3}[cholesterol],
        "gluc": {"Low":1, "Medium":2, "High":3}[gluc],
        "smoke": smoke,
        "alco": alco,
        "active": active,
        "age_years": age,
        "bmi": bmi,
    }

    df = pd.DataFrame([row])
    
    # Feature Engineering (Must match training logic)
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
    df['map'] = df['ap_lo'] + (df['pulse_pressure'] / 3)
    
    # Scale numerical columns
    num_cols = ["height","weight","ap_hi","ap_lo","bmi","age_years","pulse_pressure","map"]
    df[num_cols] = scaler.transform(df[num_cols])

    proba = model.predict_proba(df)[0,1]
    proba_percent = proba * 100
    pred = int(proba >= threshold)
    
    # Store results in session state
    st.session_state.prediction_result = {
        "proba_percent": proba_percent,
        "pred": pred
    }

# Display results if they exist in session state
if "prediction_result" in st.session_state and st.session_state.prediction_result is not None:
    result = st.session_state.prediction_result
    proba_percent = result["proba_percent"]
    pred = result["pred"]
    
    st.subheader("🔍 Result")

    # choose color
    if proba_percent < 40:
        bar_color = "#00c853"   # green
    elif proba_percent < 70:
        bar_color = "#ffeb3b"   # yellow
    else:
        bar_color = "#ff1744"   # red

    # probability bar
    st.markdown(f"""
        <div style="
            width:100%;
            height:28px;
            background:#e0e0e0;
            border-radius:14px;
            overflow:hidden;
            margin-top:10px;
            margin-bottom:20px;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);
        ">
            <div style="
                width:{proba_percent}%;
                height:100%;
                background:{bar_color};
                border-radius:14px;
                transition: width 0.6s ease;
                background-image: linear-gradient(45deg,rgba(255,255,255,.15) 25%,transparent 25%,transparent 50%,rgba(255,255,255,.15) 50%,rgba(255,255,255,.15) 75%,transparent 75%,transparent);
                background-size: 1rem 1rem;
            ">
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Custom Result Display
    st.markdown(f"""
        <div style="text-align: center; margin-bottom: 20px;">
            <p style="font-size: 1.2rem; margin-bottom: 5px; color: var(--text-color);">Risk Probability</p>
            <h1 style="font-size: 4.5rem; color: {bar_color}; margin: 0; font-weight: 800; text-shadow: 0 2px 4px rgba(0,0,0,0.1);">{proba_percent:.2f}%</h1>
        </div>
    """, unsafe_allow_html=True)

    if pred == 1:
        st.markdown(f"""
            <div style="
                background-color: rgba(255, 23, 68, 0.1); 
                border: 1px solid #ff1744; 
                padding: 20px; 
                border-radius: 12px; 
                text-align: center; 
                margin-top: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            ">
                <h2 style="color: #ff1744; margin: 0; display: flex; align-items: center; justify-content: center; gap: 10px;">
                    ⚠️ High Risk Detected
                </h2>
                <p style="margin-top: 10px; font-size: 1.1rem; opacity: 0.9;">
                    The model predicts a significant likelihood of cardiovascular disease. 
                    <strong>Please consult a healthcare professional for a thorough check-up.</strong>
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style="
                background-color: rgba(0, 200, 83, 0.1); 
                border: 1px solid #00c853; 
                padding: 20px; 
                border-radius: 12px; 
                text-align: center; 
                margin-top: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            ">
                <h2 style="color: #00c853; margin: 0; display: flex; align-items: center; justify-content: center; gap: 10px;">
                    ✔️ Low Risk Detected
                </h2>
                <p style="margin-top: 10px; font-size: 1.1rem; opacity: 0.9;">
                    Your results look good! Keep up the healthy lifestyle habits.
                </p>
            </div>
        """, unsafe_allow_html=True)

# close inner wrapper
st.markdown('</div>', unsafe_allow_html=True)
