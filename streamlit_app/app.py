import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load Model & Scaler
@st.cache_resource
def load_models():
    model = joblib.load("../models/best_model.pkl")
    scaler = joblib.load("../models/scaler.pkl")
    try:
        threshold = joblib.load("../models/threshold.pkl")
    except FileNotFoundError:
        threshold = 0.7
    return model, scaler, threshold

model, scaler, threshold = load_models()

# Page Config - use wide layout
st.set_page_config(
    page_title="Cardio Risk Prediction",
    layout="wide",            # allow full width so we can control container size
    initial_sidebar_state="expanded"
)

# Custom CSS - override Streamlit container and style
st.markdown("""
<style>
/* Make Streamlit's main block container wider and centered */
.main > div.block-container {
    max-width: 1200px !important;   /* <-- increase this (900 / 1200 / 1400) */
    padding-top: 1rem !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* Additional container to control inner width (optional) */
.app-inner-container {
    max-width: 1100px;
    margin: 0 auto;
}

/* tighten input spacing (keep compact) */
.st-emotion-cache-16idsys,
.st-emotion-cache-1vbkxk1,
.st-emotion-cache-1r6slb0,
.st-emotion-cache-1cdf7s8,
.st-emotion-cache-1y4p8pa {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* small visual tweaks */
.main-header {
    font-size: 2.6rem;
    color: #2E7D32;
    text-align: center;
    margin-bottom: 0.5rem;
}
.sub-header {
    font-size: 1.05rem;
    color: #999;
    text-align: center;
    margin-bottom: 1rem;
}

.prediction-card {
    background: #f8f9fa;
    padding: 1.1rem;
    border-radius: 10px;
    border-left: 5px solid #4CAF50;
    margin-top: 8px;
}
.risk-high { border-left-color: #f44336 !important; }
.risk-low { border-left-color: #4CAF50 !important; }

/* make selectboxes/inputs stretch nicely inside columns */
[data-testid="stVerticalBlock"] .stNumberInput, 
[data-testid="stVerticalBlock"] .stSelectbox {
    width: 100%;
}

/* responsive: reduce width on small screens */
@media (max-width: 900px) {
  .main > div.block-container {
      padding-left: 1rem !important;
      padding-right: 1rem !important;
  }
  .app-inner-container { max-width: 100% !important; }
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">💓 Cardiovascular Disease Risk Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Enter your details below to assess your heart health risk.</p>', unsafe_allow_html=True)

# Optional inner wrapper (controls how much area the form covers inside the block-container)
st.markdown('<div class="app-inner-container">', unsafe_allow_html=True)

# Inputs
st.subheader("👤 Personal & Physical Details")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("👴 Age (years)", 1, 120, 45)
    gender = st.selectbox("⚥ Gender", [1,2], format_func=lambda x: "Female" if x==1 else "Male")

with col2:
    height = st.number_input("📏 Height (cm)", 100, 220, 165)
    weight = st.number_input("⚖️ Weight (kg)", 20, 200, 70)

with col3:
    bmi = weight / ((height / 100) ** 2)
    st.metric("📌 BMI", f"{bmi:.1f}")

st.subheader("🩺 Vital Signs")
col4, col5 = st.columns(2)

with col4:
    ap_hi = st.number_input("💉 Systolic BP", 80, 200, 120)
    ap_lo = st.number_input("💉 Diastolic BP", 50, 150, 80)

with col5:
    cholesterol = st.selectbox("🧪 Cholesterol", [1,2,3])
    gluc = st.selectbox("🧪 Glucose", [1,2,3])

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
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active,
        "age_years": age,
        "bmi": bmi,
    }

    df = pd.DataFrame([row])
    df[["height","weight","ap_hi","ap_lo","bmi","age_years"]] = scaler.transform(
        df[["height","weight","ap_hi","ap_lo","bmi","age_years"]]
    )

    proba = model.predict_proba(df)[0,1]
    proba_percent = proba * 100
    pred = int(proba >= threshold)

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
            background:#2f2f2f;
            border-radius:10px;
            overflow:hidden;
            margin-top:10px;
            margin-bottom:6px;
        ">
            <div style="
                width:{proba_percent}%;
                height:100%;
                background:{bar_color};
                transition: width 0.6s ease;
            ">
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.metric("Risk Probability", f"{proba_percent:.2f}%")

    if pred == 1:
        st.error("⚠️ High Risk of Cardiovascular Disease")
    else:
        st.success("✔️ Low Risk of Cardiovascular Disease")

# close inner wrapper
st.markdown('</div>', unsafe_allow_html=True)
