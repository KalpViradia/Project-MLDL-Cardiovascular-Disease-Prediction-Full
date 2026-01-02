import streamlit as st
import sys
import os
import joblib

# Ensure parent directory is in path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import setup_page

setup_page("About the Model", "ℹ️")

st.markdown("""
<style>

/* Ensure nothing inside metric forces dark color */
div[data-testid="stMetric"] * {
    color: inherit;
}

/* Metric label */
div[data-testid="stMetric"] label {
    color: #94a3b8 !important;
    font-weight: 600;
}

/* Metric number (the real value) */
div[data-testid="stMetric"] > div {
    color: #38bdf8 !important;
    font-size: 2.4rem !important;
    font-weight: 900 !important;
}

/* Light mode override */
html[data-theme="light"] div[data-testid="stMetric"] > div {
    color: #0f172a !important;
}

</style>
""", unsafe_allow_html=True)

# Load model metrics
@st.cache_data
def load_metrics():
    try:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        metrics_path = os.path.join(curr_dir, "../../models/model_metrics.pkl")
        metrics = joblib.load(metrics_path)
        return metrics
    except FileNotFoundError:
        return None

metrics = load_metrics()

st.markdown('<h1 class="main-header">ℹ️ About the Model</h1>', unsafe_allow_html=True)

st.markdown("""
### 🧠 How It Works
This application uses a Machine Learning model to assess the probability of cardiovascular disease based on several health and lifestyle factors. The model was trained on a comprehensive dataset of patient records, learning patterns that correlate with heart health issues.

### 🔢 The Risk Score
The **Risk Probability** is a score between 0% and 100% that represents the likelihood of the presence of cardiovascular disease.

- **0% - 39% (Low Risk)**: Indicates a lower likelihood. Maintain healthy habits!
- **40% - 69% (Moderate/Warning)**: Suggests some risk factors are present. Consider consulting a professional to improve lifestyle.
- **70% - 100% (High Risk)**: Indicates a stronger likelihood. It is highly recommended to consult a healthcare provider for a thorough check-up.

### 📥 Inputs Used
The model considers the following parameters:
- **Age**: Risk generally increases with age.
- **Gender**: Biological sex can influence risk profiles.
- **Height & Weight (BMI)**: Body Mass Index is a key indicator.
- **Blood Pressure**: High systolic or diastolic pressure is a major risk factor.
- **Cholesterol & Glucose**: High levels can damage arteries over time.
- **Lifestyle**: Smoking, Alcohol consumption, and Physical Activity levels.

### ⚙️ Technology
- **Algorithm**: The core prediction engine is **LightGBM (Light Gradient Boosting Machine)**. It outperformed Random Forest, XGBoost, and Stacking Ensembles in our benchmarks. LightGBM is chosen for its efficiency, high accuracy, and ability to handle complex feature interactions.
- **Preprocessing & Frequency**: Robust scaling is applied to all numerical inputs.
- **Feature Engineering**: The model intelligently calculates derived health metrics like **Pulse Pressure** and **Mean Arterial Pressure (MAP)** from your blood pressure readings to enhance prediction capability.

### 📊 Model Performance
""")

# Display metrics if available
if metrics:
    # Add spacing before metrics
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display metrics in columns with improved styling
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Test Accuracy",
            value=f"{metrics['test_accuracy']*100:.2f}%",
            help="Percentage of correct predictions on the test set"
        )
    
    with col2:
        st.metric(
            label="ROC-AUC Score",
            value=f"{metrics['roc_auc']:.4f}",
            help="Area Under the ROC Curve (0.5 = random, 1.0 = perfect)"
        )
    
    with col3:
        st.metric(
            label="F1 Score",
            value=f"{metrics['f1_score']:.4f}",
            help="Harmonic mean of precision and recall"
        )
    
    # Add spacing after metrics
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Additional metrics in an expander for cleaner layout
    with st.expander("📈 **Additional Metrics**", expanded=False):
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown(f"""
            <div class="additional-metrics-content">
                <strong>Model Architecture</strong>
                <ul style="margin-top: 5px;">
                    <li><strong>Algorithm</strong>: {metrics['model_name']}</li>
                    <li><strong>Training Method</strong>: RandomizedSearchCV with 3-fold CV</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown(f"""
            <div class="additional-metrics-content">
                <strong>Detailed Performance</strong>
                <ul style="margin-top: 5px;">
                    <li><strong>Precision</strong>: {metrics['precision']:.4f} (Accuracy of positive predictions)</li>
                    <li><strong>Recall</strong>: {metrics['recall']:.4f} (Coverage of actual positives)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Performance context
    st.info(f"""
    ℹ️ **Performance Context**: The {metrics['model_name']} model achieved **{metrics['test_accuracy']*100:.2f}% accuracy** on unseen test data, 
    demonstrating strong generalization capability for cardiovascular disease risk prediction.
    """)
else:
    st.markdown("- **Test Accuracy**: Optimized using **RandomizedSearchCV** to find the best hyperparameters.")
    st.warning("⚠️ Model metrics not available. Train the model to see performance statistics.")

