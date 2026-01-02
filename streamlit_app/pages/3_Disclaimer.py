import streamlit as st
import sys
import os

# Ensure parent directory is in path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import setup_page

setup_page("Disclaimer", "⚠️")

# Using shared theme classes where appropriate


st.markdown('<h1 class="main-header" style="color:#D32F2F !important;">⚠️ Disclaimer</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box">
    <h3>1. Not a Medical Device</h3>
    <p>
        This purely educational application allows you to explore the capabilities of Machine Learning in healthcare.
        <strong>It is NOT a certified medical device</strong> and should NOT be used for self-diagnosis or treatment.
    </p>
</div>

<div class="disclaimer-box">
    <h3>2. Consult a Professional</h3>
    <p>
        The results provided by this tool are probabilistic estimates based on statistical patterns. They do not constitute a medical diagnosis.
        If you are concerned about your heart health, please <strong>consult a qualified healthcare provider</strong> immediately.
    </p>
</div>

<div class="disclaimer-box">
    <h3>3. AI Limitations</h3>
    <p>
        AI models can make errors. The predictions may not account for your specific medical history, genetics, or other individual factors not included in the input data.
        Always trust professional medical advice over an algorithm.
    </p>
</div>
""", unsafe_allow_html=True)

st.warning("By using this application, you acknowledge that you understand these limitations.")
