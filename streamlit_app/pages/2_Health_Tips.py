import streamlit as st
import sys
import os

# Ensure parent directory is in path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import setup_page

setup_page("Health Tips", "🍏")

# Using shared visual hierarchy


st.markdown('<h1 class="main-header">🍏 Heart Health Tips</h1>', unsafe_allow_html=True)

st.markdown("""
Taking care of your heart is a lifelong commitment. Small changes in your daily routine can make a big difference.

### 🥗 Balanced Diet
- **Eat more:** Fruits, vegetables, whole grains, and healthy fats (like avocados, nuts, and olive oil).
- **Limit:** Salt (sodium), sugar, and saturated/trans fats.
- **Hydration:** Drink plenty of water throughout the day.

### 🏃‍♂️ Physical Activity
- Aim for at least **150 minutes of moderate-intensity exercise** per week (e.g., brisk walking, cycling).
- Incorporate strength training exercises at least 2 days a week.
- Stay active throughout the day, avoid prolonged sitting.

### 🚬 Lifestyle Choices
- **Quit Smoking:** Smoking is a major risk factor for heart disease. Quitting drastically reduces your risk.
- **Limit Alcohol:** Excessive drinking can raise blood pressure and lead to heart failure.
- **Manage Stress:** Chronic stress can damage arteries. Practice relaxation techniques like deep breathing or meditation.

### 😴 Sleep & Monitoring
- **Get Quality Sleep:** Aim for 7-9 hours of sleep per night.
- **Know Your Numbers:** Regularly check your blood pressure, cholesterol, and blood sugar levels.
- **Healthy Weight:** Maintaining a healthy BMI reduces the strain on your heart.

> [!NOTE]
> *These are general guidelines. Always follow the specific advice provided by your healthcare professional.*
""")
