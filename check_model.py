import joblib
import os
import sys

try:
    model_path = os.path.join("models", "best_model.pkl")
    model = joblib.load(model_path)
    print(f"Model type: {type(model).__name__}")
except Exception as e:
    print(f"Error: {e}")
