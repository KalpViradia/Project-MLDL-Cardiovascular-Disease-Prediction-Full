import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV

import os

# 1. Load Preprocessed Data
# Use absolute path relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "../data/processed/cleaned_data.csv")
data = pd.read_csv(DATA_PATH)

# --- Feature Engineering ---
# (Handled in preprocess.py: bmi, pulse_pressure, map are already in the loaded CSV)
# data['pulse_pressure'] = data['ap_hi'] - data['ap_lo']
# data['map'] = data['ap_lo'] + (data['pulse_pressure'] / 3)

X = data.drop(columns=["cardio", "id"])
y = data["cardio"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 2. Logistic Regression From Scratch
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class CustomLogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)  # bias term
        self.weights = np.zeros(X.shape[1])
        for _ in range(self.epochs):
            z = np.dot(X, self.weights)
            y_pred = sigmoid(z)
            gradient = np.dot(X.T, (y_pred - y)) / y.size
            self.weights -= self.lr * gradient

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        probabilities = sigmoid(np.dot(X, self.weights))
        return (probabilities >= 0.5).astype(int)

# Train Logistic Regression from scratch
X_train_np = X_train.values
X_test_np  = X_test.values

scratch_model = CustomLogisticRegression(lr=0.01, epochs=1500)
scratch_model.fit(X_train_np, y_train)

y_pred_scratch = scratch_model.predict(X_test_np)

print("Custom Logistic Regression Metrics:")
print("Accuracy :", accuracy_score(y_test, y_pred_scratch))
print("Precision:", precision_score(y_test, y_pred_scratch))
print("Recall   :", recall_score(y_test, y_pred_scratch))
print("F1 Score :", f1_score(y_test, y_pred_scratch))
print("\n" + "="*50 + "\n")

# 3. Random Forest Classifier (Best Model) with Scaling

# Identify numeric columns to scale (including new features)
num_cols = ['height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'age_years', 'pulse_pressure', 'map']

# Copy to avoid modifying original
X_train_scaled = X_train.copy()
X_test_scaled  = X_test.copy()

# Initialize and fit scaler
scaler = StandardScaler()
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols]  = scaler.transform(X_test[num_cols])

# Train Random Forest on scaled data
# 3. Model Training with Hyperparameter Tuning & Comparison
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier

# --- Random Forest ---
print("\n[Random Forest] Starting Expanded Hyperparameter Tuning...")
rf = RandomForestClassifier(random_state=42, class_weight="balanced")

rf_param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 8],
    'bootstrap': [True, False]
}

rf_random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_param_dist,
    n_iter=30,  # Increased iterations
    cv=3,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_random_search.fit(X_train_scaled, y_train)
best_rf_model = rf_random_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print(f"[Random Forest] Best Params: {rf_random_search.best_params_}")
print(f"[Random Forest] Test Accuracy: {rf_accuracy:.4f}")

# --- XGBoost ---
print("\n[XGBoost] Starting Expanded Hyperparameter Tuning...")
xgb = XGBClassifier(eval_metric='logloss', random_state=42)

xgb_param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [0.01, 0.1, 1, 10]
}

xgb_random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=xgb_param_dist,
    n_iter=30, # Increased iterations
    cv=3,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

xgb_random_search.fit(X_train_scaled, y_train)
best_xgb_model = xgb_random_search.best_estimator_
y_pred_xgb = best_xgb_model.predict(X_test_scaled)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)

print(f"[XGBoost] Best Params: {xgb_random_search.best_params_}")
print(f"[XGBoost] Test Accuracy: {xgb_accuracy:.4f}")

# --- LightGBM ---
from lightgbm import LGBMClassifier
print("\n[LightGBM] Starting Hyperparameter Tuning...")
lgbm = LGBMClassifier(random_state=42, verbose=-1)

lgbm_param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'num_leaves': [20, 31, 50, 80],
    'min_child_samples': [20, 50, 100],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
}

lgbm_random_search = RandomizedSearchCV(
    estimator=lgbm,
    param_distributions=lgbm_param_dist,
    n_iter=30,
    cv=3,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

lgbm_random_search.fit(X_train_scaled, y_train)
best_lgbm_model = lgbm_random_search.best_estimator_
y_pred_lgbm = best_lgbm_model.predict(X_test_scaled)
lgbm_accuracy = accuracy_score(y_test, y_pred_lgbm)

print(f"[LightGBM] Best Params: {lgbm_random_search.best_params_}")
print(f"[LightGBM] Test Accuracy: {lgbm_accuracy:.4f}")

# --- Stacking Classifier (Advanced Ensemble) ---
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

print("\n[Ensemble] Training Stacking Classifier...")
estimators = [
    ('rf', best_rf_model),
    ('xgb', best_xgb_model),
    ('lgbm', best_lgbm_model)
]

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=3,
    n_jobs=-1
)

stacking_clf.fit(X_train_scaled, y_train)
y_pred_stacking = stacking_clf.predict(X_test_scaled)
stacking_accuracy = accuracy_score(y_test, y_pred_stacking)
print(f"[Stacking Classifier] Test Accuracy: {stacking_accuracy:.4f}")

# --- Comparison & Selection ---
print("\n" + "="*50)
print("--- Model Comparison ---")
# Compare accuracy of all models
models = {
    "XGBoost": (xgb_accuracy, best_xgb_model, y_pred_xgb),
    "Random Forest": (rf_accuracy, best_rf_model, y_pred_rf),
    "LightGBM": (lgbm_accuracy, best_lgbm_model, y_pred_lgbm),
    "Stacking Classifier": (stacking_accuracy, stacking_clf, y_pred_stacking)
}

best_model_name = max(models, key=lambda k: models[k][0])
final_accuracy, best_model, y_pred_final = models[best_model_name]

print(f"🏆 Winning Model: {best_model_name} with Accuracy: {final_accuracy:.4f}")
print("="*50 + "\n")

# 4. Evaluation (of the winner)
print(f"Evaluating Best Model ({best_model_name})...")

print("Classification Report:")
print(classification_report(y_test, y_pred_final))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_final)
print(cm)

# ROC-AUC
if hasattr(best_model, "predict_proba"):
    y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
else:
    # Fallback if model doesn't support predict_proba (unlikely for these)
    y_prob = best_model.decision_function(X_test_scaled)

roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Calculate additional metrics
precision = precision_score(y_test, y_pred_final)
recall = recall_score(y_test, y_pred_final)
f1 = f1_score(y_test, y_pred_final)

# Save the best model
models_dir = os.path.join(SCRIPT_DIR, "../models")
os.makedirs(models_dir, exist_ok=True)

print(f"Saving {best_model_name} model to {models_dir}...")
joblib.dump(best_model, os.path.join(models_dir, "best_model.pkl"))
joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

# Save model metrics for display in About page
model_metrics = {
    "model_name": best_model_name,
    "test_accuracy": final_accuracy,
    "roc_auc": roc_auc,
    "precision": precision,
    "recall": recall,
    "f1_score": f1
}
joblib.dump(model_metrics, os.path.join(models_dir, "model_metrics.pkl"))
print(f"Saved model metrics to {os.path.join(models_dir, 'model_metrics.pkl')}")

# Save optimal threshold (using Youden's J statistic)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
J = tpr - fpr
ix = np.argmax(J)
best_thresh = thresholds[ix]
print(f"Best Threshold (Youden's J): {best_thresh:.4f}")
joblib.dump(best_thresh, os.path.join(models_dir, "best_threshold.pkl"))
# Also save generic 'threshold.pkl' for compatibility
joblib.dump(best_thresh, os.path.join(models_dir, "threshold.pkl"))

print("Done! Model training and comparison complete.")

