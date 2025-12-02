"""
train_model.py
----------------
Trains ML models for Cardiovascular Disease Risk Prediction.
Includes:
1. Logistic Regression from scratch (Week-3 requirement)
2. Random Forest Classifier (final deployment model)
3. StandardScaler for numeric features
4. Evaluation metrics, confusion matrix, ROC–AUC
5. Saves best model as 'best_model.pkl' and scaler as 'scaler.pkl'
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------------------------
# 1. Load Preprocessed Data
# ---------------------------
data = pd.read_csv("../data/processed/cleaned_data.csv")
X = data.drop(columns=["cardio", "id"])
y = data["cardio"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# ---------------------------
# 2. Logistic Regression From Scratch
# ---------------------------
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

# ---------------------------
# 3. Random Forest Classifier (Best Model) with Scaling
# ---------------------------

# Identify numeric columns to scale
num_cols = ['height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'age_years']

# Copy to avoid modifying original
X_train_scaled = X_train.copy()
X_test_scaled  = X_test.copy()

# Initialize and fit scaler
scaler = StandardScaler()
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols]  = scaler.transform(X_test[num_cols])

# Train Random Forest on scaled data
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=4,
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_train_scaled, y_train)

# Predictions
y_pred_rf = rf.predict(X_test_scaled)
y_proba_rf = rf.predict_proba(X_test_scaled)[:, 1]

# ---------------------------
# 4. Evaluation
# ---------------------------
print("Random Forest Metrics (Scaled Features):")
print("Accuracy :", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall   :", recall_score(y_test, y_pred_rf))
print("F1 Score :", f1_score(y_test, y_pred_rf))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Random Forest")
plt.show()

# ROC–AUC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba_rf)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label="Random Forest")
plt.plot([0,1], [0,1], '--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Random Forest")
plt.legend()
plt.show()

roc_auc = roc_auc_score(y_test, y_proba_rf)
print(f"ROC–AUC Score: {roc_auc:.4f}")

j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]
print(f"Best threshold (Youden's J): {best_threshold:.4f}")

# ---------------------------
# 5. Save Best Model and Scaler
# ---------------------------
joblib.dump(rf, "../models/best_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")
joblib.dump(best_threshold, "../models/threshold.pkl")
print("Random Forest model, scaler, and threshold saved successfully!")