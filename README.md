# Cardio Risk Prediction

Predicts the risk of cardiovascular disease using clinical measurements (age, blood pressure, cholesterol, BMI, lifestyle factors) based on the **Cardiovascular Disease Dataset** from Kaggle.

Dataset: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

This project includes:

- Logistic Regression **from scratch** (for learning purposes)
- **Random Forest**, **XGBoost**, and **LightGBM** classifiers for comparative analysis (with one selected as the final deployment model)
- Feature engineering (age in years, BMI)
- Scaling of numeric features with `StandardScaler`
- Evaluation (accuracy, precision, recall, F1, confusion matrix, ROC–AUC)
- A **Streamlit** web app for interactive predictions

---

## Project Structure

```text
cardio-risk-prediction/
├─ data/
│  ├─ processed/
│  │  ├─ cleaned_data.csv      # Preprocessed dataset
│  ├─ raw/                     # Original dataset
├─ models/
│  ├─ best_model.pkl          # Trained classifier (RF/XGB/LGBM)
│  ├─ scaler.pkl              # Fitted StandardScaler
│  ├─ threshold.pkl           # Optimal probability threshold
│  ├─ best_threshold.pkl      # (Variant) Optimal threshold
│  ├─ model_metrics.pkl       # Performance metrics
├─ notebooks/
│  ├─ 01_exploratory_data_analysis.ipynb
│  ├─ 02_preprocessing.ipynb
│  ├─ 03_model_training.ipynb
├─ src/
│  ├─ preprocess.py           # Data preprocessing script
│  ├─ train_model.py          # Model training script
├─ streamlit_app/
│  ├─ pages/                  # Multi-page app support
│  │  ├─ 1_About_the_Model.py
│  │  ├─ 2_Health_Tips.py
│  │  ├─ 3_Disclaimer.py
│  ├─ app.py                  # Main Streamlit application
│  ├─ utils.py                # Helper functions and themes
├─ check_model.py             # Utility to check model status
├─ requirements.txt
├─ README.md
```

> Note: Most preprocessing is done in the notebooks and saved into
> `data/processed/cleaned_data.csv`. The Python script `train_model.py`
> loads this file instead of redoing the notebook steps.

---

## Installation

1. **Clone** or copy this project into your local machine.
2. (Recommended) Create a virtual environment.

   ```bash
   python -m venv venv
   # Activate (Windows PowerShell)
   venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

## Data & Preprocessing

1. Download the original dataset from Kaggle:

   - https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

2. Use the notebooks in `notebooks/` (especially `02_preprocessing.ipynb`)
   to clean and preprocess the raw data. The notebook exports a processed
   CSV:

   - `data/processed/cleaned_data.csv`

3. The processed file has columns similar to:

   ```text
   id, gender, height, weight, ap_hi, ap_lo,
   cholesterol, gluc, smoke, alco, active,
   cardio, age_years, bmi
   ```

4. The training script uses:

   - Features `X` = all columns except `cardio` and `id`
   - Target `y` = `cardio` (0 = low risk, 1 = high risk)

---

## Training the Model

Training is handled by `src/train_model.py`.

Run from the **project root**:

```bash
python src/train_model.py
```

What this script does:

- Loads `data/processed/cleaned_data.csv`
- Drops `cardio` and `id` from the features
- Splits data into train/test using `train_test_split` with stratification
- Trains:
  - A **Custom Logistic Regression** from scratch (for demonstration)
  - A **Random Forest** classifier on scaled features
- Scales numeric columns using `StandardScaler`:

  - `height`, `weight`, `ap_hi`, `ap_lo`, `bmi`, `age_years`

- Evaluates the Random Forest on the test set:

  - Accuracy, precision, recall, F1
  - Classification report
  - Confusion matrix
  - ROC curve & ROC–AUC

- Computes the **optimal probability threshold** using Youden's J statistic
  from the ROC curve and saves it to `models/threshold.pkl`.
- Saves artifacts to `models/`:

  - `best_model.pkl`  – trained RandomForest
  - `scaler.pkl`      – fitted StandardScaler
  - `threshold.pkl`   – probability cutoff used by the app

After running, you should see evaluation metrics printed to the console
and the model files updated.

---

## Running the Streamlit App

The web app is defined in `streamlit_app/app.py` and loads the trained
model, scaler, and threshold.

From the project root, run:

```bash
streamlit run streamlit_app/app.py
```

Then open the local URL shown in the terminal (typically
`http://localhost:8501`).

### Features used in the app

The UI asks for:

- Age in years
- Height (cm)
- Weight (kg)
- Systolic blood pressure (`ap_hi`)
- Diastolic blood pressure (`ap_lo`)
- Cholesterol level (1 = normal, 2 = above normal, 3 = well above)
- Glucose level (1 = normal, 2 = above normal, 3 = well above)
- Smoking (0/1)
- Alcohol intake (0/1)
- Physical activity (0/1)
- Gender (1 = female, 2 = male)

The app computes BMI internally and builds a feature vector matching the
training columns.

### Prediction logic

- Numeric features are scaled with the same `StandardScaler` used during
  training.
- The model outputs the probability of **high cardiovascular risk**.
- The probability is compared against `threshold.pkl` (optimal threshold
  from training). If `proba >= threshold` → **High Risk**, else → **Low Risk**.
- The app displays both the predicted class and the risk probability.

---

## Notes on Model Behaviour

- The dataset is **imbalanced** and biased toward high-risk cases.
- We use `class_weight="balanced"` for the Random Forest to handle class
  imbalance during training.
- Even with this, some realistic low-risk profiles might still receive
  moderate probabilities. Adjusting the threshold or model
  hyperparameters can tune sensitivity vs. specificity.

If you want to experiment:

- Change the RandomForest parameters in `train_model.py`.
- Remove or modify `class_weight="balanced"`.
- Manually override the threshold in `streamlit_app/app.py`.

---

## Requirements

Main Python dependencies (see `requirements.txt`):

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- streamlit
- joblib
- xgboost
- lightgbm
