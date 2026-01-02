import pandas as pd
import numpy as np
import os

def preprocess_data():
    """
    Reads raw data, cleans it (outliers, duplicates), performs feature engineering,
    and saves the processed data.
    """
    # 1. Setup Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(script_dir, "../data/raw/cardio_train.csv")
    processed_data_path = os.path.join(script_dir, "../data/processed/cleaned_data.csv")

    # Ensure processed directory exists
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

    print(f"Loading raw data from: {raw_data_path}")
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data file not found at {raw_data_path}")

    # Load Data (Separator is ';')
    df = pd.read_csv(raw_data_path, sep=";")
    print(f"Initial shape: {df.shape}")

    # 2. Data Cleaning
    
    # duplicates
    initial_dupes = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    print(f"Removed {initial_dupes} duplicate rows.")

    # Convert age to years
    df['age_years'] = (df['age'] / 365).astype(int)
    df.drop(columns=['age'], inplace=True)

    # 3. Outlier Removal (Based on notebook & domain knowledge)
    # Height: 120-220 cm
    df = df[(df['height'] >= 120) & (df['height'] <= 220)]
    
    # Weight: 30-200 kg
    df = df[(df['weight'] >= 30) & (df['weight'] <= 200)]
    
    # Systolic BP (ap_hi): 80-200
    df = df[(df['ap_hi'] >= 80) & (df['ap_hi'] <= 200)]
    
    # Diastolic BP (ap_lo): 50-150
    df = df[(df['ap_lo'] >= 50) & (df['ap_lo'] <= 150)]
    
    print(f"Shape after filtering outliers: {df.shape}")

    # 4. Feature Engineering
    
    # BMI = weight (kg) / height (m)^2
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    
    # Pulse Pressure = ap_hi - ap_lo
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
    
    # Mean Arterial Pressure (MAP) = ap_lo + (pulse_pressure / 3)
    df['map'] = df['ap_lo'] + (df['pulse_pressure'] / 3)

    print("Added features: bmi, pulse_pressure, map")

    # 5. Save Data
    df.to_csv(processed_data_path, index=False)
    print(f"Processed data saved to: {processed_data_path}")

if __name__ == "__main__":
    preprocess_data()
