import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
TARGET_FEATURES = ["Temperature (°C)", "Humidity (%)", "Wind Speed (km/h)", "Wind Direction (°)"]
TIMESTAMP_COL = "Timestamp"
LAG_FEATURES = 24 # Define the constant here

def load_data(city_name):
    """Loads CSV data for a given city."""
    file_path = os.path.join(DATA_DIR, f"{city_name}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found for city: {city_name} at {file_path}")
    df = pd.read_csv(file_path)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df.sort_values(TIMESTAMP_COL, inplace=True)
    df.set_index(TIMESTAMP_COL, inplace=True)
    # Ensure all target columns exist
    for col in TARGET_FEATURES:
        if col not in df.columns:
            raise ValueError(f"Missing expected column '{col}' in {city_name}.csv")
    return df[TARGET_FEATURES] # Return only target features, indexed by timestamp

def create_features(df, lag_features=LAG_FEATURES): # Optionally use the constant as default
    """Creates lag features and time-based features."""
    df_feat = df.copy()
    # Lag features for each target variable
    for col in TARGET_FEATURES:
        for lag in range(1, lag_features + 1):
            df_feat[f'{col}_lag_{lag}'] = df_feat[col].shift(lag)

    # Time-based features (use index)
    df_feat['hour'] = df_feat.index.hour
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['month'] = df_feat.index.month
    df_feat['dayofyear'] = df_feat.index.dayofyear

    # Drop rows with NaNs created by lagging
    df_feat.dropna(inplace=True)
    return df_feat

def prepare_data_for_training(city_name, lag_features=LAG_FEATURES): # Optionally use the constant as default
    """Loads and preprocesses data for model training."""
    df = load_data(city_name)
    df_processed = create_features(df, lag_features)

    # Separate features (X) and targets (y)
    X = df_processed.drop(columns=TARGET_FEATURES)
    y = df_processed[TARGET_FEATURES]

    return X, y

def prepare_data_for_prediction(df_history, lag_features=LAG_FEATURES): # Optionally use the constant as default
    """Prepares the most recent data slice for making the next prediction step."""
    if len(df_history) < lag_features:
         raise ValueError(f"Need at least {lag_features} hours of history for prediction.")
    # Take the most recent 'lag_features' hours
    recent_data = df_history.iloc[-lag_features:].copy()
    # Create features based on this slice - we only need the *last* row's features
    df_feat = create_features(recent_data, lag_features) # This will likely result in 1 row if len(recent_data) == lag_features
    if df_feat.empty:
         # If create_features dropped the only row due to NaNs (shouldn't happen with correct slicing)
         # Re-create features using slightly more data if needed, ensuring the last point is valid
         temp_data = df_history.iloc[-(lag_features+1):].copy() # Get one extra point
         df_feat_temp = create_features(temp_data, lag_features)
         if df_feat_temp.empty:
              raise ValueError("Could not create valid feature row for prediction.")
         return df_feat_temp.iloc[-1:].drop(columns=TARGET_FEATURES) # Return last row as DataFrame

    return df_feat.iloc[-1:].drop(columns=TARGET_FEATURES) # Return last row as DataFrame