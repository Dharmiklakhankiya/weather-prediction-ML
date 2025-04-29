import pandas as pd
import numpy as np
import joblib
import os
from datetime import timedelta
from app.utils.preprocess import load_data, prepare_data_for_prediction, TARGET_FEATURES, TIMESTAMP_COL, LAG_FEATURES

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def load_model(city_name, model_name):
    """Loads a trained model."""
    model_filename = os.path.join(MODELS_DIR, f"{city_name}_{model_name}.pkl")
    if not os.path.exists(model_filename):
        raise FileNotFoundError(f"Model file not found: {model_filename}")
    model = joblib.load(model_filename)
    return model

def make_predictions(city_name, model_name, hours_to_predict=48):
    """Loads a model and makes iterative predictions."""
    model = load_model(city_name, model_name)
    df_history = load_data(city_name) # Load full history to get latest data

    if df_history.empty:
        raise ValueError(f"No historical data found for {city_name} to make predictions.")

    predictions = []
    current_history = df_history.copy() # Work with a copy
    last_timestamp = current_history.index.max()

    print(f"Starting prediction for {city_name} using {model_name} for {hours_to_predict} hours.")
    print(f"Latest data point timestamp: {last_timestamp}")

    for i in range(hours_to_predict):
        # 1. Prepare the input features from the *current* history
        try:
            X_pred_input = prepare_data_for_prediction(current_history, lag_features=LAG_FEATURES)
        except ValueError as e:
             print(f"Error preparing data at step {i+1}/{hours_to_predict}: {e}")
             print(f"History length: {len(current_history)}")
             # Handle cases where history might be too short after iterative additions if something went wrong
             break # Stop prediction if input cannot be prepared

        # 2. Make prediction for the next hour
        next_hour_pred_values = model.predict(X_pred_input)[0] # Get the single prediction array

        # 3. Store the prediction
        next_timestamp = last_timestamp + timedelta(hours=i + 1)
        pred_record = {TIMESTAMP_COL: next_timestamp}
        for idx, feature in enumerate(TARGET_FEATURES):
             # Ensure predictions are sensible (e.g., humidity 0-100) - basic clipping
             if feature == "Humidity (%)":
                 pred_record[feature] = np.clip(next_hour_pred_values[idx], 0, 100)
             elif feature == "Wind Direction (°)":
                 pred_record[feature] = np.clip(next_hour_pred_values[idx], 0, 360) # Or use modulo 360 if needed
             else:
                 pred_record[feature] = next_hour_pred_values[idx]
        predictions.append(pred_record)

        # 4. Append the prediction to history for the next iteration
        # Create a DataFrame for the new prediction to append correctly
        new_row_df = pd.DataFrame([pred_record])
        new_row_df[TIMESTAMP_COL] = pd.to_datetime(new_row_df[TIMESTAMP_COL])
        new_row_df.set_index(TIMESTAMP_COL, inplace=True)
        # Ensure columns match exactly before concatenating
        new_row_df = new_row_df[TARGET_FEATURES]

        # Use concat instead of append
        current_history = pd.concat([current_history, new_row_df])


    print(f"Finished prediction. Generated {len(predictions)} data points.")
    return pd.DataFrame(predictions)
