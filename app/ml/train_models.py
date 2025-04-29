import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from app.utils.preprocess import prepare_data_for_training, TARGET_FEATURES

CITIES = ["ahmedabad", "mumbai", "delhi", "bengaluru"]
LAG_FEATURES = 24
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

os.makedirs(MODELS_DIR, exist_ok=True)

models_config = {
    "LightGBM": MultiOutputRegressor(lgb.LGBMRegressor(random_state=42)),
    "CatBoost": MultiOutputRegressor(cb.CatBoostRegressor(random_state=42, verbose=0)),
    "ExtraTrees": ExtraTreesRegressor(random_state=42, n_estimators=100),
    "XGBoost": MultiOutputRegressor(xgb.XGBRegressor(random_state=42, objective='reg:squarederror')),
    "HistGradientBoosting": MultiOutputRegressor(HistGradientBoostingRegressor(random_state=42))
}

def train_all_models():
    for city in CITIES:
        print(f"\n--- Processing City: {city.title()} ---")
        city_file = os.path.join(DATA_DIR, f"{city}.csv")
        if not os.path.exists(city_file):
            print(f"⚠️ Data file not found for {city}. Skipping.")
            continue

        try:
            X, y = prepare_data_for_training(city, lag_features=LAG_FEATURES)
            print(f"Loaded and preprocessed data for {city}. Shape X: {X.shape}, y: {y.shape}")

            if X.empty or y.empty:
                print(f"⚠️ No data available for training {city} after preprocessing. Skipping.")
                continue

            for model_name, model in models_config.items():
                print(f"Training {model_name} for {city}...")
                try:
                    model.fit(X, y)

                    model_filename = os.path.join(MODELS_DIR, f"{city}_{model_name}.pkl")
                    joblib.dump(model, model_filename)
                    print(f"✅ Saved {model_name} model for {city} to {model_filename}")

                except Exception as train_error:
                    print(f"❌ Error training {model_name} for {city}: {train_error}")

        except FileNotFoundError as e:
            print(f"❌ {e}")
        except ValueError as e:
             print(f"❌ Error processing data for {city}: {e}")
        except Exception as e:
            print(f"❌ An unexpected error occurred for city {city}: {e}")

if __name__ == "__main__":
    print("🚀 Starting model training process...")
    train_all_models()
    print("\n🎯 Model training finished.")
