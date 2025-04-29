from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os
from typing import Optional

from app.ml.predict import make_predictions
from app.ml.ensemble import predict_ensemble, BASE_MODEL_NAMES
from app.utils.preprocess import TARGET_FEATURES, TIMESTAMP_COL

app = FastAPI(title="Weather Forecast API")

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1",
    "http://127.0.0.1:8080",
    "null",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def filter_by_day(df: pd.DataFrame, day_of_week: int) -> pd.DataFrame:
    if TIMESTAMP_COL not in df.columns:
         df[TIMESTAMP_COL] = pd.to_datetime(df.index)

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    return df[df[TIMESTAMP_COL].dt.dayofweek == day_of_week].copy()

@app.get("/predict")
async def get_prediction(
    city: str = Query(..., description="City name (e.g., ahmedabad)"),
    model_name: str = Query(..., description=f"Model name (e.g., {' / '.join(BASE_MODEL_NAMES)} / Ensemble)"),
    forecast_type: str = Query(..., description="Forecast duration ('48h' or '1week')"),
    day_of_week: Optional[int] = Query(None, description="Day of week (0=Mon, 6=Sun) - only if forecast_type is '1week'")
):
    allowed_cities = ["ahmedabad", "mumbai", "delhi", "bengaluru"]
    allowed_models = BASE_MODEL_NAMES + ["Ensemble"]
    allowed_forecast_types = ["48h", "1week"]

    if city not in allowed_cities:
        raise HTTPException(status_code=400, detail=f"Invalid city. Allowed: {', '.join(allowed_cities)}")
    if model_name not in allowed_models:
        raise HTTPException(status_code=400, detail=f"Invalid model name. Allowed: {', '.join(allowed_models)}")
    if forecast_type not in allowed_forecast_types:
        raise HTTPException(status_code=400, detail=f"Invalid forecast type. Allowed: {', '.join(allowed_forecast_types)}")
    if forecast_type == "48h" and day_of_week is not None:
         raise HTTPException(status_code=400, detail="Day of week selection is only valid for '1week' forecast type.")
    if day_of_week is not None and not (0 <= day_of_week <= 6):
         raise HTTPException(status_code=400, detail="Invalid day_of_week. Must be between 0 (Monday) and 6 (Sunday).")


    hours_to_predict = 48 if forecast_type == "48h" else 168

    try:
        print(f"Received request: city={city}, model={model_name}, type={forecast_type}, day={day_of_week}")
        if model_name == "Ensemble":
            df_predictions = predict_ensemble(city, hours_to_predict)
        else:
            df_predictions = make_predictions(city, model_name, hours_to_predict)

        if df_predictions.empty:
             raise HTTPException(status_code=500, detail="Prediction generation failed or returned empty results.")

        df_predictions[TIMESTAMP_COL] = df_predictions[TIMESTAMP_COL].dt.strftime('%Y-%m-%d %H:%M:%S')

        if forecast_type == "1week" and day_of_week is not None:
            df_predictions = filter_by_day(df_predictions, day_of_week)
            if df_predictions.empty:
                 print(f"Warning: No predictions found for day_of_week={day_of_week} within the forecast period.")
                 return []


        result = df_predictions.round(2).to_dict(orient='records')
        return result

    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
         print(f"Error: {e}")
         raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    except Exception as e:
        print(f"Unexpected Error: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the Weather Forecast API!"}