from meteostat import Point, Hourly
from datetime import datetime
import pandas as pd
import os

cities = {
    "ahmedabad": {"coords": Point(23.0225, 72.5714, 53), "timezone": "Asia/Kolkata"},
    "mumbai": {"coords": Point(19.0760, 72.8777, 14), "timezone": "Asia/Kolkata"},
    "delhi": {"coords": Point(28.7041, 77.1025, 216), "timezone": "Asia/Kolkata"},
    "bengaluru": {"coords": Point(12.9716, 77.5946, 920), "timezone": "Asia/Kolkata"}
}

start = datetime(2020, 1, 1)
end = datetime.now()

rename_map = {
    "time": "Timestamp",
    "temp": "Temperature (°C)",
    "rhum": "Humidity (%)",
    "wspd": "Wind Speed (km/h)",
    "wdir": "Wind Direction (°)"
}

script_dir = os.path.dirname(os.path.abspath(__file__))

def process_weather_data(city_name, city_info):
    data = Hourly(city_info['coords'], start, end, timezone=city_info['timezone']).fetch()

    if data.empty:
        return

    data.reset_index(inplace=True)
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)
    data.rename(columns={col: rename_map.get(col, col) for col in data.columns}, inplace=True)
    data = data[list(rename_map.values())]
    data['Timestamp'] = pd.to_datetime(data['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

    filename = os.path.join(script_dir, f"{city_name}.csv")
    data.to_csv(filename, index=False)

for city_name, city_info in cities.items():
    process_weather_data(city_name, city_info)

print("Data processing completed successfully.")
