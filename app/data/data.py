from meteostat import Point, Hourly
from datetime import datetime
import pandas as pd
import os

# Define cities and their corresponding timezones
cities = {
    "ahmedabad": {"coords": Point(23.0225, 72.5714, 53), "timezone": "Asia/Kolkata"},
    "mumbai": {"coords": Point(19.0760, 72.8777, 14), "timezone": "Asia/Kolkata"},
    "delhi": {"coords": Point(28.7041, 77.1025, 216), "timezone": "Asia/Kolkata"},
    "bengaluru": {"coords": Point(12.9716, 77.5946, 920), "timezone": "Asia/Kolkata"}
}

# Start and end date for the data
start = datetime(2020, 1, 1)
end = datetime.now()

# Updated Rename mapping for columns (removed "prcp" for Precipitation)
rename_map = {
    "time": "Timestamp",
    "temp": "Temperature (°C)",
    "rhum": "Humidity (%)",
    "wspd": "Wind Speed (km/h)",
    "wdir": "Wind Direction (°)",
}

# Get the current directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Download and process data for each city
for city_name, city_info in cities.items():
    print(f"⚡ Downloading weather data for {city_name.title()}...")

    # Fetch weather data
    data = Hourly(city_info['coords'], start, end, timezone=city_info['timezone']).fetch()

    if data.empty:
        print(f"💀 No data for {city_name.title()} — moving on.")
        continue

    # Reset index
    data.reset_index(inplace=True)

    # Fill missing values using forward fill, then backward fill
    # Forward fill (fill from previous value) and backward fill (fill from next value)
    data.fillna(method='ffill', inplace=True)  # Fill using the previous value
    data.fillna(method='bfill', inplace=True)  # Fill using the next value if still NaN

    # Rename columns according to the updated mapping (without "prcp")
    data.rename(columns={col: rename_map[col] for col in data.columns if col in rename_map}, inplace=True)

    # Filter the data to only include columns present in the rename_map (after renaming)
    data = data[list(rename_map.values())]

    # Format timestamp
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['Timestamp'] = data['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Save to CSV in the same folder as the script
    filename = os.path.join(script_dir, f"{city_name}.csv")
    data.to_csv(filename, index=False)

    print(f"✅ {city_name.title()} data saved: {os.path.abspath(filename)}")

print("🎯 All done. Your NaNs have been conquered. Praise be to the Data Warlord.")
