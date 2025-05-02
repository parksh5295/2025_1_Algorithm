import pandas as pd
# Remove ThreadPoolExecutor imports
# from concurrent.futures import ThreadPoolExecutor
# import os 

from data_use.load_weather import get_weather_data
from data_use.load_elevation import get_elevation
from data_use.load_NDVI import get_ndvi

# Revert _fetch_all_features_for_row as it's no longer called by executor
# We can integrate its logic back into the main loop or keep it as a helper

# --- Sequential version of add_environmental_features --- START ---
def add_environmental_features(df):
    """
    Sequentially fetches and appends environment data for each row in the DataFrame.
    (Parallelism removed version)
    """
    # Initialize a list to store the results
    weather_data_list = []
    elevation_list = []
    ndvi_list = []

    print(f"\nStarting sequential environmental feature enrichment for {len(df)} rows...")

    # Collect data for each row (iterrows can be slow, but we use it for implementation simplicity)
    # Faster way: df.apply(lambda row: ..., axis=1) or vectorized operations (not suitable for API calls)
    for index, row in df.iterrows():
        latitude = row.get('latitude') # Use .get for safety
        longitude = row.get('longitude')
        date = row.get('date') # Assumes 'date' column exists and is datetime

        # Verify required column values
        if latitude is None or longitude is None or date is None:
            print(f"[WARN] Skipping row {index} due to missing lat/lon/date.")
            weather_data_list.append({'temperature': None, 'precipitation': None, 'humidity': None, 'windspeed': None, 'winddirection': None})
            elevation_list.append({'elevation': None})
            ndvi_list.append({'ndvi': None})
            continue
            
        # Check the date format (should be datetime)
        if not isinstance(date, pd.Timestamp):
             try:
                 date = pd.to_datetime(date)
             except ValueError:
                 print(f"[WARN] Skipping row {index} due to invalid date format: {date}")
                 weather_data_list.append({'temperature': None, 'precipitation': None, 'humidity': None, 'windspeed': None, 'winddirection': None})
                 elevation_list.append({'elevation': None})
                 ndvi_list.append({'ndvi': None})
                 continue

        # 1. Get weather data
        temp, precip, hum, wind_speed, wind_dir = get_weather_data(latitude, longitude, date)
        weather_data_list.append({
            'temperature': temp,
            'precipitation': precip,
            'humidity': hum,
            'windspeed': wind_speed,
            'winddirection': wind_dir
        })

        # 2. Elevation data retrieval
        elevation_value = get_elevation(latitude, longitude)
        elevation_list.append({'elevation': elevation_value})

        # 3. Get NDVI data
        ndvi_value = get_ndvi(latitude, longitude, date)
        ndvi_list.append({'ndvi': ndvi_value})

        # Show progress (optional, may print too often)
        if (index + 1) % 100 == 0:
             print(f"   Processed {index + 1} rows...")

    # Convert results to DataFrames
    weather_df = pd.DataFrame(weather_data_list)
    elevation_df = pd.DataFrame(elevation_list)
    ndvi_df = pd.DataFrame(ndvi_list)

    # combine the source DataFrame with the result DataFrame (be careful with index alignment)
    # Combine safely, preserving the indexes of the original DFs
    df.reset_index(drop=True, inplace=True) # 원Initialize the main index (if needed)
    enriched_df = pd.concat([df, weather_df, elevation_df, ndvi_df], axis=1)

    print(f"Finished sequential enrichment.")
    return enriched_df
# --- Sequential version of add_environmental_features --- END ---


# --- Original parallel code (commented out) --- START ---
# def _fetch_all_features_for_row(lat, lon, date):
#     # ... (previous helper function code) ...
#
# def add_environmental_features(df, max_workers=None):
#     # ... (previous parallel implementation using ThreadPoolExecutor) ...
# --- Original parallel code (commented out) --- END ---
