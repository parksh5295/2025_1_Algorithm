import pandas as pd
# Remove ThreadPoolExecutor imports
# from concurrent.futures import ThreadPoolExecutor
# import os 
import time # Import time module

from data_use.load_weather import get_weather_data_batch
from data_use.load_elevation import get_elevation
from data_use.load_NDVI import get_ndvi
from collections import defaultdict
import math

# Revert _fetch_all_features_for_row as it's no longer called by executor
# We can integrate its logic back into the main loop or keep it as a helper

# --- Sequential version of add_environmental_features --- START ---
def add_environmental_features(df):
    """
    Fetches and appends environment data. Weather data is fetched in batches per day,
    then specific hourly data is mapped. Elevation and NDVI are fetched sequentially.
    """
    if df.empty:
        print("[INFO] Input DataFrame is empty. Skipping environmental feature enrichment.")
        # Ensure all expected columns exist even for an empty DataFrame
        for col in ['temperature', 'precipitation', 'humidity', 'windspeed', 'winddirection', 'elevation', 'ndvi']:
            df[col] = pd.Series(dtype='float64')
        return df

    print(f"\nStarting environmental feature enrichment for {len(df)} rows...")

    # Initialize columns
    for col in ['temperature', 'precipitation', 'humidity', 'windspeed', 'winddirection', 'elevation', 'ndvi']:
        df[col] = pd.NA

    # 1. Weather Data - Batch processing part
    requests_by_date_str = defaultdict(lambda: {'latitudes': [], 'longitudes': [], 'unique_points': set()})
    
    print("Preprocessing for batch weather data...")
    for index, row in df.iterrows():
        lat = row.get('latitude')
        lon = row.get('longitude')
        date_val = row.get('date')

        if pd.isna(lat) or pd.isna(lon) or pd.isna(date_val):
            continue
        try:
            current_ts = pd.to_datetime(date_val)
            date_str = current_ts.strftime('%Y-%m-%d')
            point_key = (float(lat), float(lon))

            if point_key not in requests_by_date_str[date_str]['unique_points']:
                 requests_by_date_str[date_str]['latitudes'].append(float(lat))
                 requests_by_date_str[date_str]['longitudes'].append(float(lon))
                 requests_by_date_str[date_str]['unique_points'].add(point_key)
        except Exception as e:
            print(f"[WARN] Row {index} error during weather preproc: {e}. Skipping this row for weather.")
            continue
            
    all_fetched_weather_data = {}
    print(f"Found {len(requests_by_date_str)} unique dates for weather API calls.")

    for date_str, data in requests_by_date_str.items():
        if not data['latitudes']:
            continue
        print(f"Fetching batch weather data for date: {date_str}, locations: {len(data['latitudes'])}")
        batch_results = get_weather_data_batch(data['latitudes'], data['longitudes'], date_str)
        time.sleep(1) # Add 1-second delay after each batch API call for a date
        
        for res_item in batch_results:
            if res_item and not res_item.get('error') and res_item.get('full_hourly_data'):
                key_lat = round(float(res_item['latitude']), 5)
                key_lon = round(float(res_item['longitude']), 5)
                all_fetched_weather_data[(date_str, key_lat, key_lon)] = res_item['full_hourly_data']
            elif res_item and res_item.get('error'):
                key_lat = round(float(res_item['latitude']), 5)
                key_lon = round(float(res_item['longitude']), 5)
                print(f"[WARN] Error fetching weather for {date_str}, ({key_lat},{key_lon}): {res_item['error']}")
                all_fetched_weather_data[(date_str, key_lat, key_lon)] = {'error': res_item['error']}

    print("Mapping fetched weather data to DataFrame...")
    processed_rows_weather = 0
    for index, row in df.iterrows():
        lat = row.get('latitude')
        lon = row.get('longitude')
        date_val = row.get('date')

        if pd.isna(lat) or pd.isna(lon) or pd.isna(date_val):
            continue
        
        try:
            current_ts = pd.to_datetime(date_val)
            date_str = current_ts.strftime('%Y-%m-%d')
            hour_of_day = current_ts.hour
            
            lookup_lat = round(float(lat), 5)
            lookup_lon = round(float(lon), 5)

            daily_weather_data = all_fetched_weather_data.get((date_str, lookup_lat, lookup_lon))

            if daily_weather_data and not daily_weather_data.get('error'):
                def get_val_at_hour(var_name, hour):
                    var_data = daily_weather_data.get(var_name)
                    if var_data and isinstance(var_data, list) and hour < len(var_data):
                        val = var_data[hour]
                        return val if val is not None and not (isinstance(val, float) and math.isnan(val)) else None
                    return None

                df.loc[index, 'temperature'] = get_val_at_hour('temperature_2m', hour_of_day)
                df.loc[index, 'precipitation'] = get_val_at_hour('precipitation', hour_of_day)
                df.loc[index, 'humidity'] = get_val_at_hour('relative_humidity_2m', hour_of_day)
                df.loc[index, 'windspeed'] = get_val_at_hour('wind_speed_10m', hour_of_day)
                df.loc[index, 'winddirection'] = get_val_at_hour('wind_direction_10m', hour_of_day)
            elif daily_weather_data and daily_weather_data.get('error'):
                pass
            else:
                print(f"[WARN] No pre-fetched weather data found for row {index}: {date_str}, ({lookup_lat},{lookup_lon}). Values will be NA.")
            
            processed_rows_weather +=1
            if processed_rows_weather % 200 == 0:
                 print(f"   Mapped weather for {processed_rows_weather} rows...")

        except Exception as e:
            print(f"[WARN] Row {index} error during weather mapping: {e}. Weather values will be NA.")
            continue

    print("Fetching and mapping elevation and NDVI data...")
    processed_rows_other = 0
    for index, row in df.iterrows():
        latitude = row.get('latitude')
        longitude = row.get('longitude')
        date_val = row.get('date')

        if pd.isna(latitude) or pd.isna(longitude):
            continue
        
        df.loc[index, 'elevation'] = get_elevation(latitude, longitude)

        if pd.isna(date_val):
            df.loc[index, 'ndvi'] = None
        else:
            try:
                current_ts_ndvi = pd.to_datetime(date_val)
                df.loc[index, 'ndvi'] = get_ndvi(latitude, longitude, current_ts_ndvi)
            except ValueError:
                 print(f"[WARN] Row {index} invalid date for NDVI: {date_val}. NDVI will be NA.")
                 df.loc[index, 'ndvi'] = None 
            except Exception as e:
                 print(f"[WARN] Row {index} error during NDVI processing: {e}. NDVI will be NA.")
                 df.loc[index, 'ndvi'] = None

        processed_rows_other += 1
        if processed_rows_other % 200 == 0:
            print(f"   Processed elevation/NDVI for {processed_rows_other} rows...")

    cols_to_float = ['temperature', 'precipitation', 'humidity', 'windspeed', 'winddirection', 'elevation', 'ndvi']
    for col in cols_to_float:
        if col in df.columns: # Check if column exists in df
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print("Finished environmental feature enrichment.")
    return df

# --- Sequential version of add_environmental_features --- END ---

# If there are other functions in this file, they should be preserved.
# The edit tool will attempt to merge this change.
# Ensure no other functions are accidentally removed or altered if they exist.
