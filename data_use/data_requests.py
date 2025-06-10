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

    MAX_LOCATIONS_PER_BATCH = 30 # Limit batch size for API calls, reduced from 50
    API_CALL_DELAY_SECONDS = 2  # Increased from 1 to 2
    DATE_CHANGE_DELAY_SECONDS = 10 # New: Delay when the date changes
    DELAYED_RETRY_WAIT_SECONDS = 180 # 3 minutes
    MAX_DELAYED_RETRIES = 1 # Max number of delayed retries for a single task

    # Create a flat list of tasks for the queue
    task_queue_initial = []
    for date_str_loop, data_loop in requests_by_date_str.items():
        if not data_loop['latitudes']:
            continue
        
        all_lats_for_date_loop = data_loop['latitudes']
        all_lons_for_date_loop = data_loop['longitudes']
        num_total_locations_for_date_loop = len(all_lats_for_date_loop)

        for i_loop in range(0, num_total_locations_for_date_loop, MAX_LOCATIONS_PER_BATCH):
            batch_lats_loop = all_lats_for_date_loop[i_loop:i_loop + MAX_LOCATIONS_PER_BATCH]
            batch_lons_loop = all_lons_for_date_loop[i_loop:i_loop + MAX_LOCATIONS_PER_BATCH]
            if batch_lats_loop: 
                task_queue_initial.append({
                    'date_str': date_str_loop, 
                    'batch_lats': batch_lats_loop, 
                    'batch_lons': batch_lons_loop, 
                    'original_total': num_total_locations_for_date_loop, 
                    'current_batch_start_idx': i_loop,
                    'retry_count': 0 # Initialize retry count
                })

    print(f"Generated {len(task_queue_initial)} initial API tasks.")

    # Main task processing loop with delayed retry queue
    task_queue_live = list(task_queue_initial) # Make a mutable copy
    delayed_retry_queue = [] # Stores (retry_at_timestamp, task_details)
    
    last_processed_date_str_live = None
    processed_task_count = 0

    while task_queue_live or delayed_retry_queue:
        current_time_live = time.time()

        # Check delayed_retry_queue and move tasks back to live queue if ready
        newly_added_to_live_queue = 0
        remaining_in_delayed_live = []
        for retry_at_ts, delayed_task_details_item in delayed_retry_queue:
            if current_time_live >= retry_at_ts:
                task_queue_live.append(delayed_task_details_item)
                newly_added_to_live_queue += 1
            else:
                remaining_in_delayed_live.append((retry_at_ts, delayed_task_details_item))
        delayed_retry_queue = remaining_in_delayed_live
        if newly_added_to_live_queue > 0:
            print(f"[INFO] Moved {newly_added_to_live_queue} tasks from delayed_retry_queue to live_task_queue.")

        if not task_queue_live:
            if delayed_retry_queue:
                # print(f"[INFO] Live task queue is empty. Waiting for tasks from delayed_retry_queue (next check in ~{API_CALL_DELAY_SECONDS}s)...")
                time.sleep(API_CALL_DELAY_SECONDS) 
            continue

        # Get task from the front of the live queue
        current_task_details = task_queue_live.pop(0)
        processed_task_count += 1

        date_str_task = current_task_details['date_str']
        batch_lats_task = current_task_details['batch_lats']
        batch_lons_task = current_task_details['batch_lons']
        num_total_locations_task = current_task_details['original_total']
        current_batch_start_idx_task = current_task_details['current_batch_start_idx']
        task_retry_count = current_task_details['retry_count']
        
        # Apply date change delay only for tasks that are not from delayed retry (retry_count == 0)
        # and if the date actually changed.
        if task_retry_count == 0 and last_processed_date_str_live is not None and date_str_task != last_processed_date_str_live:
            print(f"Date changed from {last_processed_date_str_live} to {date_str_task}. Applying extra delay of {DATE_CHANGE_DELAY_SECONDS}s...")
            time.sleep(DATE_CHANGE_DELAY_SECONDS)

        print(f"Processing task {processed_task_count} (UID: {date_str_task}_{current_batch_start_idx_task}, Retry: {task_retry_count}): Date {date_str_task}, locations {current_batch_start_idx_task + 1} to {min(current_batch_start_idx_task + MAX_LOCATIONS_PER_BATCH, num_total_locations_task)} of {num_total_locations_task}")
        
        current_batch_results = get_weather_data_batch(batch_lats_task, batch_lons_task, date_str_task)
        
        task_had_critical_failure = False
        for res_item in current_batch_results:
            # Process successful results or non-critical errors into all_fetched_weather_data
            if res_item and not res_item.get('error') and res_item.get('full_hourly_data'):
                key_lat = round(float(res_item['latitude']), 5)
                key_lon = round(float(res_item['longitude']), 5)
                all_fetched_weather_data[(date_str_task, key_lat, key_lon)] = res_item['full_hourly_data']
            elif res_item and res_item.get('error'):
                key_lat = round(float(res_item['latitude']), 5)
                key_lon = round(float(res_item['longitude']), 5)
                error_message = res_item['error']
                print(f"[WARN] Error for {date_str_task}, ({key_lat},{key_lon}): {error_message}") # Log individual errors
                all_fetched_weather_data[(date_str_task, key_lat, key_lon)] = {'error': error_message}
                if "Failed to fetch weather data after 5 retries" in error_message:
                    task_had_critical_failure = True # Mark task for potential delayed retry

        if task_had_critical_failure:
            if task_retry_count < MAX_DELAYED_RETRIES:
                current_task_details['retry_count'] += 1
                retry_at_timestamp = time.time() + DELAYED_RETRY_WAIT_SECONDS
                delayed_retry_queue.append((retry_at_timestamp, current_task_details))
                print(f"[INFO] Task (UID: {date_str_task}_{current_batch_start_idx_task}) critically failed. Moved to delayed_retry_queue (Retry {current_task_details['retry_count']}/{MAX_DELAYED_RETRIES}). Will retry around {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(retry_at_timestamp))}.")
            else:
                print(f"[ERROR] Task (UID: {date_str_task}_{current_batch_start_idx_task}) critically failed after {task_retry_count} delayed retries. Giving up.")
        else:
             print(f"[INFO] Task (UID: {date_str_task}_{current_batch_start_idx_task}) processed.")

        # Update last processed date only if the task wasn't a critical failure moved to retry
        # (to ensure date change delay applies correctly if a retried task starts a new date sequence)
        if not task_had_critical_failure or task_retry_count >= MAX_DELAYED_RETRIES :
            last_processed_date_str_live = date_str_task
        
        print(f"Finished processing current task logic. Waiting for {API_CALL_DELAY_SECONDS}s...")
        time.sleep(API_CALL_DELAY_SECONDS)

    print(f"All API tasks (initial and retries) have been processed.")
    # The old loop structure is gone.

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
                pass # Error was already logged, data will remain NA
            else:
                # This case might occur if a sub-batch failed silently or data for specific point wasn't in any successful sub-batch
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
            df.loc[index, 'ndvi'] = None # pd.NA would be more consistent
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
