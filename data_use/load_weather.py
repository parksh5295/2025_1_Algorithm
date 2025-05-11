# load weather data from open-meteo
# temperature, precipitation, humidity, wind speed, wind direction
import requests
import pandas as pd
import math
import time # Add for time.sleep()
import sys

def get_weather_data_batch(latitudes, longitudes, target_date_str_yyyy_mm_dd):
    """
    Fetches all hourly weather data for multiple locations for a single target date 
    using Open-Meteo API with exponential backoff for retries.
    
    Args:
        latitudes (list): List of latitudes.
        longitudes (list): List of longitudes.
        target_date_str_yyyy_mm_dd (str): The single date string (YYYY-MM-DD) 
                                            for which to fetch data.
    
    Returns:
        list: A list of dictionaries, where each dictionary contains weather data 
              for the corresponding lat/lon pair. The order matches the input order.
              Each dictionary will have keys: 'latitude', 'longitude', 
              'date' (Timestamp object for the target_date_str_yyyy_mm_dd at 00:00),
              'full_hourly_data' (dict containing lists of 24 hourly values for each variable),
              'error' (string, optional, if an error occurred for this point or batch).
    """
    if not latitudes or not longitudes or len(latitudes) != len(longitudes):
        print("[WARN] Latitudes and longitudes lists are empty or mismatched. Skipping weather fetch.")
        return [{'latitude': lat, 'longitude': lon, 'date': None, 'full_hourly_data': None, 'error': 'Input list error'} 
                for lat, lon in zip(latitudes, longitudes)] if latitudes else []

    try:
        base_target_date_ts = pd.to_datetime(target_date_str_yyyy_mm_dd).normalize()
    except ValueError:
        print(f"[WARN] Invalid date format string for weather data: {target_date_str_yyyy_mm_dd}. Skipping.")
        return [{'latitude': lat, 'longitude': lon, 'date': None, 'full_hourly_data': None, 'error': 'Invalid date format'}
                for lat, lon in zip(latitudes, longitudes)]

    lat_str = ",".join(map(str, [round(lat, 5) for lat in latitudes]))
    lon_str = ",".join(map(str, [round(lon, 5) for lon in longitudes]))
    
    timezone = "UTC"
    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat_str,
        "longitude": lon_str,
        "hourly": "temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
        "start_date": target_date_str_yyyy_mm_dd,
        "end_date": target_date_str_yyyy_mm_dd,
        "timezone": timezone
    }

    max_retries = 5
    base_delay_seconds = 1
    results_list = [None] * len(latitudes)

    for attempt in range(max_retries):
        try:
            print(f"[DEBUG] Requesting full day weather data (Attempt {attempt + 1}/{max_retries}). Locations: {len(latitudes)}, Date: {target_date_str_yyyy_mm_dd}")
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            api_response_data = response.json()

            if isinstance(api_response_data, dict) and api_response_data.get("error"):
                error_reason = api_response_data.get('reason', 'API Error')
                print(f"[ERROR] API error for batch: {error_reason}")
                for i in range(len(latitudes)):
                    results_list[i] = {
                        'latitude': latitudes[i], 'longitude': longitudes[i], 
                        'date': base_target_date_ts, 'full_hourly_data': None, 'error': error_reason
                    }
                return results_list

            # Expecting a list of results, one for each location if API call for multiple locations is successful
            if not isinstance(api_response_data, list):
                raise ValueError(f"Unexpected API response format. Expected list, got {type(api_response_data)}")

            if len(api_response_data) != len(latitudes):
                raise ValueError(f"API response list length ({len(api_response_data)}) mismatch with request ({len(latitudes)})")

            for i, single_loc_data in enumerate(api_response_data):
                current_lat = latitudes[i]
                current_lon = longitudes[i]
                
                hourly_data_content = single_loc_data.get('hourly')
                if hourly_data_content and isinstance(hourly_data_content.get('time'), list):
                    results_list[i] = {
                        'latitude': current_lat, 
                        'longitude': current_lon,
                        'date': base_target_date_ts,
                        'full_hourly_data': hourly_data_content,
                        'error': None
                    }
                else:
                    print(f"[WARN] No 'hourly' data or time array in API response for lat={current_lat}, lon={current_lon} on {target_date_str_yyyy_mm_dd}. API item: {single_loc_data}")
                    results_list[i] = {
                        'latitude': current_lat, 'longitude': current_lon,
                        'date': base_target_date_ts, 'full_hourly_data': None, 
                        'error': "Missing hourly data in API response"
                    }
            return results_list

        except requests.exceptions.HTTPError as e:
            error_message = f"HTTPError: {e}"
            status_code = e.response.status_code if e.response is not None else None
            if status_code == 429:
                wait_time = base_delay_seconds * (2 ** attempt)
                print(f"[WARN] HTTP 429 Too Many Requests. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            elif status_code == 400:
                error_message = f"Bad Request (400) for weather API. URL: {e.request.url if e.request else params}. Response: {e.response.text if e.response else 'N/A'}. No retry."
                print(f"[ERROR] {error_message}")
                for i in range(len(latitudes)):
                    results_list[i] = {'latitude': latitudes[i], 'longitude': longitudes[i], 'date': base_target_date_ts, 'full_hourly_data': None, 'error': 'Bad Request (400)'}
                return results_list
            else:
                print(f"[ERROR] {error_message} (Attempt {attempt+1}/{max_retries})")
                wait_time = base_delay_seconds * (2 ** attempt)
                time.sleep(wait_time)
        except requests.exceptions.Timeout:
            error_message = "Timeout fetching weather data"
            print(f"[ERROR] {error_message} (Attempt {attempt+1}/{max_retries}). Retrying...")
            wait_time = base_delay_seconds * (2 ** attempt)
            time.sleep(wait_time)
        except requests.exceptions.RequestException as e:
            error_message = f"RequestException: {e}"
            print(f"[ERROR] {error_message} (Attempt {attempt+1}/{max_retries}). Retrying...")
            wait_time = base_delay_seconds * (2 ** attempt)
            time.sleep(wait_time)
        except Exception as e: 
            error_message = f"Unexpected error parsing weather data: {e}"
            print(f"[ERROR] {error_message} (Attempt {attempt+1}/{max_retries})")
            wait_time = base_delay_seconds * (2 ** attempt)
            time.sleep(wait_time)

    final_error_msg = f"Failed to fetch weather data after {max_retries} retries for date {target_date_str_yyyy_mm_dd}"
    print(f"[ERROR] {final_error_msg}")
    for i in range(len(latitudes)):
        if results_list[i] is None:
            results_list[i] = {
                'latitude': latitudes[i], 'longitude': longitudes[i], 
                'date': base_target_date_ts, 'full_hourly_data': None, 'error': final_error_msg
            }
    return results_list

def get_weather_data(latitude, longitude, date_with_hour):
    """
    Wrapper for fetching weather data for a single location and specific hour.
    This function is now less efficient due to batch changes.
    It calls the batch function for a single point for a full day, then extracts one hour.
    Consider modifying calling code to use get_weather_data_batch directly if possible.
    """
    if not isinstance(date_with_hour, pd.Timestamp):
        try:
            date_ts = pd.to_datetime(date_with_hour)
        except ValueError:
            print(f"[WARN] Invalid date format for single weather data request: {date_with_hour}. Skipping.")
            return None, None, None, None, None
    else:
        date_ts = date_with_hour

    target_date_str = date_ts.strftime("%Y-%m-%d")
    hour_of_day = date_ts.hour

    try:
        lat_f = float(latitude)
        lon_f = float(longitude)
        if not (-90 <= lat_f <= 90 and -180 <= lon_f <= 180):
            print(f"[WARN] Invalid latitude/longitude: ({latitude}, {longitude}). Skipping.")
            return None, None, None, None, None
    except (ValueError, TypeError):
        print(f"[WARN] Non-numeric latitude/longitude: ({latitude}, {longitude}). Skipping.")
        return None, None, None, None, None

    batch_results = get_weather_data_batch([lat_f], [lon_f], target_date_str)
    
    if not batch_results or len(batch_results) != 1:
        print(f"[WARN] Single weather fetch (via batch) returned unexpected result for {lat_f}, {lon_f}, {target_date_str}")
        return None, None, None, None, None
        
    single_loc_result = batch_results[0]

    if single_loc_result.get('error') or not single_loc_result.get('full_hourly_data'):
        error_msg = single_loc_result.get('error', 'Missing full_hourly_data')
        print(f"[WARN] Single weather fetch failed for {lat_f}, {lon_f}, {date_ts}: {error_msg}")
        return None, None, None, None, None

    full_hourly = single_loc_result['full_hourly_data']
    
    def get_specific_hourly_value(variable_name, hour_idx):
        values = full_hourly.get(variable_name)
        if values and isinstance(values, list) and hour_idx < len(values):
            value = values[hour_idx]
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                return value
        print(f"[DEBUG] Variable {variable_name} not found or invalid for hour {hour_idx} in full_hourly_data for {lat_f},{lon_f},{target_date_str}")
        return None

    temp = get_specific_hourly_value('temperature_2m', hour_of_day)
    precip = get_specific_hourly_value('precipitation', hour_of_day)
    hum = get_specific_hourly_value('relative_humidity_2m', hour_of_day)
    ws = get_specific_hourly_value('wind_speed_10m', hour_of_day)
    wd = get_specific_hourly_value('wind_direction_10m', hour_of_day)
    
    return temp, precip, hum, ws, wd
