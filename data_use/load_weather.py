# load weather data from open-meteo
# temperature, precipitation, humidity, wind speed, wind direction
import requests
import pandas as pd
import math
import sys


def get_weather_data(latitude, longitude, date):
    # Removed import from function start

    # Input Validation --- START ---
    try:
        lat = float(latitude)
        lon = float(longitude)
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            print(f"[WARN] Invalid latitude/longitude values: ({latitude}, {longitude}). Skipping weather fetch.")
            return None, None, None, None, None
    except (ValueError, TypeError):
        print(f"[WARN] Non-numeric latitude/longitude: ({latitude}, {longitude}). Skipping weather fetch.")
        return None, None, None, None, None
    # Input Validation --- END ---

    # Ensure date is a pandas Timestamp object for hour access
    if not isinstance(date, pd.Timestamp):
        try:
            date = pd.to_datetime(date)
        except ValueError:
            print(f"[WARN] Invalid date format for weather data: {date}. Skipping.")
            return None, None, None, None, None

    api_date = date.strftime("%Y-%m-%d")
    hour_index = date.hour # Get the hour (0-23) for indexing

    # Construct API URL with rounded coordinates, UTC timezone, and corrected precipitation variable
    lat_rounded = round(lat, 5)
    lon_rounded = round(lon, 5)
    # Set timezone to UTC
    timezone = "UTC"
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat_rounded}&longitude={lon_rounded}&hourly=temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m,wind_direction_10m&start_date={api_date}&end_date={api_date}&timezone={timezone}"
    
    temperature, precipitation, humidity, wind_speed, wind_direction = None, None, None, None, None

    try:
        '''
        # --- Diagnostic codes START ---
        print("\n--- Debugging NameError ---")
        print(f"Checking 'requests' in globals(): {globals().get('requests')}")
        print(f"Checking 'requests' in locals(): {locals().get('requests')}")
        print(f"Checking 'requests' in sys.modules: {sys.modules.get('requests')}")
        if 'requests' in sys.modules:
             print(f"   requests module path: {sys.modules['requests'].__file__}")
        print("--- End Debugging NameError ---\n")
        # --- Diagnostic codes END ---
        '''

        # Import requests explicitly right before use inside the try block
        # import requests as local_requests 

        print(f"[DEBUG] Requesting URL: {url}")
        # Use the top-level imported requests module
        response = requests.get(url, timeout=10) 
        response.raise_for_status()
        data = response.json()

        # Safely access hourly data
        hourly_data = data.get('hourly')
        if hourly_data:
            # Helper to safely get value at the specific hour index
            def get_hourly_value(variable_name, index):
                values = hourly_data.get(variable_name)
                # Check if list exists, has enough elements, and value is not null/NaN
                if values and isinstance(values, list) and index < len(values):
                    value = values[index]
                    # Open-Meteo might return None or NaN for missing data
                    if value is not None and not (isinstance(value, float) and math.isnan(value)):
                        return value
                return None # Return None if data is missing or invalid

            # Use the helper function to get data for the specific hour
            temperature = get_hourly_value('temperature_2m', hour_index)
            precipitation = get_hourly_value('precipitation', hour_index)
            # API parameter changed from humidity_2m to relative_humidity_2m based on common usage
            humidity = get_hourly_value('relative_humidity_2m', hour_index) 
            wind_speed = get_hourly_value('wind_speed_10m', hour_index)
            wind_direction = get_hourly_value('wind_direction_10m', hour_index)
        else:
            print(f"[WARN] No 'hourly' data found in API response for {api_date}, lat={lat_rounded}, lon={lon_rounded}")

    except requests.exceptions.Timeout:
         print(f"[ERROR] Timeout while fetching weather data for {api_date}, lat={lat_rounded}, lon={lon_rounded}")
    except requests.exceptions.RequestException as e:
        # Initialize response to None to avoid UnboundLocalError in except block if request fails early
        response_status = getattr(e.response, 'status_code', None)
        if response_status == 400:
             print(f"[ERROR] Bad Request (400) for weather API. URL: {url}. Check parameters.")
        else:
             print(f"[ERROR] Failed to fetch weather data for {api_date}, lat={lat_rounded}, lon={lon_rounded}: {e}")
    except Exception as e:
        print(f"[ERROR] Failed to parse weather data or unexpected error for {api_date}, lat={lat_rounded}, lon={lon_rounded}: {e}")

    return temperature, precipitation, humidity, wind_speed, wind_direction
