# Load elevation data from Open Topo Data API (SRTM 90m)
# Replaces the previous Open-Elevation API implementation

import requests

# Your Open-Meteo API Key (ensure this is the same key as used in load_weather.py)
OPEN_METEO_API_KEY = 'y0ndQo7qSxqJrArn' 

def get_elevation(latitude, longitude):
    # Construct the API URL for Open-Meteo Elevation API (Commercial Endpoint)
    # google_api_url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={latitude},{longitude}&key={GOOGLE_API_KEY}" # Original Google API

    # url = f"https://api.opentopodata.org/v1/srtm90m?locations={latitude},{longitude}"
    # elevation = None # Initialize elevation to None

    # Open-Meteo Elevation API
    open_meteo_elevation_url = f"https://customer-api.open-meteo.com/v1/elevation"
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "apikey": OPEN_METEO_API_KEY
    }
    elevation = None

    try:
        # response = requests.get(url, timeout=10) # Add a timeout
        # response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        # response = requests.get(google_api_url, timeout=10) # Original Google request
        response = requests.get(open_meteo_elevation_url, params=params, timeout=10)
        response.raise_for_status() 

        data = response.json()

        '''
        # Check if the response status is OK and results are present
        if data.get('status') == 'OK' and data.get('results'):
            # Extract elevation (meters)
            # The structure is the same as Open-Elevation's successful response
            result = data['results'][0]
            elevation = result.get('elevation')
            if elevation is None:
                 print(f"[WARN] Elevation data missing in successful Open Topo Data response for {latitude},{longitude}. Response: {data}")
        '''

        # Check if the response status is OK and results are present (Open-Meteo Elevation specific)
        # The actual successful response from Open-Meteo elevation is simpler: e.g. {"elevation":[34.0]}
        # It does not have a 'status' key in the same way Google's or the previous OpenTopoData API did.
        # A 200 OK with valid JSON containing an 'elevation' list is considered a success.
        if isinstance(data.get('elevation'), list) and len(data.get('elevation')) > 0:
            elevation_value = data['elevation'][0]
            if elevation_value is not None: # Open-Meteo might return null in the list for errors/no data for a point
                elevation = float(elevation_value)
            else:
                print(f"[WARN] Elevation value is null in Open-Meteo response for {latitude},{longitude}. Data: {data}")
        # Handle Open-Meteo specific error structure if known, or rely on raise_for_status for HTTP errors.
        # For example, if Open-Meteo returns a JSON with an error key on 200 OK for some reason:
        elif data.get('error') and data.get('reason'):
             print(f"[ERROR] Open-Meteo Elevation API returned an error for {latitude},{longitude}. Reason: {data.get('reason')}")
        else:
            print(f"[WARN] Open-Meteo Elevation API request for {latitude},{longitude} did not return expected 'elevation' list. Response: {data}")

    except requests.exceptions.HTTPError as e:
        # Handle specific HTTP errors, e.g., 401/403 for API key issues if not caught by a JSON error response
        if e.response.status_code == 401 or e.response.status_code == 403:
            print(f"[ERROR] Open-Meteo Elevation API request unauthorized/forbidden (API Key issue?) for {latitude},{longitude}. Status: {e.response.status_code}. Response: {e.response.text}")
        else:
            print(f"[ERROR] HTTP error fetching elevation from Open-Meteo Elevation API for {latitude},{longitude}: {e}")
    except requests.exceptions.Timeout:
        print(f"[ERROR] Timeout while fetching elevation data from Open-Meteo Elevation API for {latitude},{longitude}")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to fetch elevation data from Open-Meteo Elevation API for {latitude},{longitude}: {e}")
    except ValueError as e: # Catches JSON decoding errors
        print(f"[ERROR] Failed to parse JSON response from Open-Meteo Elevation API for {latitude},{longitude}: {e}")
        print(f"Raw response content: {response.text if 'response' in locals() else 'N/A'}")
    except Exception as e:
        print(f"[ERROR] Unexpected error fetching elevation from Open-Meteo Elevation API for {latitude},{longitude}: {e}")

    return elevation
