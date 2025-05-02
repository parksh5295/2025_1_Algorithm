# Load elevation data from Open Topo Data API (SRTM 90m)
# Replaces the previous Open-Elevation API implementation

import requests

def get_elevation(latitude, longitude):
    # Construct the API URL for Open Topo Data (SRTM 90m dataset)
    # Documentation: https://www.opentopodata.org/api/
    url = f"https://api.opentopodata.org/v1/srtm90m?locations={latitude},{longitude}"
    elevation = None # Initialize elevation to None

    try:
        response = requests.get(url, timeout=10) # Add a timeout
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        data = response.json()

        # Check if the response status is OK and results are present
        if data.get('status') == 'OK' and data.get('results'):
            # Extract elevation (meters)
            # The structure is the same as Open-Elevation's successful response
            result = data['results'][0]
            elevation = result.get('elevation')
            if elevation is None:
                 print(f"[WARN] Elevation data missing in successful Open Topo Data response for {latitude},{longitude}. Result: {result}")
        else:
            print(f"[WARN] Open Topo Data API request failed or returned no results for {latitude},{longitude}. Status: {data.get('status')}, Response: {data}")

    except requests.exceptions.Timeout:
        print(f"[ERROR] Timeout while fetching elevation data from Open Topo Data for {latitude},{longitude}")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to fetch elevation data from Open Topo Data for {latitude},{longitude}: {e}")
        # You might get ConnectionRefused if the service is down, handle it gracefully
    except ValueError as e: # Catches JSON decoding errors
        print(f"[ERROR] Failed to parse JSON response from Open Topo Data for {latitude},{longitude}: {e}")
        print(f"Raw response content: {response.text if 'response' in locals() else 'N/A'}")
    except Exception as e:
        print(f"[ERROR] Unexpected error fetching elevation from Open Topo Data for {latitude},{longitude}: {e}")

    return elevation
