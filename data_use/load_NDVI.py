# from Google Earth Engine
# load NDVI data from MODIS

import ee
import pandas as pd # Import pandas for timestamp check
from pathlib import Path
import sys # For potential exit on error
import numpy as np
from datetime import datetime, timedelta
import time
from typing import List, Tuple, Dict, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Service account information
SERVICE_ACCOUNT = 'earth-engine-accessor@sustained-drake-458413-e1.iam.gserviceaccount.com'
# Key file name (3 levels up from script location)
KEY_FILENAME = 'service_account_key.json'

# GEE initialization status flag
gee_initialized = False

# Calculate the absolute path to the key file based on the absolute path of the script file
try:
    # Handle cases where __file__ is not defined (e.g., REPL)
    if '__file__' not in locals():
        raise NameError("__file__ is not defined. Cannot determine relative path.")

    current_script_path = Path(__file__).resolve()
    # Check if parents[2] is valid (i.e., not going beyond the root)
    if len(current_script_path.parents) < 3:
         raise FileNotFoundError("Cannot go up three parent directories from script location.")

    KEY_FILE_PATH = current_script_path.parents[2] / KEY_FILENAME

    # Check if the file exists
    if not KEY_FILE_PATH.is_file():
        print(f"[ERROR] Service account key file not found at expected path: {KEY_FILE_PATH}")
        # If the file does not exist, GEE initialization is not possible, and subsequent code cannot proceed
        # sys.exit(1) # Alternatively, handle this differently
    else:
        # Create service account credentials
        credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, str(KEY_FILE_PATH))

        # Initialize GEE (using service account)
        # opt_url continues to use the high-volume endpoint
        ee.Initialize(credentials=credentials, project='sustained-drake-458413-e1', opt_url='https://earthengine-highvolume.googleapis.com')
        print("[INFO] Earth Engine initialized successfully using service account.")
        gee_initialized = True

except NameError as e:
    print(f"[ERROR] Could not determine key file path: {e}")
except FileNotFoundError as e:
    print(f"[ERROR] Problem finding or accessing key file path: {e}")
except Exception as e:
    # If ee.Initialize fails, ee.ee_exception.EEException may occur
    print(f"[ERROR] Failed to initialize Earth Engine with service account: {e}")

# If initialization fails, fallback or explicit error handling
if not gee_initialized:
    print("[WARN] Earth Engine initialization failed. NDVI data will likely be unavailable.")
    # If needed, try default authentication here or terminate the program.
    # ee.Initialize() # For example, try default authentication (may fail on server)
    # sys.exit("Critical error: Earth Engine could not be initialized.")

def get_ndvi(latitude, longitude, date):
    # If GEE was not initialized successfully, NDVI calculation is not possible
    if not gee_initialized:
        # This is a critical setup issue, print immediately
        print(f"[WARN] Skipping NDVI fetch for {latitude},{longitude} due to GEE initialization failure.")
        return None

    # Check date format and convert (existing code remains)
    if not isinstance(date, pd.Timestamp):
        try:
            date = pd.to_datetime(date)
        except ValueError:
            # Invalid input, print immediately
            print(f"[WARN] Invalid date format for NDVI data: {date}. Skipping.")
            return None

    # Define a date window ending *on* the event date (inclusive)
    window_days = 30 # Look back 30 days from the event date
    start_date_dt = date - pd.Timedelta(days=window_days)
    end_date_inclusive_dt = date # Target date

    # Format dates for GEE filterDate(start, end) where 'end' is exclusive
    start_date_str = start_date_dt.strftime('%Y-%m-%d')
    # To include the end_date_inclusive_dt, the filter's end date must be the next day
    end_date_exclusive_str = (end_date_inclusive_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    ndvi_value = None
    log_messages = [] # Temporary log storage

    try:
        log_messages.append(f"[DEBUG] GEE NDVI Date Range for {latitude},{longitude}: {start_date_str} to {end_date_inclusive_dt.strftime('%Y-%m-%d')} (exclusive end: {end_date_exclusive_str})")
        collection = ee.ImageCollection('MODIS/061/MOD13Q1') \
                       .filterDate(start_date_str, end_date_exclusive_str) \
                       .select('NDVI')

        # Define the point of interest with CRS
        # proj='EPSG:4326' is crucial for Image.sample
        point = ee.Geometry.Point([float(longitude), float(latitude)], proj='EPSG:4326')

        # Get the NDVI value for the point, take the mean over the period
        mean_ndvi = collection.mean()

        # Sample the image at the point. Scale is important for MODIS.
        # Using the nominal scale of MOD13Q1 (250m)
        scale = 250
        
        # First attempt: Extract NDVI values for the exact point
        log_messages.append(f"[DEBUG] Attempting NDVI at exact point for {latitude},{longitude}")
        ndvi_data_at_point = mean_ndvi.sample(point, scale).first()
        primary_ndvi_value = None
        try:
            if ndvi_data_at_point is not None:
                ndvi_property = ndvi_data_at_point.get('NDVI') 
                if ndvi_property is not None: 
                    raw_ndvi = ndvi_property.getInfo() 
                    if raw_ndvi is not None: 
                        primary_ndvi_value = raw_ndvi * 0.0001
                        log_messages.append(f"[INFO] NDVI found at exact point for {latitude},{longitude}: {primary_ndvi_value}")
                    else:
                        log_messages.append(f"[DEBUG] NDVI getInfo() returned null for exact point {latitude},{longitude}")
                else:
                    log_messages.append(f"[DEBUG] 'NDVI' property is null or missing in sampled data for exact point {latitude},{longitude}")
            else:
                 log_messages.append(f"[DEBUG] Sampled data (ndvi_data_at_point) is None for exact point {latitude},{longitude}")

        except ee.ee_exception.EEException as gee_err:
            log_messages.append(f"[DEBUG] GEE Exception (1st attempt) for {latitude},{longitude}: {gee_err}")
            if "Parameter 'object' is required" in str(gee_err):
                 log_messages.append(f"[DEBUG] This likely means the sampled data object (ndvi_data_at_point) was invalid/null internally in GEE for exact point.")
        except Exception as e:
            log_messages.append(f"[DEBUG] Unexpected error (1st attempt) for {latitude},{longitude}: {e}")

        if primary_ndvi_value is not None:
            ndvi_value = primary_ndvi_value
        else:
            # Second attempt
            log_messages.append(f"[INFO] NDVI not at exact point for {latitude},{longitude}. Trying nearby region...")
            try:
                buffer_radius_meters = 1000 # Buffer radius (e.g., 1km)
                buffered_point_geometry = point.buffer(buffer_radius_meters)
                
                # Calculate the average NDVI for the nearby region (mean_ndvi image is reused)
                # .get('NDVI') is used to directly get the average value of the 'NDVI' band from GEE server
                ndvi_data_buffered_obj = mean_ndvi.reduceRegion(
                    reducer=ee.Reducer.mean(), # Use the average value
                    geometry=buffered_point_geometry,
                    scale=scale, 
                    crs='EPSG:4326', 
                    maxPixels=1e9 
                ).get('NDVI') # The result is a GEE object (ee.Number, etc.)

                # Extract the actual value from the GEE object
                if ndvi_data_buffered_obj is not None:
                    # .getInfo() brings data to the client side
                    raw_ndvi_buffered = ndvi_data_buffered_obj.getInfo() 
                    if raw_ndvi_buffered is not None:
                        ndvi_value = raw_ndvi_buffered * 0.0001
                        log_messages.append(f"[INFO] NDVI found in nearby region (buffer {buffer_radius_meters}m) for {latitude},{longitude}: {ndvi_value}")
                    else:
                        log_messages.append(f"[WARN] NDVI getInfo() returned null for buffered region at {latitude},{longitude} (2nd attempt)")
                else:
                    log_messages.append(f"[WARN] 'NDVI' property is null or missing in buffered region data for {latitude},{longitude} (2nd attempt)")
                
            except ee.ee_exception.EEException as gee_err_buffer:
                # For 2nd attempt GEE errors, we might want a higher log level if this is critical before final failure
                log_messages.append(f"[WARN] GEE Exception (2nd attempt) for {latitude},{longitude}: {gee_err_buffer}")
            except Exception as e_buffer:
                log_messages.append(f"[WARN] Unexpected error (2nd attempt) for {latitude},{longitude}: {e_buffer}")

    except Exception as e_outer:
        # This is a critical setup error for this specific call, print immediately
        print(f"[ERROR] Outer GEE setup error for NDVI {latitude},{longitude}: {e_outer}")
        # Print any logs gathered so far for context before returning
        for msg in log_messages:
            print(msg)
        return None

    # Print all gathered log messages
    for msg in log_messages:
        print(msg)

    # Final error message if all attempts failed
    if ndvi_value is None:
        print(f"[ERROR] Failed to get NDVI for {latitude},{longitude} after all attempts.")
    
    return ndvi_value

def get_NDVI_for_batch(lats: List[float], lons: List[float], date_str: str, 
                      batch_size: int = 30) -> Dict[Tuple[float, float], float]:
    """
    Get NDVI values for a batch of coordinates.
    
    Args:
        lats: List of latitudes
        lons: List of longitudes
        date_str: Date string in 'YYYY-MM-DD' format
        batch_size: Number of coordinates to process in each batch
        
    Returns:
        Dictionary mapping (lat, lon) tuples to NDVI values
    """
    results = {}
    
    # Process coordinates in batches
    for i in range(0, len(lats), batch_size):
        batch_lats = lats[i:i + batch_size]
        batch_lons = lons[i:i + batch_size]
        
        # Create feature collection for the batch
        features = []
        for lat, lon in zip(batch_lats, batch_lons):
            point = ee.Geometry.Point([lon, lat])
            features.append(ee.Feature(point, {'lat': lat, 'lon': lon}))
        fc = ee.FeatureCollection(features)
        
        # Get NDVI for the batch
        batch_results = get_NDVI_for_date(fc, date_str)
        results.update(batch_results)
        
        # Add a small delay between batches to avoid overwhelming the API
        if i + batch_size < len(lats):
            time.sleep(1)
    
    return results

def get_NDVI_for_date(fc: ee.FeatureCollection, date_str: str) -> Dict[Tuple[float, float], float]:
    """
    Get NDVI values for a feature collection on a specific date.
    
    Args:
        fc: Earth Engine FeatureCollection containing points
        date_str: Date string in 'YYYY-MM-DD' format
        
    Returns:
        Dictionary mapping (lat, lon) tuples to NDVI values
    """
    date = datetime.strptime(date_str, '%Y-%m-%d')
    start_date = (date - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = (date + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Get MODIS NDVI data
    modis = ee.ImageCollection('MODIS/006/MOD13Q1')
    ndvi = modis.filterDate(start_date, end_date).select('NDVI')
    
    # Get the most recent image
    image = ndvi.sort('system:time_start', False).first()
    
    # Sample the image at the points
    sampled = image.sampleRegions(
        collection=fc,
        scale=250,  # MODIS resolution
        geometries=True
    )
    
    # Get the results
    results = {}
    for feature in sampled.getInfo()['features']:
        lat = feature['properties']['lat']
        lon = feature['properties']['lon']
        ndvi_value = feature['properties']['NDVI'] / 10000.0  # Scale factor
        results[(lat, lon)] = ndvi_value
    
    return results

def load_NDVI(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    """
    Load NDVI data for all coordinates in the DataFrame.
    
    Args:
        df: DataFrame containing 'latitude' and 'longitude' columns
        date_str: Date string in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame with added 'NDVI' column
    """
    # Extract coordinates
    lats = df['latitude'].tolist()
    lons = df['longitude'].tolist()
    
    # Get NDVI values for all coordinates in batches
    ndvi_values = get_NDVI_for_batch(lats, lons, date_str)
    
    # Add NDVI values to DataFrame
    df['NDVI'] = df.apply(lambda row: ndvi_values.get((row['latitude'], row['longitude']), np.nan), axis=1)
    
    return df
