# from Google Earth Engine
# load NDVI data from MODIS

import ee
import pandas as pd # Import pandas for timestamp check
from pathlib import Path
import sys # For potential exit on error

# Service account information
SERVICE_ACCOUNT = 'earth-engine-accessor@sustained-drake-458413-e1.iam.gserviceaccount.com'
# Key file name (3 levels up from script location)
KEY_FILENAME = 'serivce_account_key.json'

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
        print(f"[WARN] Skipping NDVI fetch for {latitude},{longitude} due to GEE initialization failure.")
        return None

    # Check date format and convert (existing code remains)
    if not isinstance(date, pd.Timestamp):
        try:
            date = pd.to_datetime(date)
        except ValueError:
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
    try:
        # Use the correct, updated dataset ID and apply scaling factor
        # Filter for the period *before* and *including* the event date
        print(f"[DEBUG] GEE NDVI Date Range: {start_date_str} to {end_date_inclusive_dt.strftime('%Y-%m-%d')} (exclusive end: {end_date_exclusive_str})")
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
        print(f"[INFO] Attempting to get NDVI at exact point for {latitude},{longitude} between {start_date_str} and {end_date_exclusive_str}")
        ndvi_data_at_point = mean_ndvi.sample(point, scale).first()

        primary_ndvi_value = None # First attempt result save variable
        try:
            # Check if data was found and extract the NDVI value (More robust check)
            if ndvi_data_at_point is not None: # 1. Check if ndvi_data_at_point object itself is Python None
                ndvi_property = ndvi_data_at_point.get('NDVI') # 2. Try to get the 'NDVI' property
                if ndvi_property is not None: # 3. Check if the property exists (and is not GEE null)
                    # 4. Only call getInfo() if property exists
                    raw_ndvi = ndvi_property.getInfo() # (Might raise error here too)
                    if raw_ndvi is not None: # 5. Check if getInfo() returned a non-null value
                        primary_ndvi_value = raw_ndvi * 0.0001
                        print(f"[INFO] NDVI found at exact point: {primary_ndvi_value}")
                    else:
                        print(f"[WARN] NDVI getInfo() returned null for exact point {latitude},{longitude}")
                else:
                    print(f"[WARN] 'NDVI' property is null or missing in sampled data for exact point {latitude},{longitude}")
            else:
                 print(f"[WARN] Sampled data (ndvi_data_at_point) is None for exact point {latitude},{longitude}")

        except ee.ee_exception.EEException as gee_err:
             print(f"[ERROR] GEE Exception during NDVI property access for exact point {latitude},{longitude}: {gee_err}")
             if "Parameter 'object' is required" in str(gee_err):
                 print(f"[INFO] This likely means the sampled data object (ndvi_data_at_point) was invalid/null internally in GEE for exact point.")
        except Exception as e:
             print(f"[ERROR] Unexpected error during NDVI property access for exact point {latitude},{longitude}: {e}")

        # If a value was obtained in the first attempt, assign it to ndvi_value
        if primary_ndvi_value is not None:
            ndvi_value = primary_ndvi_value
        else:
            # Second attempt: Search for NDVI in the nearby region
            print(f"[WARN] NDVI not found at exact point. Attempting to get NDVI from nearby region for {latitude},{longitude}...")
            try:
                buffer_radius_meters = 1000 # Buffer radius (e.g., 1km)
                buffered_point_geometry = point.buffer(buffer_radius_meters)
                
                # Calculate the average NDVI for the nearby region (mean_ndvi image is reused)
                # .get('NDVI') is used to directly get the average value of the 'NDVI' band from GEE server
                ndvi_data_buffered_obj = mean_ndvi.reduceRegion(
                    reducer=ee.Reducer.mean(), # Use the average value
                    geometry=buffered_point_geometry,
                    scale=scale, # Use the same resolution
                    crs='EPSG:4326', 
                    maxPixels=1e9 
                ).get('NDVI') # The result is a GEE object (ee.Number, etc.)

                # Extract the actual value from the GEE object
                if ndvi_data_buffered_obj is not None:
                    # .getInfo() brings data to the client side
                    raw_ndvi_buffered = ndvi_data_buffered_obj.getInfo() 
                    if raw_ndvi_buffered is not None:
                        ndvi_value = raw_ndvi_buffered * 0.0001
                        print(f"[INFO] NDVI found in nearby region (buffer {buffer_radius_meters}m): {ndvi_value}")
                    else:
                        # The result of .getInfo() is Python None
                        print(f"[WARN] NDVI getInfo() returned null for buffered region at {latitude},{longitude}")
                        # ndvi_value is already initialized to None
                else:
                    # The result of .get('NDVI') is GEE null or the property is missing
                    print(f"[WARN] 'NDVI' property is null or missing in buffered region data for {latitude},{longitude}")
                    # ndvi_value is already initialized to None
                
            except ee.ee_exception.EEException as gee_err_buffer:
                print(f"[ERROR] GEE Exception during NDVI retrieval for buffered region at {latitude},{longitude}: {gee_err_buffer}")
                # ndvi_value is already initialized to None
            except Exception as e_buffer:
                print(f"[ERROR] Unexpected error during NDVI retrieval for buffered region at {latitude},{longitude}: {e_buffer}")
                # ndvi_value is already initialized to None
            # --- 2nd attempt (nearby region) logic end ---

    except Exception as e_outer:
        # This try block is for catching errors in GEE collection filtering, point creation, etc.
        print(f"[ERROR] Outer GEE setup error for NDVI {latitude},{longitude}: {e_outer}")
        # ndvi_value is already initialized to None

    if ndvi_value is None:
        print(f"[WARN] Final NDVI value is None for {latitude},{longitude} after all attempts.")

    return ndvi_value
