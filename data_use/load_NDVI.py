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

    start_date = (date - pd.Timedelta(days=15)).strftime('%Y-%m-%d')
    end_date = (date + pd.Timedelta(days=15)).strftime('%Y-%m-%d')

    ndvi_value = None
    try:
        # Use the correct, updated dataset ID and apply scaling factor
        collection = ee.ImageCollection('MODIS/061/MOD13Q1') \
                       .filterDate(start_date, end_date) \
                       .select('NDVI')

        # Define the point of interest with CRS
        # proj='EPSG:4326' is crucial for Image.sample
        point = ee.Geometry.Point([float(longitude), float(latitude)], proj='EPSG:4326')

        # Get the NDVI value for the point, take the mean over the period
        mean_ndvi = collection.mean()

        # Sample the image at the point. Scale is important for MODIS.
        # Using the nominal scale of MOD13Q1 (250m)
        scale = 250
        ndvi_data = mean_ndvi.sample(point, scale).first()

        # Check if data was found and extract the NDVI value (More robust check)
        if ndvi_data is not None: # 1. Check if ndvi_data object itself is valid
            ndvi_property = ndvi_data.get('NDVI') # 2. Try to get the 'NDVI' property
            if ndvi_property is not None: # 3. Check if the property exists
                # 4. Only call getInfo() if property exists
                raw_ndvi = ndvi_property.getInfo()
                if raw_ndvi is not None: # 5. Check if getInfo() returned a value
                    ndvi_value = raw_ndvi * 0.0001
                else:
                    # This case might occur if the property exists but its value is explicitly null
                    print(f"[WARN] NDVI property exists but its value is null for {latitude},{longitude} between {start_date} and {end_date}")
            else:
                # Property 'NDVI' not found in the sampled feature
                print(f"[WARN] 'NDVI' property not found in sampled data for {latitude},{longitude} between {start_date} and {end_date}")
        else:
             # This handles the case where .sample().first() returned null
             print(f"[WARN] No NDVI data feature found (sample returned null) for {latitude},{longitude} between {start_date} and {end_date}")


    # More specific error handling for GEE
    except ee.ee_exception.EEException as e:
        print(f"[ERROR] Earth Engine error fetching NDVI for {latitude},{longitude}: {e}")
        # Common errors: Computation timed out., User memory limit exceeded., etc.
    except Exception as e:
        print(f"[ERROR] Unexpected error fetching NDVI for {latitude},{longitude}: {e}")

    return ndvi_value
