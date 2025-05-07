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
        print(f"[DEBUG] GEE NDVI Date Range: {start_date_str} to {end_date_inclusive_dt.strftime('%Y-%m-%d')} (exclusive end: {end_date_exclusive_str})") # 디버깅 메시지 추가
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
        ndvi_data = mean_ndvi.sample(point, scale).first()

        # --- Revised Error Handling Block START ---
        try:
            # Check if data was found and extract the NDVI value (More robust check)
            if ndvi_data is not None: # 1. Check if ndvi_data object itself is Python None
                ndvi_property = ndvi_data.get('NDVI') # 2. Try to get the 'NDVI' property (Might raise error here)
                if ndvi_property is not None: # 3. Check if the property exists (and is not GEE null)
                    # 4. Only call getInfo() if property exists
                    raw_ndvi = ndvi_property.getInfo() # (Might raise error here too)
                    if raw_ndvi is not None: # 5. Check if getInfo() returned a non-null value
                        ndvi_value = raw_ndvi * 0.0001
                    else:
                        # This case might occur if the property exists but its value is explicitly null server-side
                        print(f"[WARN] NDVI getInfo() returned null for {latitude},{longitude} between {start_date_str} and {end_date_exclusive_str}")
                else:
                    # Property 'NDVI' not found or was null in the sampled feature
                    print(f"[WARN] 'NDVI' property is null or missing in sampled data for {latitude},{longitude} between {start_date_str} and {end_date_exclusive_str}")
            else:
                 # This handles the case where .sample().first() returned Python None
                 print(f"[WARN] Sampled data (ndvi_data) is None for {latitude},{longitude} between {start_date_str} and {end_date_exclusive_str}")

        except ee.ee_exception.EEException as gee_err:
             # Catch potential GEE errors during .get() or .getInfo() specifically
             print(f"[ERROR] GEE Exception during NDVI property access/retrieval for {latitude},{longitude}: {gee_err}")
             # Check if it's the specific "Parameter 'object' is required" error
             if "Parameter 'object' is required" in str(gee_err):
                 print(f"[INFO] This likely means the sampled data object (ndvi_data) was invalid/null internally in GEE.")
             # Ensure ndvi_value remains None in case of error
             ndvi_value = None
        except Exception as e:
             # Catch any other unexpected errors during this block
             print(f"[ERROR] Unexpected error during NDVI property access for {latitude},{longitude}: {e}")
             ndvi_value = None
        # --- Revised Error Handling Block END ---

    except Exception as e:
        print(f"[ERROR] Unexpected error setup or outside property access for NDVI {latitude},{longitude}: {e}")
        ndvi_value = None # Ensure None on other errors too

    return ndvi_value
