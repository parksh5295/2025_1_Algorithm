# from Google Earth Engine
# load NDVI data from MODIS

import ee
import pandas as pd # Import pandas for timestamp check

# initialize GEE with project ID
try:
    # Use the provided GCP Project ID
    ee.Initialize(project='sustained-drake-458413-e1', opt_url='https://earthengine-highvolume.googleapis.com')
    print("Google Earth Engine initialized successfully with project sustained-drake-458413-e1.")
except Exception as e:
    print(f"Failed to initialize Google Earth Engine: {e}")
    # Depending on requirements, either raise the exception or handle it (e.g., return None in get_ndvi)
    raise

def get_ndvi(lat, lon, date):
    try:
        # Ensure date is a suitable format for ee.Date (e.g., string or datetime)
        if isinstance(date, pd.Timestamp):
            # Convert pandas Timestamp to string format GEE understands well
            date_string = date.strftime('%Y-%m-%d') 
        else:
            # Assume it might already be a string or needs conversion
            date_string = str(date).split(' ')[0] # Basic handling for potential datetime strings

        # Create Earth Engine Geometry Point with explicit CRS
        point = ee.Geometry.Point([lon, lat], proj='EPSG:4326')

        # Use the updated MODIS dataset ID (Version 6.1)
        dataset = ee.ImageCollection('MODIS/061/MOD13Q1') \
            .filterDate(ee.Date(date_string), ee.Date(date_string).advance(1, 'day')) \
            .filterBounds(point) \
            .select('NDVI')

        # Calculate the mean NDVI for the period (usually one image) and sample at the point
        # Use .first() to handle cases where multiple images might exist (e.g., overlapping scenes)
        image = dataset.mean() # Or use dataset.first() if only one image is expected
        
        # Check if an image exists for the date range
        # .size() is a server-side operation
        count = dataset.size().getInfo()
        if count == 0:
            print(f"[WARN] No MODIS image found for date {date_string} at ({lat}, {lon}). Returning None for NDVI.")
            return None
            
        # Sample the image - Use unweighted reducer if needed, getInfo() can be slow
        # Consider scale carefully - 250m is MODIS NDVI resolution
        # Adding error handling for sampling
        sampled_data = image.sample(region=point, scale=250, projection='EPSG:4326').first()
        
        # Check if sampling returned a feature
        if sampled_data is None or sampled_data.getInfo() is None:
             print(f"[WARN] Sampling returned no feature for date {date_string} at ({lat}, {lon}). Returning None for NDVI.")
             return None

        # Safely get the NDVI property
        ndvi_value = sampled_data.get('NDVI').getInfo()
        
        # GEE NDVI values are scaled by 10000, convert to standard -1 to 1 range
        if ndvi_value is not None:
            ndvi_value = ndvi_value * 0.0001 
            
        return ndvi_value

    except ee.EEException as e:
        print(f"[ERROR] Earth Engine error getting NDVI for date {date_string} at ({lat}, {lon}): {e}")
        return None
    except Exception as e:
        # Catch other potential errors (e.g., network issues with getInfo)
        print(f"[ERROR] Unexpected error getting NDVI for date {date_string} at ({lat}, {lon}): {e}")
        return None
