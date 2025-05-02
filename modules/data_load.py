import pandas as pd
import os
import sys

# Move imports into the try block and remove original import lines
# from data_use.data_requests import add_environmental_features
# from data_use.dtype_optimize import optimize_loaded_df


# --- add project root to sys.path ---
# assume this file (data_load.py) is in 'project_root/modules/'
# data_use folder is in 'project_root/data_use/'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # 'modules' folder's upper folder
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"[INFO] Added project root to sys.path: {project_root}")
# --- sys.path modification end ---

try:
    # Indent the imports inside the try block
    from data_use.data_requests import add_environmental_features
    from data_use.dtype_optimize import optimize_loaded_df
except ImportError as e:
    print(f"[ERROR] Required modules (data_requests, dtype_optimize) could not be imported: {e}")
    print("Please ensure 'data_use' directory exists at the project root and is accessible.")
    sys.exit(1) # if required modules are not found, exit

def load_and_enrich_data(csv_path, date_col='date', lat_col='latitude', lon_col='longitude', time_col=None, max_workers=None):
    print(f"--- Starting Data Loading and Enrichment Process ---")
    internal_date_col = 'date' # Internal standard name for date/datetime column

    # 1. Load base CSV data
    print(f"Step 1: Loading base CSV from '{csv_path}'...")
    try:
        base_df = pd.read_csv(csv_path)
        print(f"   Successfully loaded base DataFrame with shape: {base_df.shape}")
    except FileNotFoundError:
        print(f"[ERROR] File not found: {csv_path}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to read CSV file: {e}")
        return None

    # --- Combine Date and Time if time_col is provided --- START ---
    datetime_created = False
    if time_col:
        print(f"   Attempting to combine date ('{date_col}') and time ('{time_col}')...")
        if date_col in base_df.columns and time_col in base_df.columns:
            try:
                # Convert date column to string first to handle potential mixed types
                date_str = base_df[date_col].astype(str)
                # Ensure time column is string and pad with leading zeros if needed (HHMM format)
                time_str = base_df[time_col].astype(str).str.zfill(4)
                # Combine date and time strings
                datetime_str = date_str + ' ' + time_str
                # Convert to datetime objects, coercing errors to NaT (Not a Time)
                base_df[internal_date_col] = pd.to_datetime(datetime_str, format='%Y-%m-%d %H%M', errors='coerce')
                # Check if conversion resulted in any NaT values
                if base_df[internal_date_col].isnull().any():
                    print(f"[WARN] Some values could not be parsed into datetimes from '{date_col}' and '{time_col}'. Resulting rows might be dropped or cause issues later.")
                print(f"   Successfully created '{internal_date_col}' column.")
                datetime_created = True
            except Exception as e:
                print(f"[ERROR] Failed to combine date and time columns: {e}")
                print(f"        Please check formats. Expected date like 'YYYY-MM-DD' and time like 'HHMM'.")
                return None # Stop processing if datetime creation fails
        else:
            print(f"[WARN] Provided date_col ('{date_col}') or time_col ('{time_col}') not found in CSV. Skipping combination.")
    # --- Combine Date and Time if time_col is provided --- END ---

    # 2. Verify required columns for enrichment
    # If datetime was created, check for 'datetime'; otherwise, check for original date_col
    check_date_col = internal_date_col if datetime_created else date_col
    required_cols = {check_date_col, lat_col, lon_col}
    if not required_cols.issubset(base_df.columns):
        missing_cols = required_cols - set(base_df.columns)
        print(f"[ERROR] Base DataFrame is missing required columns for enrichment: {missing_cols}")
        print(f"        Expected columns: {check_date_col}, {lat_col}, {lon_col}")
        return None

    # Rename columns temporarily for add_environmental_features which expects 'date', 'latitude', 'longitude'
    rename_map = {}
    # If datetime was created, rename it to 'date'. Otherwise, rename original date_col if needed.
    if check_date_col != internal_date_col: rename_map[check_date_col] = internal_date_col
    if lat_col != 'latitude': rename_map[lat_col] = 'latitude'
    if lon_col != 'longitude': rename_map[lon_col] = 'longitude'

    # Apply renaming only if necessary
    df_renamed = base_df.rename(columns=rename_map) if rename_map else base_df
    if rename_map:
      print(f"   (Columns temporarily renamed for processing: {rename_map})")

    # Ensure the final date column used for enrichment exists
    if internal_date_col not in df_renamed.columns:
         print(f"[ERROR] Internal date column '{internal_date_col}' not found after renaming steps.")
         return None

    # 3. Enrich data using data_requests (now sequential)
    print(f"\nStep 2: Enriching DataFrame with environmental features (sequentially)...")
    try:
        # Call the sequential version without max_workers argument
        enriched_df = add_environmental_features(df_renamed.copy())
        print(f"   Successfully enriched DataFrame. New shape: {enriched_df.shape}")
    except Exception as e:
        print(f"[ERROR] Failed during data enrichment step: {e}")
        return None

    # 4. Optimize memory usage
    print("\nStep 3: Optimizing memory usage of the final DataFrame...")
    try:
        optimized_df = optimize_loaded_df(enriched_df)
        print(f"   Memory optimization complete.")
    except Exception as e:
        print(f"[ERROR] Failed during memory optimization step: {e}")
        print("[WARN] Returning the enriched but unoptimized DataFrame.")
        return enriched_df

    print("\n--- Data Loading and Enrichment Process Finished ---")
    return optimized_df

# --- Example Usage ---
if __name__ == '__main__':
    print("\n--- Running Example Usage of data_load module ---")

    # IMPORTANT: Replace with the actual path to your base CSV file!
    # Example assumes a 'data' folder exists at the project root
    # example_csv = os.path.join(project_root, 'data', 'your_base_data.csv')
    example_csv = r"C:\Users\LG\Desktop\Algorithm\team_project\data\FRT003801_1974\TB_FFAS_FF_OCCRR_TMSRES_1974.csv" # 예시 경로 (실제 경로로 수정 필요)

    # --- prepare data: SHP -> CSV conversion (temporary test) ---
    if not os.path.exists(example_csv):
        print(f"\n[INFO] Example CSV not found. Trying to create from SHP...")
        try:
            import geopandas as gpd
            # use the SHP path used in test.py
            shp_path = r"C:\Users\LG\Desktop\Algorithm\team_project\data\FRT003801_1974\TB_FFAS_FF_OCCRR_TMSRES_1974.shp"
            temp_gdf = gpd.read_file(shp_path)
            # add latitude, longitude columns (assuming EPSG:5179 -> EPSG:4326 conversion)
            # need to check actual coordinate system (CRS) and conversion
            if temp_gdf.crs is None:
                 print("[WARN] Input SHP has no CRS defined. Assuming EPSG:5179 for conversion to Lat/Lon (EPSG:4326).")
                 temp_gdf.crs = "EPSG:5179" # example coordinate system, need to check actual data
            temp_gdf_4326 = temp_gdf.to_crs(epsg=4326)
            temp_gdf['longitude'] = temp_gdf_4326.geometry.x
            temp_gdf['latitude'] = temp_gdf_4326.geometry.y
            # create date column (assuming OCCRR_DTM format: YYYYMMDDHHMM)
            temp_gdf['date'] = pd.to_datetime(temp_gdf['OCCRR_DTM'], format='%Y%m%d%H%M')
            # save CSV excluding geometry column
            temp_gdf.drop(columns=['geometry']).to_csv(example_csv, index=False, encoding='utf-8-sig') # save with UTF-8 BOM
            print(f"   Created example CSV: {example_csv}")
        except ImportError:
            print("[ERROR] 'geopandas' is required to convert SHP to CSV for the example. Please install it.")
            example_csv = None # failed to create CSV
        except FileNotFoundError:
             print(f"[ERROR] Source SHP file not found at {shp_path}")
             example_csv = None
        except Exception as e:
            print(f"[ERROR] Failed to create example CSV from SHP: {e}")
            example_csv = None
    # --- data preparation end ---


    if example_csv and os.path.exists(example_csv):
        # call load_and_enrich_data function
        # note: CSV converted from SHP has 'latitude', 'longitude', 'date' columns based on the example above
        final_dataframe = load_and_enrich_data(
            csv_path=example_csv,
            date_col='date',        # actual date column name in the CSV file
            lat_col='latitude',     # actual latitude column name in the CSV file
            lon_col='longitude'     # actual longitude column name in the CSV file
            # max_workers=4 # specify thread count if needed
        )

        if final_dataframe is not None:
            print("\n--- Final DataFrame Details ---")
            print("Shape:", final_dataframe.shape)
            print("Columns:", final_dataframe.columns.tolist())
            print("\nInfo:")
            final_dataframe.info(memory_usage='deep')
            print("\nHead:")
            print(final_dataframe.head())
        else:
            print("\n[ERROR] Failed to load and enrich data.")
    else:
        print(f"\n[ERROR] Example CSV file ('{example_csv}') not found or could not be created. Cannot run example.")
