import sys
import os
import pandas as pd

print("--- Runtime Environment Check ---")
print(f"Python Executable: {sys.executable}")
print("System Path (sys.path):")
for path in sys.path:
    print(f"  - {path}")
print("--- End Runtime Environment Check ---\n")

# draw the spread graph of wildfire

import argparse

from data_use.data_path import load_data_path
from modules.data_load import load_and_enrich_data
from utiles.estimate_time import estimate_fire_spread_times
from modules.graph_module import graph_module


def main():
    # 0. argparser
    # Create an instance that can receive argument values
    parser = argparse.ArgumentParser(description='Argparser')

    # Set the argument values to be input (default value can be set)
    parser.add_argument('--data_number', type=int, default=1)
    parser.add_argument('--train_or_test', type=str, default="train")
    parser.add_argument('--draw_figure', type=str, default="N")
    parser.add_argument('--save_figure', type=str, default="N")

    # Save the above in args
    args = parser.parse_args()

    # Output the value of the input arguments
    data_number = args.data_number
    train_or_test = args.train_or_test
    draw_figure = args.draw_figure
    save_figure = args.save_figure


    # 1. Collecting data
    csv_path = load_data_path(data_number)
    
    # Generate path for the cached file
    base, ext = os.path.splitext(csv_path)
    cached_csv_path = f"{base}_with_env{ext}"

    if os.path.exists(cached_csv_path):
        print(f"[INFO] Loading data from cached file: {cached_csv_path}")
        # Important: Ensure date/time columns are parsed correctly if stored as strings
        # And other dtypes are preserved. This might need adjustment based on how
        # load_and_enrich_data structures the final DataFrame.
        # For now, a simple read, assuming columns are correctly typed or will be handled.
        df = pd.read_csv(cached_csv_path)
        # Potentially re-parse date columns if they are not stored in a directly usable format
        # Example: df['date'] = pd.to_datetime(df['date'])
        # This needs to match the output of load_and_enrich_data
        # For now, we assume load_and_enrich_data saves them in a way that pd.read_csv handles well
        # or that subsequent processing (like estimate_fire_spread_times) handles type conversion.
        # A common practice is to ensure 'date' or similar columns are converted after loading.
        # Let's assume 'date' is the primary datetime column after enrichment.
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        if 'start_time' in df.columns: # If clustering renames it
             df['start_time'] = pd.to_datetime(df['start_time'])

    else:
        print(f"[INFO] Cached file not found. Processing original data: {csv_path}")
        df = load_and_enrich_data(
            csv_path=csv_path,
            date_col='acq_date',
            time_col='acq_time',
            lat_col='latitude',
            lon_col='longitude'
        )

        if df is not None:
            print(f"[INFO] Saving enriched data to cache: {cached_csv_path}")
            df.to_csv(cached_csv_path, index=False)
        else:
            print("[ERROR] Data loading and enrichment failed. Exiting.")
            exit()

    if df is None: # Should be redundant if the above logic is correct, but as a safeguard
        print("[ERROR] Data loading and enrichment failed. Exiting.")
        exit()

    df = df[df['confidence'].isin(['h', 'n'])]
    df = estimate_fire_spread_times(df)
    
    # Calculate the overall latitude/longitude bounds with margin
    lat_min_orig, lat_max_orig = df['latitude'].min(), df['latitude'].max()
    lon_min_orig, lon_max_orig = df['longitude'].min(), df['longitude'].max()
    
    # Calculate margin (e.g., 10% of the range)
    lat_range = lat_max_orig - lat_min_orig
    lon_range = lon_max_orig - lon_min_orig
    margin_lat = lat_range * 0.10 # 10% margin
    margin_lon = lon_range * 0.10 # 10% margin
    
    # Apply margin
    lat_min = lat_min_orig - margin_lat
    lat_max = lat_max_orig + margin_lat
    lon_min = lon_min_orig - margin_lon
    lon_max = lon_max_orig + margin_lon
    
    latlon_bounds = (lat_min, lat_max, lon_min, lon_max)

    # 2. Forming a graph
    graph_module(df, data_number, latlon_bounds=latlon_bounds)
    


if __name__ == '__main__':
    main()