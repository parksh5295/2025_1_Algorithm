import sys
import os
import pandas as pd
import argparse

print("--- Runtime Environment Check ---")
print(f"Python Executable: {sys.executable}")
print("System Path (sys.path):")
for path in sys.path:
    print(f"  - {path}")
print("--- End Runtime Environment Check ---\n")

from data_use.data_path import get_prediction_paths
from utiles.estimate_time import estimate_fire_spread_times, add_datetime_column
from modules.graph_module import graph_module

def main():
    # 0. argparser
    parser = argparse.ArgumentParser(description='Simgraph to Original Style GIF Generator')
    parser.add_argument('--data_number', type=int, required=True, help="The data number to process.")
    args = parser.parse_args()

    print(f"--- Running Simgraph to Original Style GIF for data_number: {args.data_number} ---")

    # 1. Get simgraph file path
    try:
        paths = get_prediction_paths(args.data_number)
        predicted_path_orig = paths['predicted']
        simgraph_path = predicted_path_orig.parent / f"simgraph_{predicted_path_orig.name}"
    except (ValueError, FileNotFoundError) as e:
        print(f"[ERROR] Could not retrieve paths: {e}")
        return

    if not os.path.exists(simgraph_path):
        print(f"[ERROR] Simgraph data file not found at {simgraph_path}.")
        print("Please run '--run_mode similar' first to generate it.")
        return

    # 2. Load simgraph data
    print(f"[INFO] Loading simgraph data from: {simgraph_path}")
    df = pd.read_csv(simgraph_path)

    # 3. Add dummy environmental columns
    env_columns = ['elevation', 'ndvi', 'wind_speed', 'wind_direction', 'temperature', 
                   'humidity', 'precipitation', 'soil_moisture', 'vegetation_density']
    for col in env_columns:
        if col not in df.columns:
            df[col] = 0

    # 4. Add date column
    if 'date' not in df.columns:
        if 'acq_date' in df.columns and 'acq_time' in df.columns:
            df = add_datetime_column(df, 'acq_date', 'acq_time')
        else:
            print("[ERROR] No 'date' or ('acq_date'+'acq_time') columns found in simgraph data.")
            return

    # 5. Filter confidence column
    if 'confidence' not in df.columns:
        df['confidence'] = 'h'
    df = df[df['confidence'].isin(['h', 'n'])]

    # 6. Estimate fire spread times
    df = estimate_fire_spread_times(df)
    
    # 7. Calculate map bounds
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

    # 8. Call graph module (clearly distinguish file names)
    gif_identifier = f"simgraph_original_{args.data_number}"
    graph_module(df, gif_identifier, latlon_bounds=latlon_bounds)
    
    print(f"--- Simgraph original style GIF created successfully ---")
    print(f"--- Output file: animation_{gif_identifier}.gif ---")

if __name__ == '__main__':
    main() 