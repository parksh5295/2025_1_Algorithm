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
from graph.build_graph import cluster_and_build_graph
from graph.snapshot import draw_graph_snapshot
from utiles.gen_gif import generate_gif_for_dataset

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

    # 8. Graph creation (NetworkX/Matplotlib)
    print("--- Using original NetworkX/Matplotlib style graph generation ---")
    
    # Time grouping (15-minute increments)
    df['date_10min'] = pd.to_datetime(df['date']).dt.floor('15T')
    interval_groups = list(df.groupby('date_10min'))
    total_intervals = len(interval_groups)
    print(f"Found {total_intervals} unique 15-min intervals to process.")
    if total_intervals == 0:
        print("No data to process.")
        return

    # File identifier
    filenumber = f"simgraph_original_{args.data_number}"
    
    # Create frame directory
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)
    frame_base_dir = os.path.join(project_root, 'data', 'graph')
    os.makedirs(os.path.join(frame_base_dir, 'frame', f"frame_{filenumber}"), exist_ok=True)

    # Create cumulative frames
    cumulative_df = pd.DataFrame(columns=df.columns)
    for i, (interval, interval_df) in enumerate(interval_groups):
        sequence_id = i + 1
        cumulative_df = pd.concat([cumulative_df, interval_df], ignore_index=True)
        
        # Create graph and snapshot (original style)
        _processed_df, nodes_df, G = cluster_and_build_graph(cumulative_df.copy())
        draw_graph_snapshot(G, filenumber, sequence_id, latlon_bounds=latlon_bounds)
        print(f"   Snapshot saved for sequence {sequence_id}")

    # Create GIF (original style)
    print(f"Generating GIF for {filenumber}...")
    generate_gif_for_dataset(
        filenumber=filenumber,
        frame_base_dir=frame_base_dir,
        output_gif_name=f"{filenumber}.gif",
        duration=0.25,
        frame_image_pattern='*.png'
    )
    
    print(f"--- Simgraph original style GIF created successfully ---")
    print(f"--- Output file: {filenumber}.gif ---")

if __name__ == '__main__':
    main() 