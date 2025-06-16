import argparse
import pandas as pd
from pathlib import Path

# Assuming data_path.py is in data_use directory
from data_use.data_path import get_prediction_paths
from modules.graph_module import graph_module
from utiles.estimate_time import add_datetime_column

def visualize_and_compare(data_number: int, use_simgraph: bool = False):
    """
    Generates a comparison GIF visualizing the difference between actual and 
    predicted/similar wildfire spread and calculates the overlap percentage.
    """
    print(f"--- Starting Comparison GIF Generation for data_number: {data_number} ---")

    # 1. Get file paths
    try:
        paths = get_prediction_paths(data_number)
        actual_path = paths['all_nodes']
        
        if use_simgraph:
            print("--- Comparing against SIMGRAPH data ---")
            predicted_path_orig = paths['predicted']
            predicted_path = predicted_path_orig.parent / f"simgraph_{predicted_path_orig.name}"
            gif_name_suffix = f"simgraph_{data_number}"
        else:
            print("--- Comparing against PREDICTION data ---")
            predicted_path = paths['predicted']
            gif_name_suffix = str(data_number)

    except ValueError as e:
        print(f"[ERROR] {e}. Please use a valid data number.")
        return

    # Check if files exist
    if not actual_path.exists():
        print(f"[ERROR] Actual fire data not found at: {actual_path}")
        return
    if not predicted_path.exists():
        print(f"[ERROR] Comparison data not found at: {predicted_path}")
        if use_simgraph:
            print("Please run `Wildfire_spread_graph.py --run_mode similar` first.")
        else:
            print("Please run the prediction script first to generate this file.")
        return

    print(f"Loading actual data from: {actual_path}")
    print(f"Loading comparison data from: {predicted_path}")

    # 2. Load data
    actual_df = pd.read_csv(actual_path)
    predicted_df = pd.read_csv(predicted_path)

    # Ensure latitude and longitude columns exist
    for df_name, df in [("Actual", actual_df), ("Comparison", predicted_df)]:
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            print(f"[ERROR] {df_name} data must contain 'latitude' and 'longitude' columns.")
            return

    # 3. Calculate Overlap and prepare for visualization
    actual_df['location'] = list(zip(actual_df['latitude'].round(6), actual_df['longitude'].round(6)))
    predicted_df['location'] = list(zip(predicted_df['latitude'].round(6), predicted_df['longitude'].round(6)))

    actual_points = set(actual_df['location'])
    predicted_points = set(predicted_df['location'])

    intersection = actual_points.intersection(predicted_points)
    union = actual_points.union(predicted_points)
    actual_only_points = actual_points - intersection
    predicted_only_points = predicted_points - intersection

    if not union:
        print("No fire points found in either dataset.")
        return

    overlap_percentage = (len(intersection) / len(union)) * 100
    print("\n--- Overlap Analysis ---")
    print(f"Actual fire points: {len(actual_points)}")
    print(f"Comparison points: {len(predicted_points)}")
    print(f"Common points (Intersection): {len(intersection)}")
    print(f"Total unique points (Union): {len(union)}")
    print(f"Overlap Percentage (Intersection / Union): {overlap_percentage:.2f}%")

    # 4. Create a combined DataFrame for the graph_module
    
    # Filter original dataframes to get the rows for each category
    actual_only_df = actual_df[actual_df['location'].isin(actual_only_points)].copy()
    actual_only_df['comparison_type'] = 'Actual Only'
    
    predicted_only_df = predicted_df[predicted_df['location'].isin(predicted_only_points)].copy()
    predicted_only_df['comparison_type'] = 'Predicted Only'
    
    # For overlap, we can take the data from the actual_df
    overlap_df = actual_df[actual_df['location'].isin(intersection)].copy()
    overlap_df['comparison_type'] = 'Overlap'

    # Combine all parts into a single DataFrame
    comparison_df = pd.concat([actual_only_df, predicted_only_df, overlap_df], ignore_index=True)

    # Ensure the combined dataframe has a 'date' column for the graph module
    # The `add_datetime_column` function can create it from 'acq_date' and 'acq_time'
    if 'date' not in comparison_df.columns:
        if 'acq_date' in comparison_df.columns and 'acq_time' in comparison_df.columns:
            comparison_df = add_datetime_column(comparison_df, 'acq_date', 'acq_time')
        else:
            # If no date columns, create a dummy date to allow graph_module to run
            print("[WARN] No 'acq_date' or 'acq_time' columns found. Using a single timestamp for all points.")
            comparison_df['date'] = pd.Timestamp.now()
    
    # Add dummy confidence if it doesn't exist
    if 'confidence' not in comparison_df.columns:
        comparison_df['confidence'] = 'h'


    # 5. Calculate bounds and call graph module
    print("\n--- Generating Comparison GIF ---")
    lat_min, lat_max = comparison_df['latitude'].min(), comparison_df['latitude'].max()
    lon_min, lon_max = comparison_df['longitude'].min(), comparison_df['longitude'].max()
    margin = 0.1
    lat_margin = (lat_max - lat_min) * margin
    lon_margin = (lon_max - lon_min) * margin
    latlon_bounds = (lat_min - lat_margin, lat_max + lat_margin, lon_min - lon_margin, lon_max + lon_margin)

    # Call the graph module
    graph_module(comparison_df, gif_name_suffix, latlon_bounds)
    
    print("\n--- Comparison GIF Generation Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualize and compare actual vs. predicted wildfire spread.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data_number",
        type=int,
        required=True,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="The data number to process for visualization."
    )
    parser.add_argument(
        '--use_simgraph',
        action='store_true',
        help="If specified, compares the actual data against the 'simgraph' dummy data instead of the real prediction."
    )
    args = parser.parse_args()
    visualize_and_compare(args.data_number, args.use_simgraph) 