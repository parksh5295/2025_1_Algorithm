import sys
import os
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

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


def draw_prediction_graph(prediction_path, nodes_path, output_filename="prediction_graph.html"):
    """
    Visualizes the output of the wildfire prediction model.

    Args:
        prediction_path (str): Path to the prediction result CSV (e.g., predicted_spread.csv).
        nodes_path (str): Path to the CSV containing all node information (lat, lon, etc.).
        output_filename (str): Name for the output HTML file.
    """
    print("--- Visualizing Prediction Result ---")
    
    # 1. Load data
    try:
        predictions_df = pd.read_csv(prediction_path)
        nodes_df = pd.read_csv(nodes_path)
        print("Prediction and node data loaded successfully.")
    except FileNotFoundError as e:
        print(f"[ERROR] Could not find a required data file: {e}")
        return

    # 2. Merge data to get coordinates for each predicted node
    # We need coordinates for both the target node and the source node
    
    # Merge for the target node
    merged_df = pd.merge(
        predictions_df, 
        nodes_df[['node_id', 'latitude', 'longitude']], 
        on='node_id', 
        how='left'
    )
    
    # Merge for the source node
    merged_df = pd.merge(
        merged_df,
        nodes_df[['node_id', 'latitude', 'longitude']].rename(columns={
            'node_id': 'source_node_id',
            'latitude': 'source_latitude',
            'longitude': 'source_longitude'
        }),
        on='source_node_id',
        how='left'
    )
    
    merged_df = merged_df.sort_values(by='ignition_time').reset_index(drop=True)
    print("Data merged and sorted.")

    # 3. Create Plotly graph
    fig = go.Figure()

    # Add edges (spread paths)
    for _, row in merged_df.iterrows():
        if pd.notna(row['source_node_id']):
            fig.add_trace(
                go.Scattermapbox(
                    mode="lines",
                    lon=[row['source_longitude'], row['longitude']],
                    lat=[row['source_latitude'], row['latitude']],
                    line=dict(width=2, color="red"),
                    name=f"Spread to {int(row['node_id'])}"
                )
            )

    # Add nodes (ignition points)
    fig.add_trace(
        go.Scattermapbox(
            mode="markers",
            lon=merged_df['longitude'],
            lat=merged_df['latitude'],
            marker=dict(
                size=10,
                color=merged_df['prediction_step_count'],
                colorscale="YlOrRd",
                colorbar_title="Prediction Step"
            ),
            text=[f"Node: {int(nid)}<br>Step: {int(step)}" for nid, step in zip(merged_df['node_id'], merged_df['prediction_step_count'])],
            hoverinfo="text",
            name="Ignition Points"
        )
    )

    # 4. Update layout to use a map background
    fig.update_layout(
        title="Wildfire Spread Prediction",
        mapbox_style="open-street-map",
        mapbox_center_lon=merged_df['longitude'].mean(),
        mapbox_center_lat=merged_df['latitude'].mean(),
        mapbox_zoom=10,
        margin={"r":0,"t":40,"l":0,"b":0},
        showlegend=False
    )
    
    # 5. Save and show graph
    fig.write_html(output_filename)
    print(f"Graph saved to {output_filename}")


def main():
    # 0. argparser
    parser = argparse.ArgumentParser(description='Wildfire Spread Graph Generator')
    parser.add_argument('--run_mode', type=str, default="original", choices=['original', 'prediction'], help="Mode to run: 'original' for historical data, 'prediction' for model output.")
    parser.add_argument('--data_number', type=int, default=1, help="[Original Mode] The data number to process.")
    # New arguments for prediction mode
    parser.add_argument('--prediction_path', type=str, default='prediction_data/predicted_spread.csv', help="[Prediction Mode] Path to the prediction result CSV.")
    parser.add_argument('--nodes_path', type=str, help="[Prediction Mode] Path to the master nodes data CSV.")
    
    args = parser.parse_args()

    if args.run_mode == 'prediction':
        if not args.nodes_path:
            print("[ERROR] For 'prediction' mode, --nodes_path is required.")
            return
        draw_prediction_graph(args.prediction_path, args.nodes_path)
        return # End execution after drawing prediction

    # --- Original Mode Execution ---
    print("--- Running in Original Mode ---")
    data_number = args.data_number
    
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