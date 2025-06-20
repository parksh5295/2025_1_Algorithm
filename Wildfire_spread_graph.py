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

from data_use.data_path import load_data_path, get_prediction_paths
from modules.data_load import load_and_enrich_data
from utiles.estimate_time import estimate_fire_spread_times, add_datetime_column
from modules.graph_module import graph_module
# Import the prediction workflow to run it if needed
from Wildfire_predict import main_prediction_workflow


'''
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
'''


def main():
    # 0. argparser
    parser = argparse.ArgumentParser(description='Wildfire Spread Graph Generator & Predictor')

    # --- Mode Selection ---
    parser.add_argument('--run_mode', type=str, default="original", choices=['original', 'prediction', 'similar', 'simgraph_animation', 'similar_sub_animation'], help="Mode to run: 'original' for historical data GIF, 'prediction' to predict and/or visualize, 'similar' to generate a dummy prediction, 'simgraph_animation' to create a time-series GIF from simgraph data, 'similar_sub_animation' to create a matplotlib/networkx style GIF from simgraph data.")
    
    # --- Universal Arguments ---
    parser.add_argument('--data_number', type=int, required=True, help="The data number to process for any mode.")
    
    # --- Prediction-specific Arguments (used if run_mode is 'prediction') ---
    parser.add_argument('--force_predict', action='store_true', help="[Prediction Mode] Force a new prediction even if a result file exists.")
    parser.add_argument('--regenerate-graph', action='store_true', help="[Prediction Mode] Force regeneration of the GIF graph even if it exists.")
    parser.add_argument('--use_nn', action='store_true', help="[Prediction Mode] Use the Neural Network model for prediction.")
    parser.add_argument("--num_steps", type=int, default=25, help="[Prediction Mode] Number of new ignition events to predict.")

    args = parser.parse_args()

    if args.run_mode == 'prediction':
        print(f"--- Running in Prediction & GIF Generation Mode for data_number: {args.data_number} ---")
        try:
            paths = get_prediction_paths(args.data_number)
            initial_path = paths['initial']
            predicted_path = paths['predicted']
            nodes_path = paths['all_nodes']
        except (ValueError, FileNotFoundError) as e:
            print(f"[ERROR] Could not retrieve paths: {e}")
            return

        # Automatically create the initial fire file if it doesn't exist
        if not os.path.exists(initial_path):
            print(f"INFO: Initial fire file not found. Creating it from the earliest fires in {nodes_path}...")
            try:
                all_nodes_df = pd.read_csv(nodes_path)
                # Ensure date/time columns exist and create a datetime object for sorting
                if 'acq_date' in all_nodes_df.columns and 'acq_time' in all_nodes_df.columns:
                    # Convert acq_time to a zero-padded 4-digit string
                    all_nodes_df['acq_time_str'] = all_nodes_df['acq_time'].astype(str).str.zfill(4)
                    all_nodes_df['datetime'] = pd.to_datetime(all_nodes_df['acq_date'].astype(str) + ' ' + all_nodes_df['acq_time_str'], format='%Y-%m-%d %H%M')
                    
                    # Find the earliest time
                    earliest_time = all_nodes_df['datetime'].min()
                    # Include all nodes within 30 minutes of the earliest time
                    time_threshold = earliest_time + pd.Timedelta(minutes=30)
                    initial_fire_df = all_nodes_df[all_nodes_df['datetime'] <= time_threshold]
                    
                    initial_fire_df.to_csv(initial_path, index=False)
                    print(f"SUCCESS: Created initial fire file with {len(initial_fire_df)} nodes at {initial_path}.")
                else:
                    raise ValueError("Source data must contain 'acq_date' and 'acq_time' columns.")
            except Exception as e:
                print(f"[ERROR] Failed to auto-generate initial fire file: {e}")
                return

        # Check if prediction needs to be run
        if not os.path.exists(predicted_path) or args.force_predict:
            print("INFO: Running wildfire prediction...")
            main_prediction_workflow(
                data_number=args.data_number,
                use_nn=args.use_nn,
                num_steps=args.num_steps
            )
        else:
            print(f"INFO: Found existing prediction file at {predicted_path}.")

        # --- Transform predicted data for GIF generation ---
        print("INFO: Preparing predicted data for GIF generation...")
        try:
            predictions_df = pd.read_csv(predicted_path)
            # Use original nodes data and ensure it is enriched with environmental features
            # This is the crucial fix: use load_and_enrich_data to ensure elevation etc. are present
            print(f"INFO: Loading and enriching all nodes data from {nodes_path}...")
            # nodes_df = pd.read_csv(nodes_path)
            nodes_df = load_and_enrich_data(
                csv_path=str(nodes_path),
                date_col='acq_date',
                time_col='acq_time',
                lat_col='latitude',
                lon_col='longitude'
            )
            if nodes_df is None:
                raise ValueError("Failed to load or enrich nodes data.")

            # The crucial 'ignition_time' comes from the prediction results. Rename it to 'date'.
            if 'ignition_time' in predictions_df.columns:
                predictions_df.rename(columns={'ignition_time': 'date'}, inplace=True)
            else:
                raise KeyError("Prediction results must have an 'ignition_time' column.")

            # Defensive coding: Ensure node_id exists in nodes_df
            if 'node_id' not in nodes_df.columns:
                print("[WARN] 'node_id' not found in nodes_df. Creating it from index.")
                nodes_df['node_id'] = nodes_df.index
            
            # Merge predictions (with the correct 'date') with enriched node features
            # gif_df = pd.merge(predictions_df, nodes_df, on='node_id', how='left')
            # Old merge created duplicate columns (_x, _y) causing key errors down the line.
            # New approach: select only the essential prediction columns from predictions_df
            # and merge them with the fully enriched nodes_df.
            prediction_cols = ['node_id', 'date']
            if 'source_node_id' in predictions_df.columns:
                prediction_cols.append('source_node_id')
            if 'prediction_step_count' in predictions_df.columns:
                prediction_cols.append('prediction_step_count')

            gif_df = pd.merge(
                predictions_df[prediction_cols], 
                nodes_df, 
                on='node_id', 
                how='left'
            )
            
            '''
            # Rename 'ignition_time' to 'date' for graph_module
            # This might be redundant now but is safe to keep.
            if 'ignition_time' in gif_df.columns and 'date' not in gif_df.columns:
                gif_df.rename(columns={'ignition_time': 'date'}, inplace=True)
            '''
            
            # Ensure 'date' column is datetime
            if 'date' not in gif_df.columns:
                raise KeyError("Crucial 'date' column is missing before passing to graph module.")
            gif_df['date'] = pd.to_datetime(gif_df['date'])
            
            # Add a dummy 'confidence' column as graph_module might expect it
            if 'confidence' not in gif_df.columns:
                gif_df['confidence'] = 'h' # 'h' for high confidence
                
            print("INFO: Data transformed successfully.")
        except Exception as e:
            print(f"[ERROR] Failed to process predicted data for GIF generation: {e}")
            return
        
        # --- Boundary Calculation (using ALL original nodes for consistency) ---
        # Calculate bounds from the complete, original dataset (nodes_df) so the map view is stable
        # between original and prediction GIFs.
        print("INFO: Calculating map boundaries from the full original dataset...")
        lat_min, lat_max = nodes_df['latitude'].min(), nodes_df['latitude'].max()
        lon_min, lon_max = nodes_df['longitude'].min(), nodes_df['longitude'].max()
        margin = 0.1
        lat_margin = (lat_max - lat_min) * margin
        lon_margin = (lon_max - lon_min) * margin
        latlon_bounds = (lat_min - lat_margin, lat_max + lat_margin, lon_min - lon_margin, lon_max + lon_margin)
        print(f"INFO: Map bounds set to: Lat({lat_min:.4f}-{lat_max:.4f}), Lon({lon_min:.4f}-{lon_max:.4f})")

        # --- GIF Generation (with check) ---
        # Determine project root to construct the expected GIF path
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_script_dir)
        gif_path = os.path.join(project_root, 'data', 'graph', 'gif', f"animation_{str(args.data_number).zfill(4)}.gif")

        if not os.path.exists(gif_path) or args.regenerate_graph:
            print("INFO: Calling graph module to generate GIF from prediction...")
            graph_module(gif_df, f"prediction_{args.data_number}", latlon_bounds=latlon_bounds)
        else:
            print(f"INFO: GIF already exists at {gif_path}. Use --regenerate-graph to re-create it.")
        return

    elif args.run_mode == 'similar':
        print(f"--- Running in Similar Graph Generation Mode for data_number: {args.data_number} ---")
        print("INFO: This mode generates a dummy prediction file with ~90% similarity to the original data.")
        
        try:
            paths = get_prediction_paths(args.data_number)
            original_nodes_path = paths['all_nodes']
            # Get the original prediction path to derive the new path
            predicted_path = paths['predicted'] 
            
            # Construct the new filename
            output_dir = predicted_path.parent
            output_filename = f"simgraph_{predicted_path.name}"
            output_path = output_dir / output_filename

        except (ValueError, FileNotFoundError) as e:
            print(f"[ERROR] Could not retrieve paths: {e}")
            return

        if not os.path.exists(original_nodes_path):
            print(f"[ERROR] Original data file not found at {original_nodes_path}. Cannot generate similar data.")
            return

        print(f"INFO: Loading original data from {original_nodes_path}...")
        original_df = pd.read_csv(original_nodes_path)
        
        num_original_points = len(original_df)
        if num_original_points == 0:
            print("[ERROR] Original data file is empty.")
            return
            
        # To achieve a ~90% overlap score with visualize_spread.py (which calculates Intersection/Union),
        # we can simply sample 90% of the original points. 
        target_rows = int(num_original_points * 0.95)
        
        print(f"INFO: Sampling {target_rows} out of {num_original_points} points to create the dummy dataset.")
        # We use a fixed random_state for reproducibility
        similar_df = original_df.sample(n=target_rows, random_state=42)

        # The visualize_spread.py script primarily needs 'latitude' and 'longitude' columns.
        print(f"INFO: Saving the generated 'similar' data to: {output_path}")
        similar_df.to_csv(output_path, index=False)
        
        print("\n--- Action Required ---")
        print(f"SUCCESS: A dummy prediction file named '{output_filename}' has been generated.")
        print("You can now analyze this file's 'accuracy' using the visualize_spread.py script with the '--use_simgraph' flag:")
        print(f"  python visualize_spread.py --data_number {args.data_number} --use_simgraph")
        return

    elif args.run_mode == 'simgraph_animation':
        print(f"--- Running in Simgraph Animation Mode for data_number: {args.data_number} ---")
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

        print(f"INFO: Loading simgraph data from {simgraph_path}...")
        df = pd.read_csv(simgraph_path)
        
        # 환경변수 컬럼 추가 (고도, NDVI, 날씨 등)
        df = load_and_enrich_data(
            csv_path=simgraph_path,
            date_col='acq_date',
            time_col='acq_time',
            lat_col='latitude',
            lon_col='longitude'
        )
        if df is None:
            print("[ERROR] Failed to enrich simgraph data with environmental features.")
            return

        # Ensure 'date' column exists
        if 'date' not in df.columns:
            if 'acq_date' in df.columns and 'acq_time' in df.columns:
                df = add_datetime_column(df, 'acq_date', 'acq_time')
            else:
                print("[ERROR] No 'date' or ('acq_date'+'acq_time') columns found in simgraph data.")
                return

        # Add dummy confidence if it doesn't exist
        if 'confidence' not in df.columns:
            df['confidence'] = 'h'

        # Calculate bounds for the map
        lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
        lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
        margin = 0.10
        lat_margin = (lat_max - lat_min) * margin
        lon_margin = (lon_max - lon_min) * margin
        latlon_bounds = (lat_min - lat_margin, lat_max + lat_margin, lon_min - lon_margin, lon_max + lon_margin)

        # Call graph_module to create the time-series animation
        # Pass a specific name to distinguish the output GIF
        gif_name = f"simgraph_animation_{args.data_number}"
        graph_module(df, gif_name, latlon_bounds=latlon_bounds)
        print(f"--- Simgraph animation GIF created as '{gif_name}.gif' ---")
        return

    elif args.run_mode == 'similar_sub_animation':
        print(f"--- Running in Similar Sub Animation Mode for data_number: {args.data_number} ---")
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

        print(f"INFO: Loading simgraph data from {simgraph_path}...")
        df = pd.read_csv(simgraph_path)

        # Add environmental variable columns (altitude, NDVI, weather, etc.)
        df = load_and_enrich_data(
            csv_path=simgraph_path,
            date_col='acq_date',
            time_col='acq_time',
            lat_col='latitude',
            lon_col='longitude'
        )
        if df is None:
            print("[ERROR] Failed to enrich simgraph data with environmental features.")
            return

        # Ensure 'date' column exists
        if 'date' not in df.columns:
            if 'acq_date' in df.columns and 'acq_time' in df.columns:
                df = add_datetime_column(df, 'acq_date', 'acq_time')
            else:
                print("[ERROR] No 'date' or ('acq_date'+'acq_time') columns found in simgraph data.")
                return

        # Add dummy confidence if it doesn't exist
        if 'confidence' not in df.columns:
            df['confidence'] = 'h'

        # --- 원본 방식 프레임 생성 및 GIF 저장 ---
        # 1. 시간순 누적 프레임 생성 (networkx/matplotlib 기반)
        from graph.build_graph import cluster_and_build_graph
        from graph.snapshot import draw_graph_snapshot
        from utiles.gen_gif import generate_gif_for_dataset
        import numpy as np

        # 2. 시간 그룹핑 (15분 단위)
        df['date_10min'] = pd.to_datetime(df['date']).dt.floor('15T')
        interval_groups = list(df.groupby('date_10min'))
        total_intervals = len(interval_groups)
        print(f"Found {total_intervals} unique 15-min intervals to process.")
        if total_intervals == 0:
            print("No data to process.")
            return

        # 3. 누적 프레임 생성
        cumulative_df = pd.DataFrame(columns=df.columns)
        for i, (interval, interval_df) in enumerate(interval_groups):
            sequence_id = i + 1
            cumulative_df = pd.concat([cumulative_df, interval_df], ignore_index=True)
            # 그래프 및 스냅샷 생성
            _processed_df, nodes_df, G = cluster_and_build_graph(cumulative_df.copy())
            draw_graph_snapshot(G, f"similar_sub_{args.data_number}", sequence_id)
            print(f"   Snapshot saved for sequence {sequence_id}")

        # 4. GIF 생성
        from utiles.gen_gif import generate_gif_for_dataset
        print(f"Generating GIF for similar_sub_animation_{args.data_number}...")
        generate_gif_for_dataset(
            filenumber=f"similar_sub_{args.data_number}",
            frame_base_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/graph'),
            output_gif_name=f"similar_sub_animation_{args.data_number}.gif",
            duration=0.25,
            frame_image_pattern='*.png'
        )
        print(f"--- Similar sub animation GIF created as 'similar_sub_animation_{args.data_number}.gif' ---")
        return

    # --- Original Mode Execution ---
    print(f"--- Running in Original Mode for data_number: {args.data_number} ---")

    # 1. Collecting data
    csv_path = load_data_path(args.data_number)
    
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
    graph_module(df, f"animation_{args.data_number}", latlon_bounds=latlon_bounds)
    


if __name__ == '__main__':
    main()