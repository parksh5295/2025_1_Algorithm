import pandas as pd
import os
import plotly.graph_objects as go
import imageio
from tqdm import tqdm

# Re-enable parallel processing import
import concurrent.futures 
import time
import imageio
from tqdm import tqdm

# Re-add imports here to ensure they are available in the thread's scope
from graph.build_graph import cluster_and_build_graph
from graph.snapshot import draw_graph_snapshot


# Imports for graph/utiles modules
try:
    from graph.snapshot import draw_graph_snapshot
    from utiles.gen_gif import generate_gif_for_dataset
    from graph.build_graph import cluster_and_build_graph
except ImportError as e:
    print(f"[ERROR] Failed to import required graph/utiles modules: {e}")

# --- Worker function for parallel processing (Uncommented) --- START ---
def _process_single_date(args):
    """Helper function to process graph building and snapshot saving for a single date."""
    date_arg, daily_df, filenumber, sequence_id = args # daily_df is the DataFrame in question

    # === DEBUGGING: Print daily_df info === START
    print(f"[WORKER_DEBUG] Processing for date_arg: {date_arg}, sequence: {sequence_id}")
    print(f"[WORKER_DEBUG] daily_df columns: {daily_df.columns.tolist()}")
    if not daily_df.empty:
        print(f"[WORKER_DEBUG] daily_df head:\n{daily_df.head()}")
    else:
        print(f"[WORKER_DEBUG] daily_df is empty.")
    # === DEBUGGING: Print daily_df info === END

    print(f"Processing date: {date_arg} (Sequence {sequence_id})...") # General logs
    try:
        # 1. Build graph
        # Ensure cluster_and_build_graph handles potential missing columns gracefully or we ensure they exist
        _processed_df, _nodes_df, G = cluster_and_build_graph(daily_df.copy())

        # 2. Save snapshot
        draw_graph_snapshot(G, filenumber, sequence_id)
        print(f"   Snapshot saved for sequence {sequence_id}")
        return True # Success flag
    except Exception as e:
        print(f"[ERROR] Failed processing date {date_arg}, sequence {sequence_id}: {e}")
        import traceback # For detailed error output
        print(traceback.format_exc()) # Print stack trace
        return False # Failure flag
# --- Worker function for parallel processing (Uncommented) --- END ---

# Main module functions (currently sequential processing)
# Add max_workers parameter back for potential future parallel use
def graph_module(df, data_number, latlon_bounds=None):
    """
    Generates a GIF animation of fire spread.
    Now supports two modes:
    1. Standard time-based animation (if 'comparison_type' column is absent).
    2. Comparison animation (if 'comparison_type' column is present).
    """
    # Check if we are in comparison mode
    is_comparison_mode = 'comparison_type' in df.columns

    # --- Directory and Path Setup ---
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)
    gif_dir = os.path.join(project_root, 'data', 'graph', 'gif')
    frame_dir_base = os.path.join(project_root, 'data', 'graph', 'frame')
    
    os.makedirs(gif_dir, exist_ok=True)

    # --- Data Preparation ---
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    if is_comparison_mode:
        print("--- Running Graph Module in COMPARISON mode ---")
        output_filename_base = f"comparison_{data_number}"
        # In comparison mode, all points are plotted in a single frame.
        # We create a dummy 'time_group' to fit the loop structure.
        df['time_group'] = 0 
        time_groups = [df['date'].min()] # A single group for one frame
    else:
        print("--- Running Graph Module in STANDARD (time-series) mode ---")
        output_filename_base = f"animation_{data_number}"
        df['time_group'] = df['date'].dt.floor('1H')
        time_groups = sorted(df['time_group'].unique())

    frame_dir = os.path.join(frame_dir_base, output_filename_base)
    os.makedirs(frame_dir, exist_ok=True)
    
    # --- Map Boundary Calculation ---
    if latlon_bounds:
        lat_min, lat_max, lon_min, lon_max = latlon_bounds
    else:
        # Fallback if no bounds are provided
        lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
        lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
        margin = 0.1
        lat_margin = (lat_max - lat_min) * margin
        lon_margin = (lon_max - lon_min) * margin
        lat_min, lat_max = lat_min - lat_margin, lat_max + lat_margin
        lon_min, lon_max = lon_min - lon_margin, lon_max + lon_margin

    # --- Frame Generation ---
    frames = []
    print(f"Generating {len(time_groups)} frame(s) for '{output_filename_base}.gif'...")

    cumulative_df = pd.DataFrame()

    for i, time_group in enumerate(tqdm(time_groups, desc="Processing Frames")):
        if is_comparison_mode:
            # For comparison, plot all points at once
            frame_df = df
        else:
            # For standard animation, accumulate points over time
            group_df = df[df['time_group'] == time_group]
            cumulative_df = pd.concat([cumulative_df, group_df]).drop_duplicates(subset=['latitude', 'longitude'])
            frame_df = cumulative_df

        fig = go.Figure()

        # Add fire points
        if is_comparison_mode:
            # Color points based on their comparison type
            colors = {
                'Overlap': 'purple',
                'Actual Only': 'red',
                'Predicted Only': 'blue'
            }
            for point_type, color in colors.items():
                plot_df = frame_df[frame_df['comparison_type'] == point_type]
                if not plot_df.empty:
                    fig.add_trace(go.Scattermapbox(
                        lat=plot_df['latitude'],
                        lon=plot_df['longitude'],
                        mode='markers',
                        marker=go.scattermapbox.Marker(
                            size=8,
                            color=color,
                            opacity=0.7
                        ),
                        name=point_type,
                        hoverinfo='text',
                        text=[f"Type: {ptype}<br>Lat: {lat:.4f}<br>Lon: {lon:.4f}" for ptype, lat, lon in zip(plot_df['comparison_type'], plot_df['latitude'], plot_df['longitude'])]
                    ))
        else:
            # Standard time-series coloring
            fig.add_trace(go.Scattermapbox(
                lat=frame_df['latitude'],
                lon=frame_df['longitude'],
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=9,
                    color=frame_df['confidence'].map({'h': 'red', 'n': 'orange'}).fillna('gray'),
                    opacity=0.8
                ),
                hoverinfo='none'
            ))

        # Configure map layout
        fig.update_layout(
            mapbox_style="carto-darkmatter",
            mapbox_center_lat= (lat_max + lat_min) / 2,
            mapbox_center_lon= (lon_max + lon_min) / 2,
            mapbox_zoom=8,
            mapbox_bounds={"west": lon_min, "east": lon_max, "south": lat_min, "north": lat_max},
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            showlegend=is_comparison_mode # Show legend only for comparison GIFs
        )

        frame_path = os.path.join(frame_dir, f"frame_{i:04d}.png")
        fig.write_image(frame_path, width=800, height=700)
        frames.append(frame_path)

    # --- GIF Creation ---
    gif_path = os.path.join(gif_dir, f"{output_filename_base}.gif")
    print(f"Creating GIF at {gif_path}...")
    with imageio.get_writer(gif_path, mode='I', duration=500 if is_comparison_mode else 250, loop=0 if is_comparison_mode else 1) as writer:
        for frame_path in tqdm(frames, desc="Assembling GIF"):
            image = imageio.imread(frame_path)
            writer.append_data(image)

    # --- Cleanup ---
    print("Cleaning up temporary frame files...")
    for frame_path in frames:
        os.remove(frame_path)
    if not os.listdir(frame_dir):
        os.rmdir(frame_dir)

    print(f"✅ GIF generation complete: {gif_path}")

# --- Example calling method (for reference) ---
# if __name__ == '__main__':
#     # 1. Load data (example: use data_load.py in modules folder)
#     #    Prepare df for filenumber
#     #    df = load_and_enrich_data(...)
#
#     # 2. Target dataset number to process
#     target_filenumber = 5
#
#     # 3. Call main function
#     graph_module(df, target_filenumber) # Example: Use up to 4 processes