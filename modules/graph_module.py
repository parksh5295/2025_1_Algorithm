import pandas as pd
import os
import shutil
# Re-enable parallel processing import
import concurrent.futures 
import time

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
def graph_module(df, filenumber, latlon_bounds=None, max_workers=None):
    """
    Process the given DataFrame, generate snapshots for each 30-minute interval in parallel, 
    and then create a GIF.
    """
    print(f"\n--- Starting Parallel Graph Processing for dataset filenumber: {filenumber} ---")
    start_time = time.time()

    # Prepare date column conversion and grouping
    try:
        df['date'] = pd.to_datetime(df['date'])
    except KeyError:
        print("[ERROR] 'date' column not found.")
        return
    except Exception as e:
        print(f"[ERROR] Failed converting 'date' column: {e}")
        return

    # Group by 10-minute intervals (changed from 30T)
    df['date_10min'] = df['date'].dt.floor('15T')
    interval_groups = list(df.groupby('date_10min'))
    total_intervals = len(interval_groups)
    print(f"Found {total_intervals} unique 15-min intervals to process.")

    if total_intervals == 0:
        print("No data to process.")
        return

    # Prepare cumulative DataFrame
    cumulative_df = pd.DataFrame(columns=df.columns)
    tasks = []
    for i, (interval, interval_df) in enumerate(interval_groups):
        sequence_id = i + 1
        cumulative_df = pd.concat([cumulative_df, interval_df], ignore_index=True)
        # Use accumulated data to create snapshots
        tasks.append((interval, cumulative_df.copy(), filenumber, sequence_id, latlon_bounds))

    successful_snapshots = 0
    print(f"\nStarting sequential snapshot generation for 10-min intervals...")
    snapshot_start_time = time.time()
    
    # Initializing a Stacked Graph Object (Outside of a Loop)
    import networkx as nx
    G_cumu = nx.DiGraph()
    # # Store nodes from the previous snapshot to identify new nodes and existing nodes
    # # previous_nodes_set = set() # Initialize an empty set for the very first snapshot. Deferred for now.
    # # Declare a cumulative edge/node set -> Remove
    # cumulative_edges = set()
    # cumulative_nodes = set()

    for t_idx, t in enumerate(tasks): # Enumerate to access task index
        interval, cumu_df, filenumber, sequence_id, current_latlon_bounds = t # Renamed for clarity
        try:
            _processed_df, nodes_df, G = cluster_and_build_graph(cumu_df.copy()) # Get nodes_df as well

            # --- START: New logic to connect isolated nodes ---
            if G.number_of_nodes() > 1: # Only proceed if there's more than one node
                isolated_nodes = [node for node in G.nodes() if G.degree(node) == 0]
                
                # Ensure nodes_df is not empty and has the required columns
                if not nodes_df.empty and all(col in nodes_df.columns for col in ['cluster_id', 'center_longitude', 'center_latitude']):
                    # Create a dictionary for node positions from nodes_df for distance calculation
                    # Filter nodes_df to only include nodes present in the current graph G
                    relevant_nodes_df = nodes_df[nodes_df['cluster_id'].isin(G.nodes())]
                    node_positions = {
                        row['cluster_id']: (row['center_longitude'], row['center_latitude'])
                        for index, row in relevant_nodes_df.iterrows()
                    }

                    if isolated_nodes and len(G.nodes()) > len(isolated_nodes): # Only if there are non-isolated nodes to connect to
                        non_isolated_nodes = [node for node in G.nodes() if G.degree(node) > 0]
                        
                        for isolated_node_id in isolated_nodes:
                            if isolated_node_id not in node_positions:
                                print(f"[WARN] Position for isolated node {isolated_node_id} not found in node_positions dict. Skipping connection for this node in sequence {sequence_id}.")
                                continue

                            iso_pos = node_positions[isolated_node_id]
                            min_dist_sq = float('inf')
                            closest_neighbor_id = None

                            for neighbor_node_id in non_isolated_nodes:
                                if neighbor_node_id not in node_positions or neighbor_node_id == isolated_node_id:
                                    # This might happen if a non-isolated node in G is not in nodes_df (should not ideally occur)
                                    # or trying to connect to self (which is already handled by G.degree(node) > 0 for non_isolated_nodes)
                                    continue
                                
                                neigh_pos = node_positions[neighbor_node_id]
                                # Simple Euclidean distance (squared, for comparison)
                                # For geographic data, Haversine distance would be more accurate, but this is for visual linking.
                                dist_sq = (iso_pos[0] - neigh_pos[0])**2 + (iso_pos[1] - neigh_pos[1])**2 
                                
                                if dist_sq < min_dist_sq:
                                    min_dist_sq = dist_sq
                                    closest_neighbor_id = neighbor_node_id
                            
                            if closest_neighbor_id is not None:
                                # Connect isolated nodes first in the current snapshot's G
                                G.add_edge(isolated_node_id, closest_neighbor_id, type='inferred_connection', weight=0.1) 
                                print(f"[INFO] Added inferred edge between isolated {isolated_node_id} and {closest_neighbor_id} (dist_sq: {min_dist_sq:.4f}) for sequence {sequence_id}")
                            else:
                                print(f"[INFO] No non-isolated neighbor found to connect isolated node {isolated_node_id} in sequence {sequence_id}")
                    # else:
                        # if isolated_nodes:
                            # print(f"[INFO] All nodes are isolated in sequence {sequence_id}, no connections to make.")
                # else:
                    # if not nodes_df.empty:
                         # print(f"[WARN] nodes_df is missing required columns ('cluster_id', 'center_longitude', 'center_latitude') for sequence {sequence_id}. Skipping node connection logic.")
                    # else:
                        # print(f"[INFO] nodes_df is empty for sequence {sequence_id}. Skipping node connection logic.")
            # --- END: New logic to connect isolated nodes ---

            # === Update the cumulative graph ===
            # Add/update the edges and nodes of the current G to the cumulative graph G_cumu
            G_cumu.add_nodes_from(G.nodes(data=True))
            G_cumu.add_edges_from(G.edges(data=True))

            # Create a snapshot from the cumulative graph G_cumu
            # Pass current_latlon_bounds to draw_graph_snapshot
            draw_graph_snapshot(G_cumu, filenumber, sequence_id, latlon_bounds=current_latlon_bounds)
            print(f"   Snapshot saved for sequence {sequence_id}")
            successful_snapshots += 1
        except Exception as e:
            print(f"[ERROR] Failed processing interval {interval}, sequence {sequence_id}: {e}")
            # Print detailed error information when an error occurs
            import traceback
            print(traceback.format_exc())
    snapshot_end_time = time.time()
    print(f"Finished snapshot generation in {snapshot_end_time - snapshot_start_time:.2f} seconds.")
    print(f"Successfully generated {successful_snapshots} out of {total_intervals} snapshots.")

    # GIF generation (after all snapshots are saved)
    if successful_snapshots > 0:
        print(f"\nGenerating GIF for dataset {filenumber}...")
        gif_start_time = time.time()
        try:
            # Determine project root and the target base directory for frames/GIFs
            current_script_dir = os.path.dirname(os.path.abspath(__file__)) # .../code/modules
            code_dir = os.path.dirname(current_script_dir) # .../code
            project_root = os.path.dirname(code_dir) # .../
            
            # frame_base_dir should now point to project_root/data/graph
            # This is where gen_gif.py will look for a 'frame' subfolder 
            # and create a 'gif' subfolder.
            gif_frames_base_dir = os.path.join(project_root, 'data', 'graph')

            generate_gif_for_dataset(filenumber=filenumber, frame_base_dir=gif_frames_base_dir)
            gif_end_time = time.time()
            print(f"GIF generation complete in {gif_end_time - gif_start_time:.2f} seconds.")
        except Exception as e:
            print(f"[ERROR] Failed to generate GIF for dataset {filenumber}: {e}")
    else:
        print(f"No successful snapshots were generated for dataset {filenumber}, skipping GIF creation.")

    end_time = time.time()
    print(f"\n--- Finished Parallel Graph Processing for dataset {filenumber} in {end_time - start_time:.2f} seconds ---")

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