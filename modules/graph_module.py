import pandas as pd
import os
import shutil
# Re-enable parallel processing import
import concurrent.futures 
import time


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
    date, daily_df, filenumber, sequence_id = args
    print(f"Processing date: {date} (Sequence {sequence_id})...")
    try:
        # Assuming these imports are safe within the worker process/thread now
        # (Can remove internal imports if top-level ones work reliably)
        # from graph.build_graph import cluster_and_build_graph
        # from graph.snapshot import draw_graph_snapshot

        # 1. Build graph
        _processed_df, _nodes_df, G = cluster_and_build_graph(daily_df.copy())

        # 2. Save snapshot
        draw_graph_snapshot(G, filenumber, sequence_id)
        print(f"   Snapshot saved for sequence {sequence_id}")
        return True # Success flag
    except Exception as e:
        print(f"[ERROR] Failed processing date {date}, sequence {sequence_id}: {e}")
        return False # Failure flag
# --- Worker function for parallel processing (Uncommented) --- END ---

# Main module functions (currently sequential processing)
# Add max_workers parameter back for potential future parallel use
def graph_module(df, filenumber, max_workers=None): # max_workers is currently unused
    """
    Process the given DataFrame, generate snapshots for each date in parallel, 
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

    daily_groups = list(df.groupby(df['date'].dt.date))
    total_days = len(daily_groups)
    print(f"Found {total_days} unique dates to process.")

    if total_days == 0:
        print("No data to process.")
        return

    # --- Sequential Processing Loop (Commented out) --- START ---
    # successful_snapshots = 0
    # print(f"\nStarting sequential snapshot generation...")
    # snapshot_start_time = time.time()
    # 
    # for i, (date, daily_df) in enumerate(daily_groups):
    #     sequence_id = i + 1
    #     print(f"Processing date: {date} (Sequence {sequence_id}/{total_days})...")
    #     try:
    #         _processed_df, _nodes_df, G = cluster_and_build_graph(daily_df.copy())
    #         draw_graph_snapshot(G, filenumber, sequence_id)
    #         print(f"   Snapshot saved for sequence {sequence_id}")
    #         successful_snapshots += 1
    #     except Exception as e:
    #         print(f"[ERROR] Failed processing date {date}, sequence {sequence_id}: {e}")
    # 
    # snapshot_end_time = time.time()
    # print(f"Finished snapshot generation in {snapshot_end_time - snapshot_start_time:.2f} seconds.")
    # print(f"Successfully generated {successful_snapshots} out of {total_days} snapshots.")
    # --- Sequential Processing Loop (Commented out) --- END ---

    # --- Parallel Processing Block (Uncommented) --- START ---
    tasks = []
    for i, (date, daily_df) in enumerate(daily_groups):
        sequence_id = i + 1
        tasks.append((date, daily_df, filenumber, sequence_id))
    
    successful_snapshots = 0
    # Use ThreadPoolExecutor
    print(f"\nStarting parallel snapshot generation (using up to {max_workers or 'default'} threads)..." )
    snapshot_start_time = time.time()
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_process_single_date, tasks))
    
    successful_snapshots = sum(results)
    snapshot_end_time = time.time()
    print(f"Finished snapshot generation in {snapshot_end_time - snapshot_start_time:.2f} seconds.")
    print(f"Successfully generated {successful_snapshots} out of {total_days} snapshots.")
    # --- Parallel Processing Block (Uncommented) --- END ---

    # GIF generation (after all snapshots are saved)
    if successful_snapshots > 0:
        print(f"\nGenerating GIF for dataset {filenumber}...")
        gif_start_time = time.time()
        try:
            # Call generate_gif_for_dataset function
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_script_dir)
            graph_folder_path = os.path.join(project_root, 'graph')
            generate_gif_for_dataset(filenumber=filenumber, frame_base_dir=graph_folder_path)
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