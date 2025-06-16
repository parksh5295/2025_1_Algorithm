import argparse
import os
import pandas as pd
from utiles.estimate_time import add_datetime_column


def get_simgraph_path(data_number):
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    if data_number == 1:
        return os.path.join(base_dir, 'DL_FIRE_SV-C2_608350(250314-05)', 'simgraph_similar_fire_nrt_SV-C2_608350.csv')
    elif data_number == 2:
        return os.path.join(base_dir, 'DL_FIRE_SV-C2_608316(230402-12)', 'simgraph_similar_fire_archive_SV-C2_608316.csv')
    elif data_number == 3:
        return os.path.join(base_dir, 'DL_FIRE_J2V-C2_37482(2025-California)', 'simgraph_similar_fire_nrt_J2V-C2_37482.csv')
    elif data_number == 4:
        return os.path.join(base_dir, 'DL_FIRE_SV-C2_37483(2019-Australia)', 'simgraph_similar_fire_archive_SV-C2_37483.csv')
    elif data_number == 5:
        return os.path.join(base_dir, 'DL_FIRE_SV-C2_608318(220528-03)', 'simgraph_similar_fire_archive_SV-C2_608318.csv')
    elif data_number == 6:
        return os.path.join(base_dir, 'DL_FIRE_SV-C2_608319(220405-12)', 'simgraph_similar_fire_archive_SV-C2_608319.csv')
    elif data_number == 7:
        return os.path.join(base_dir, 'DL_FIRE_SV-C2_608320(220304-14)', 'simgraph_similar_fire_archive_SV-C2_608320.csv')
    else:
        raise ValueError("Invalid data number")

def main():
    parser = argparse.ArgumentParser(description="Create a networkx/matplotlib style wildfire spread GIF from simgraph data (no API enrich, 원본 스타일)")
    parser.add_argument('--data_number', type=int, required=True, help="The data number to process (e.g., 1, 2, 3)")
    args = parser.parse_args()

    # 1. Find the path to the simgraph file (data/ full recursive navigation)
    try:
        simgraph_path = get_simgraph_path(args.data_number)
    except Exception as e:
        print(f"[ERROR] {e}")
        return
    if not os.path.exists(simgraph_path):
        print(f"[ERROR] {simgraph_path} File not found.")
        return
    print(f"[INFO] Loading simgraph data from {simgraph_path}")

    # 2. Importing data
    df = pd.read_csv(simgraph_path)
    if 'date' not in df.columns:
        if 'acq_date' in df.columns and 'acq_time' in df.columns:
            df = add_datetime_column(df, 'acq_date', 'acq_time')
        else:
            print("[ERROR] No 'date' or ('acq_date'+'acq_time') columns found in simgraph data.")
            return

    # 3. Time grouping (in 15-minute increments)
    df['date_10min'] = pd.to_datetime(df['date']).dt.floor('15T')
    interval_groups = list(df.groupby('date_10min'))
    total_intervals = len(interval_groups)
    print(f"Found {total_intervals} unique 15-min intervals to process.")
    if total_intervals == 0:
        print("No data to process.")
        return

    # 4. Create cumulative frames and include network/edge/map snapshots
    from graph.build_graph import cluster_and_build_graph
    from graph.snapshot import draw_graph_snapshot
    from utiles.gen_gif import generate_gif_for_dataset
    cumulative_df = pd.DataFrame(columns=df.columns)
    for i, (interval, interval_df) in enumerate(interval_groups):
        sequence_id = i + 1
        cumulative_df = pd.concat([cumulative_df, interval_df], ignore_index=True)
        _processed_df, nodes_df, G = cluster_and_build_graph(cumulative_df.copy())
        draw_graph_snapshot(G, f"similar_sub_{args.data_number}", sequence_id)
        print(f"   Snapshot saved for sequence {sequence_id}")

    # 5. Create GIF
    print(f"Generating GIF for similar_sub_animation_{args.data_number}...")
    graph_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'graph')
    generate_gif_for_dataset(
        filenumber=f"similar_sub_{args.data_number}",
        frame_base_dir=graph_dir,
        output_gif_name=f"similar_sub_animation_{args.data_number}.gif",
        duration=0.25,
        frame_image_pattern='*.png'
    )
    print(f"--- Similar sub animation GIF created as 'similar_sub_animation_{args.data_number}.gif' ---")

if __name__ == "__main__":
    main() 