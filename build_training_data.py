import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# --- Project-specific Imports ---
# Assuming the script is run from the 'code' directory.
# Adjust sys.path if you run this from a different location.
import sys
sys.path.append(str(Path(__file__).parent.parent))

from data_use.data_path import load_data_path
from modules.data_load import load_and_enrich_data
from prediction.neighbor_definition import example_neighbor_finder
from prediction.prediction_utils import example_destination_calculator


def create_training_samples(fire_progression_df, all_nodes_df, neighbor_finder_func):
    """
    Creates positive and negative training samples from a chronological fire progression.

    Args:
        fire_progression_df (pd.DataFrame): DataFrame of actual fire locations, sorted by ignition_time.
        all_nodes_df (pd.DataFrame): DataFrame containing features for all nodes.
        neighbor_finder_func (function): Function to find neighbors of a given node.

    Returns:
        pd.DataFrame: A DataFrame of training samples with features and a label.
    """
    training_samples = []
    
    # Create a lookup for node features for faster access
    all_nodes_dict = all_nodes_df.set_index('node_id').to_dict('index')

    for i in range(len(fire_progression_df) - 1):
        # The source node is the current fire point in the progression
        source_node_id = fire_progression_df.iloc[i]['node_id']
        # The actual next fire point is our positive target
        positive_target_node_id = fire_progression_df.iloc[i + 1]['node_id']
        
        source_features = all_nodes_dict.get(source_node_id)
        if not source_features:
            continue

        # Find all neighbors of the source node
        neighbors_df = neighbor_finder_func(source_node_id, all_nodes_df, excluded_node_ids=set())
        
        for _, neighbor_row in neighbors_df.iterrows():
            neighbor_id = neighbor_row['node_id']
            neighbor_features = all_nodes_dict.get(neighbor_id)
            if not neighbor_features:
                continue

            # Create a combined feature row
            sample = {}
            for key, value in source_features.items():
                sample[f"source_{key}"] = value
            for key, value in neighbor_features.items():
                sample[f"target_{key}"] = value
            
            # --- Determine the label ---
            # If the neighbor is the actual next fire, it's a positive sample (1)
            # Otherwise, it's a negative sample (0)
            sample['actual_spread'] = 1 if neighbor_id == positive_target_node_id else 0
            
            training_samples.append(sample)
            
            # For the positive case, we can break early if we only want one positive sample per step
            if sample['actual_spread'] == 1:
                # We found the positive sample, no need to check other neighbors for this role
                # but we continue to generate negative samples from the other neighbors
                pass

    return pd.DataFrame(training_samples)


def main(args):
    """
    Main function to generate training data.
    """
    print(f"--- Building Training Data for data_number: {args.data_number} ---")
    
    try:
        # 1. Load the original, full dataset
        original_data_path = load_data_path(args.data_number)
        print(f"Loading original data from: {original_data_path}")
        # Use the same enrichment function for consistency
        all_nodes_df = load_and_enrich_data(
            csv_path=str(original_data_path),
            date_col='acq_date', time_col='acq_time',
            lat_col='latitude', lon_col='longitude'
        )
        if all_nodes_df is None or 'ignition_time' not in all_nodes_df.columns:
            raise ValueError("Failed to load or enrich data, or 'ignition_time' column is missing.")
        
        # Ensure node_id is present
        if 'node_id' not in all_nodes_df.columns:
            all_nodes_df['node_id'] = all_nodes_df.index

        # 2. Sort by time and select the first 10% for creating training ground truth
        all_nodes_df_sorted = all_nodes_df.sort_values(by='ignition_time').reset_index(drop=True)
        training_cutoff_index = int(len(all_nodes_df_sorted) * 0.1)
        fire_progression_for_training = all_nodes_df_sorted.iloc[:training_cutoff_index]
        
        print(f"Using first {len(fire_progression_for_training)} records (10%) to build training samples.")

        # 3. Define the neighbor finding function
        def _neighbor_finder_wrapper(node_id, nodes_df, excluded_node_ids):
            return example_neighbor_finder(node_id, nodes_df, excluded_node_ids, max_dist_config=args.neighbor_max_dist)

        # 4. Generate training samples
        print("Generating positive and negative training samples...")
        training_df = create_training_samples(fire_progression_for_training, all_nodes_df, _neighbor_finder_wrapper)

        if training_df.empty:
            print("[WARN] No training samples were generated. The input data might be too sparse or short.")
            return

        # 5. Save the processed training data
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"training_data_d{args.data_number}.csv"
        training_df.to_csv(output_path, index=False)
        
        print(f"\n--- Successfully created training data ---")
        print(f"Saved {len(training_df)} samples to: {output_path}")
        print(f"Positive samples (actual_spread=1): {training_df['actual_spread'].sum()}")
        print(f"Negative samples (actual_spread=0): {len(training_df) - training_df['actual_spread'].sum()}")

    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] Failed to process data: {e}")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build Training Data for Wildfire Spread Model")
    
    parser.add_argument("--data_number", type=int, required=True, help="The data number to process (e.g., 1, 2, 3...).")
    parser.add_argument("--output_dir", type=str, default="data_for_train", help="Directory to save the generated training data CSV file.")
    parser.add_argument("--neighbor_max_dist", type=float, default=0.1, help="Max distance in degrees to consider a node a neighbor.")
    
    args = parser.parse_args()
    main(args) 