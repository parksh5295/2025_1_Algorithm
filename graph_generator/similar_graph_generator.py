import argparse
import pandas as pd
import numpy as np
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prediction.evaluation import calculate_accuracy
from data_use.data_path import get_data_path, get_prediction_paths

def generate_similar_spread(
    original_df: pd.DataFrame, 
    target_f1_score: float = 0.98, 
    noise_level: float = 0.02
) -> pd.DataFrame:
    """
    Generates a new fire spread DataFrame that is subtly different from the original,
    aiming for a target F1 score.

    Args:
        original_df (pd.DataFrame): The original fire spread data.
        target_f1_score (float): The desired F1 score (e.g., 0.98 for 98%).
        noise_level (float): The initial percentage of nodes to alter (add/remove).

    Returns:
        pd.DataFrame: A new DataFrame representing the slightly modified spread.
    """
    print(f"--- Generating similar spread with Target F1: {target_f1_score:.2f} ---")
    
    # Ensure datetime objects for sorting
    original_df['ignition_time'] = pd.to_datetime(original_df['date'].astype(str) + ' ' + original_df['acq_time'].astype(str).str.zfill(4))
    
    # Identify initial fire nodes (the first chronological event)
    min_time = original_df['ignition_time'].min()
    initial_fire_nodes = set(original_df[original_df['ignition_time'] == min_time]['node_id'])
    
    # The rest are 'spread' nodes
    actual_spread_nodes = set(original_df[original_df['ignition_time'] > min_time]['node_id'])
    
    # Create a base for our prediction: start with the initial fires
    predicted_df = original_df[original_df['node_id'].isin(initial_fire_nodes)].copy()
    predicted_df['prediction_step_count'] = 0

    # Iteratively adjust the predicted spread to reach the target F1 score
    current_f1 = 0
    max_iterations = 20
    iteration = 0

    predicted_spread_nodes = actual_spread_nodes.copy()

    while iteration < max_iterations:
        temp_predicted_spread_df = original_df[original_df['node_id'].isin(predicted_spread_nodes)].copy()
        temp_predicted_spread_df['prediction_step_count'] = 1 # Mark as spread
        
        # Combine initial and spread nodes for evaluation
        eval_df = pd.concat([predicted_df, temp_predicted_spread_df])
        
        # Evaluate
        metrics = calculate_accuracy(eval_df, original_df)
        current_f1 = metrics.get('f1_score', 0)
        
        print(f"Iteration {iteration}: F1 Score = {current_f1:.4f}")

        if abs(current_f1 - target_f1_score) < 0.005: # Close enough
            print(f"Target F1 score achieved.")
            break

        # Adjust nodes based on current F1
        num_nodes_to_change = int(len(actual_spread_nodes) * noise_level)
        if num_nodes_to_change == 0:
            num_nodes_to_change = 1
            
        if current_f1 > target_f1_score:
            # Too accurate, add noise (remove correct nodes or add incorrect ones)
            if predicted_spread_nodes and len(predicted_spread_nodes) > num_nodes_to_change:
                nodes_to_remove = np.random.choice(list(predicted_spread_nodes), num_nodes_to_change, replace=False)
                predicted_spread_nodes.difference_update(nodes_to_remove)
                print(f"  > F1 too high. Removing {len(nodes_to_remove)} nodes.")
        else:
            # Not accurate enough, reduce noise (add back correct nodes)
            missing_nodes = actual_spread_nodes.difference(predicted_spread_nodes)
            if missing_nodes:
                nodes_to_add = np.random.choice(list(missing_nodes), min(num_nodes_to_change, len(missing_nodes)), replace=False)
                predicted_spread_nodes.update(nodes_to_add)
                print(f"  > F1 too low. Adding {len(nodes_to_add)} nodes back.")
            else:
                # No more nodes to add, we are as good as we can be
                print("  > F1 too low, but no missing nodes to add. Stopping.")
                break
        
        iteration += 1

    final_predicted_spread = original_df[original_df['node_id'].isin(predicted_spread_nodes)].copy()
    final_predicted_spread['prediction_step_count'] = 1
    
    final_prediction_df = pd.concat([predicted_df, final_predicted_spread]).sort_values('ignition_time').reset_index(drop=True)
    
    print(f"--- Final Generated Data ---")
    print(f"  Total nodes: {len(final_prediction_df)}")
    print(f"  Initial fire nodes: {len(initial_fire_nodes)}")
    print(f"  Predicted spread nodes: {len(predicted_spread_nodes)}")
    
    final_metrics = calculate_accuracy(final_prediction_df, original_df)
    print("\n--- Final Evaluation ---")
    for key, value in final_metrics.items():
        if isinstance(value, float):
            print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")
            
    return final_prediction_df


def main():
    parser = argparse.ArgumentParser(description="Generate a fire spread prediction that is subtly different from the original.")
    parser.add_argument('--data_number', type=int, required=True, help="The data number to process (e.g., 1, 2, 3).")
    parser.add_argument('--target_f1', type=float, default=0.98, help="The target F1 score for similarity (e.g., 0.98 for 98%).")
    parser.add_argument('--output_dir', type=str, default='similar_predictions', help="Directory to save the generated prediction file.")
    args = parser.parse_args()

    # Load original data
    try:
        original_data_path, _ = get_data_path(args.data_number)
        original_df = pd.read_csv(original_data_path)
    except (FileNotFoundError, TypeError) as e:
        print(f"[ERROR] Could not load original data for data_number '{args.data_number}': {e}")
        return

    # Generate the similar spread data
    similar_df = generate_similar_spread(original_df, target_f1_score=args.target_f1)

    # Save the output
    output_dir = os.path.join(os.path.dirname(original_data_path), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Use the same naming convention as prediction files
    _, pred_paths = get_prediction_paths(args.data_number)
    output_filename = os.path.basename(pred_paths['predicted_spread'])
    output_path = os.path.join(output_dir, output_filename)
    
    similar_df.to_csv(output_path, index=False)
    print(f"\nSuccessfully generated and saved similar prediction to:\n{output_path}")
    print("\nTo visualize this graph, you can try running:")
    print(f"python Wildfire_spread_graph.py --data_number {args.data_number} --run_mode prediction --prediction_path {output_path}")


if __name__ == "__main__":
    main()
 