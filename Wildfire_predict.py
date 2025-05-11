import argparse
import os
import sys
import pandas as pd

# --- Path Setup ---
# Current script is Wildfire_predict.py (in project_root)
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_script_path) # project_root is the directory of this script

# prediction_module_dir is now directly relative to project_root
prediction_module_dir = os.path.join(project_root, 'prediction')
if prediction_module_dir not in sys.path:
    sys.path.append(prediction_module_dir)

# --- Module Imports ---
from prediction_utils import calculate_bearing, calculate_spread_weight, example_destination_calculator
from prediction_core import predict_wildfire_spread
from evaluation import calculate_accuracy
from neighbor_definition import example_neighbor_finder

# --- Configuration Constants ---
COEFFICIENTS = {
    'c1': 0.1,  # For (Destination) term
    'c2': 0.05, # For Wind term
    'c3': 0.1,  # For Temperature difference
    'c4': 0.1,  # For Humidity difference
    'c5': 0.1,  # For Rainfall difference (Note: sign in formula might need review)
    'c6': 0.2,  # For NDVI sum
    'c7': 0.05  # For Elevation sum (Note: sign in formula might need review)
}

# --- Main Workflow Orchestration ---
def main_prediction_workflow(args):
    print("--- Starting Wildfire Spread Prediction Workflow ---")
    current_c_coeffs = COEFFICIENTS.copy()
    if args.c_coeffs:
        try:
            coeffs_override = dict(item.split(':') for item in args.c_coeffs.split(','))
            for key, value in coeffs_override.items():
                if key in current_c_coeffs: current_c_coeffs[key] = float(value)
                else: print(f"[WARN] Invalid coefficient '{key}' in --c_coeffs.")
            print(f"Using C coefficients (overridden or default): {current_c_coeffs}")
        except Exception as e:
            print(f"[ERROR] Parsing --c_coeffs: {e}. Using defaults.")
            print(f"Using C coefficients (default): {current_c_coeffs}")
    else:
        print(f"Using C coefficients (default): {current_c_coeffs}")

    # Output CSV path will be in project_root/prediction_data/
    output_dir = os.path.join(project_root, 'prediction_data')
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, args.output_csv_name)

    # Adjust input paths to be relative to project_root if they are not absolute
    # This assumes args paths might be given relative to where the script is called from,
    # or specifically relative to project_root/data/
    def resolve_data_path(arg_path, expected_subdir_in_data=None):
        if os.path.isabs(arg_path):
            return arg_path
        # If expected_subdir_in_data is provided, construct path from project_root/data/expected_subdir/
        if expected_subdir_in_data:
            return os.path.join(project_root, 'data', expected_subdir_in_data, os.path.basename(arg_path))
        # Otherwise, assume it might be relative to project_root or a full path including 'data' directory part
        # For simplicity, if not absolute, assume it's like 'data/subdir/file.csv' from project root
        # or just 'subdir/file.csv' which we might want to prefix with 'data'
        # A common robust way: if not absolute, join with project_root. User provides path from root.
        return os.path.join(project_root, arg_path) 

    print(f"Loading initial fire data from: {args.initial_fire_data}")
    try:
        # Using provided paths directly, assuming they are correct (absolute or relative to execution)
        initial_fire_df = pd.read_csv(args.initial_fire_data)
        initial_fire_df['ignition_time'] = pd.to_datetime(initial_fire_df['ignition_time'])
        if 'node_id' not in initial_fire_df.columns: initial_fire_df['node_id'] = initial_fire_df.index
        
        print(f"Loading all nodes data from: {args.all_nodes_data}")
        all_nodes_df = pd.read_csv(args.all_nodes_data)
        if 'node_id' not in all_nodes_df.columns: all_nodes_df['node_id'] = all_nodes_df.index
        
        # Simple check for required columns in feature dataframes
        required_cols = ['latitude', 'longitude', 'windspeed', 'winddirection', 
                         'temperature', 'humidity', 'rainfall', 'ndvi', 'elevation']
        for df_check, df_name_from_args in [(initial_fire_df, args.initial_fire_data), (all_nodes_df, args.all_nodes_data)]:
            for col in required_cols:
                if col not in df_check.columns:
                    raise ValueError(f"Missing '{col}' in data from '{df_name_from_args}'")
    except FileNotFoundError as e: print(f"[ERROR] Data file not found: {e}. Exiting."); sys.exit(1)
    except ValueError as e: print(f"[ERROR] Data loading: {e}. Exiting."); sys.exit(1)
    except Exception as e: print(f"[ERROR] Data loading error: {e}. Exiting."); sys.exit(1)
    print("Data loaded successfully.")

    neighbor_max_dist = args.neighbor_max_dist if args.neighbor_max_dist is not None else 0.1
    def _neighbor_finder_wrapper(node_id, all_nodes, excluded_node_ids):
        return example_neighbor_finder(node_id, all_nodes, excluded_node_ids, max_dist_config=neighbor_max_dist)

    print(f"Starting prediction for {args.num_steps} steps...")
    predicted_df = predict_wildfire_spread(
        initial_fire_df,
        all_nodes_df,
        args.num_steps,
        current_c_coeffs,
        destination_calculator_func=example_destination_calculator, 
        neighbor_finder_func=_neighbor_finder_wrapper)

    if predicted_df.empty:
        print("No spread predicted.")
    else:
        print(f"\nPrediction completed. {len(predicted_df[predicted_df['prediction_step_count'] > 0])} new ignitions.")
        try:
            predicted_df.to_csv(output_csv_path, index=False)
            print(f"Predictions saved to {output_csv_path}")
        except Exception as e: print(f"[ERROR] Saving predictions: {e}")

    if args.actual_fire_data:
        print(f"\nLoading actual fire data for evaluation: {args.actual_fire_data}")
        try:
            actual_df = pd.read_csv(args.actual_fire_data)
            actual_df['ignition_time'] = pd.to_datetime(actual_df['ignition_time'])
            if 'node_id' not in actual_df.columns: actual_df['node_id'] = actual_df.index
            print("Calculating accuracy...")
            accuracy_results = calculate_accuracy(predicted_df, actual_df)
            print(f"Accuracy Metrics: {accuracy_results}")
        except FileNotFoundError: print(f"[WARN] Actual fire data not found: '{args.actual_fire_data}'. Skipping eval.")
        except Exception as e: print(f"[ERROR] Accuracy calculation error: {e}")
    else: print("\nActual fire data not provided. Skipping accuracy evaluation.")
    print("--- Wildfire Spread Prediction Workflow Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Wildfire Spread Prediction Model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Input files are expected to be paths relative to the project root, or absolute paths.
    # Example: data/initial_fire_data/file.csv
    parser.add_argument("--initial_fire_data", type=str, required=True, help="Path to initial fire data CSV (e.g., data/initial_fire_data/initial.csv).")
    parser.add_argument("--all_nodes_data", type=str, required=True, help="Path to all nodes data CSV (e.g., data/all_nodes_data/all_nodes.csv).")
    parser.add_argument("--actual_fire_data", type=str, help="(Optional) Path to actual fire data CSV for evaluation (e.g., data/actual_fire_data/actual.csv).")
    parser.add_argument("--output_csv_name", type=str, default="predicted_spread.csv", help="Output CSV file name (will be saved in project_root/prediction_data/ dir).")
    parser.add_argument("--c_coeffs", type=str, help='Override C coefficients. Format: "c1:val1,c2:val2,...".')
    parser.add_argument("--num_steps", type=int, default=10, help="Number of new ignition events to predict.")
    parser.add_argument("--neighbor_max_dist", type=float, default=0.1, help="Max distance (degrees) for example neighbor finder.")
    args = parser.parse_args()
    main_prediction_workflow(args)
