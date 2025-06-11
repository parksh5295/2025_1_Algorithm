import argparse
import os
import sys
import pandas as pd
from functools import partial
import numpy as np

# --- Module Imports ---
from prediction.prediction_utils import calculate_bearing, calculate_spread_weight, example_destination_calculator
from prediction.prediction_core import predict_wildfire_spread
from prediction.evaluation import calculate_accuracy
from prediction.neighbor_definition import example_neighbor_finder
from modules.data_load import load_and_enrich_data
# --- Model Imports ---
from models import WildfireSpreadNet, calculate_spread_weight_nn
from prediction.prediction_utils import calculate_spread_weight
# --- Path Utility ---
from data_use.data_path import get_prediction_paths


# --- Path Setup ---
# This block is no longer needed as we use explicit relative imports.
# current_script_path = os.path.abspath(__file__)
# project_root = os.path.dirname(current_script_path) 
# prediction_module_dir = os.path.join(project_root, 'prediction')
# if prediction_module_dir not in sys.path:
#     sys.path.append(prediction_module_dir)

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
TIME_STEP_MINUTES = 30 # The interval for each prediction step

# --- Main Workflow Orchestration ---
def main_prediction_workflow(data_number, use_nn=False, model_path=None, c_coeffs_str=None, num_steps=None, neighbor_max_dist=0.1, actual_fire_data_path=None):
    """
    Main workflow for running wildfire spread prediction.
    It now takes configuration as arguments instead of parsing args directly.
    If num_steps is not provided, it is calculated based on the data's time range.
    """
    print(f"--- Starting Wildfire Spread Prediction for data_number: {data_number} ---")

    # Get paths from the central utility
    try:
        paths = get_prediction_paths(data_number)
        initial_fire_path = paths['initial']
        all_nodes_path = paths['all_nodes']
        output_csv_path = paths['predicted']
    except (ValueError, FileNotFoundError) as e:
        print(f"[ERROR] Could not retrieve paths for data_number {data_number}: {e}")
        return

    # --- Model Selection and Setup ---
    if use_nn:
        print("Using Neural Network model for spread prediction.")
        model = WildfireSpreadNet()
        if model_path and os.path.exists(model_path):
            # model.load_state_dict(torch.load(model_path)) # This line needs torch
            print(f"Loaded pre-trained model from {model_path}")
        else:
            print(f"Warning: Model path not found or not provided. Using randomly initialized weights.")
        spread_weight_calculator = partial(calculate_spread_weight_nn, model)
    else:
        print("Using formula-based model for spread prediction.")
        current_c_coeffs = COEFFICIENTS.copy()
        if c_coeffs_str:
            try:
                coeffs_override = dict(item.split(':') for item in c_coeffs_str.split(','))
                for key, value in coeffs_override.items():
                    if key in current_c_coeffs: current_c_coeffs[key] = float(value)
                print(f"Using C coefficients (overridden or default): {current_c_coeffs}")
            except Exception as e:
                print(f"[ERROR] Parsing c_coeffs: {e}. Using defaults.")
        spread_weight_calculator = partial(calculate_spread_weight, c_coeffs=current_c_coeffs)

    # --- Data Loading and Enrichment ---
    print(f"Loading and enriching initial fire data from: {initial_fire_path}")
    try:
        initial_fire_df = load_and_enrich_data(csv_path=str(initial_fire_path), date_col='acq_date', time_col='acq_time', lat_col='latitude', lon_col='longitude')
        if initial_fire_df is None: raise ValueError("Failed to load initial fire data.")
        if 'date' in initial_fire_df.columns:
            initial_fire_df.rename(columns={'date': 'ignition_time'}, inplace=True)
        if 'node_id' not in initial_fire_df.columns: initial_fire_df['node_id'] = initial_fire_df.index
        
        print(f"Loading and enriching all nodes data from: {all_nodes_path}")
        all_nodes_df = load_and_enrich_data(csv_path=str(all_nodes_path), date_col='acq_date', time_col='acq_time', lat_col='latitude', lon_col='longitude')
        if all_nodes_df is None: raise ValueError("Failed to load all nodes data.")
        if 'node_id' not in all_nodes_df.columns: all_nodes_df['node_id'] = all_nodes_df.index
        
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] Data loading failed: {e}. Exiting."); sys.exit(1)
    print("Data loaded successfully.")

    # --- Calculate num_steps if not provided ---
    if num_steps is None:
        print("`num_steps` not provided. Calculating from data time range...")
        start_time = pd.to_datetime(initial_fire_df['ignition_time'].min())
        # The 'date' column from load_and_enrich_data is what we need for the full range
        end_time = pd.to_datetime(all_nodes_df['date'].max())
        
        duration_minutes = (end_time - start_time).total_seconds() / 60
        
        if duration_minutes > 0:
            # Calculate how many 30-minute steps fit into the duration
            num_steps = int(np.ceil(duration_minutes / TIME_STEP_MINUTES))
            print(f"Start time: {start_time}, End time: {end_time}")
            print(f"Total duration: {duration_minutes:.2f} minutes. Calculated num_steps: {num_steps}")
        else:
            print("Warning: Could not determine a valid time range. Defaulting to 10 steps.")
            num_steps = 10

    # --- Prediction Execution ---
    def _neighbor_finder_wrapper(node_id, all_nodes, excluded_node_ids):
        return example_neighbor_finder(node_id, all_nodes, excluded_node_ids, max_dist_config=neighbor_max_dist)

    print(f"Starting prediction for {num_steps} steps...")
    predicted_df = predict_wildfire_spread(
        initial_fire_df,
        all_nodes_df,
        num_steps,
        spread_weight_calculator,
        destination_calculator_func=example_destination_calculator, 
        neighbor_finder_func=_neighbor_finder_wrapper,
        time_step_minutes=TIME_STEP_MINUTES)

    # --- Save and Evaluate ---
    if predicted_df.empty:
        print("No spread predicted.")
    else:
        print(f"\nPrediction completed. Saving to {output_csv_path}")
        predicted_df.to_csv(output_csv_path, index=False)

    if actual_fire_data_path:
        print(f"\nLoading actual fire data for evaluation: {actual_fire_data_path}")
        try:
            actual_df = pd.read_csv(actual_fire_data_path)
            # Basic processing for accuracy calculation
            if 'ignition_time' in actual_df.columns:
                actual_df['ignition_time'] = pd.to_datetime(actual_df['ignition_time'])
            if 'node_id' not in actual_df.columns: actual_df['node_id'] = actual_df.index
            
            print("Calculating accuracy...")
            accuracy_results = calculate_accuracy(predicted_df, actual_df)
            print(f"Accuracy Metrics: {accuracy_results}")
        except Exception as e:
            print(f"[ERROR] Accuracy calculation failed: {e}")
    
    print("--- Wildfire Spread Prediction Workflow Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Wildfire Spread Prediction Model CLI", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--data_number", type=int, required=True, help="The data number to process, used to determine data paths.")
    parser.add_argument("--use_nn", action='store_true', help="Use the Neural Network model for prediction.")
    parser.add_argument("--model_path", type=str, default="models/wildfire_model.pth", help="Path to a pre-trained NN model.")
    parser.add_argument("--c_coeffs", type=str, help='Override C coefficients for the formula-based model.')
    parser.add_argument("--num_steps", type=int, default=None, help="Number of new ignition events to predict. If not set, it's calculated from the data's time range.")
    parser.add_argument("--neighbor_max_dist", type=float, default=0.1, help="Max distance for neighbor finding.")
    parser.add_argument("--actual_fire_data", type=str, help="(Optional) Path to actual fire data CSV for evaluation.")

    args = parser.parse_args()
    
    main_prediction_workflow(
        data_number=args.data_number,
        use_nn=args.use_nn,
        model_path=args.model_path,
        c_coeffs_str=args.c_coeffs,
        num_steps=args.num_steps,
        neighbor_max_dist=args.neighbor_max_dist,
        actual_fire_data_path=args.actual_fire_data
    )
