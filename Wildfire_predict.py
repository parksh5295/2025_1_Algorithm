import argparse
import os
import sys
import pandas as pd
from functools import partial

# --- Module Imports ---
from prediction_utils import calculate_bearing, calculate_spread_weight, example_destination_calculator
from prediction_core import predict_wildfire_spread
from evaluation import calculate_accuracy
from neighbor_definition import example_neighbor_finder
from modules.data_load import load_and_enrich_data
# --- Model Imports ---
from models import WildfireSpreadNet, calculate_spread_weight_nn
from prediction_utils import calculate_spread_weight


# --- Path Setup ---
# Current script is Wildfire_predict.py (in project_root)
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_script_path) # project_root is the directory of this script

# prediction_module_dir is now directly relative to project_root
prediction_module_dir = os.path.join(project_root, 'prediction')
if prediction_module_dir not in sys.path:
    sys.path.append(prediction_module_dir)

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
    '''
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
    '''

    # --- Model Selection and Setup ---
    if args.use_nn:
        print("Using Neural Network model for spread prediction.")
        # Initialize the Neural Network model
        model = WildfireSpreadNet() # Input size is defaulted in the class
        
        # Here you would typically load a pre-trained model's weights
        # For example:
        # if os.path.exists(args.model_path):
        #     model.load_state_dict(torch.load(args.model_path))
        #     print(f"Loaded pre-trained model from {args.model_path}")
        # else:
        #     print(f"Warning: Model path {args.model_path} not found. Using randomly initialized weights.")

        # The weight calculator function is the NN function, with the model instance passed in.
        spread_weight_calculator = partial(calculate_spread_weight_nn, model)

    else:
        print("Using formula-based model for spread prediction.")
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
        
        # The weight calculator is the formula-based function, with coefficients passed in.
        spread_weight_calculator = partial(calculate_spread_weight, c_coeffs=current_c_coeffs)


    # Output CSV path will be in project_root/prediction_data/
    output_dir = os.path.join(project_root, 'prediction_data')
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, args.output_csv_name)

    # Adjust input paths (This helper function might need review depending on how paths are truly given)
    # For now, assuming args.initial_fire_data and args.all_nodes_data are paths
    # that load_and_enrich_data can handle (e.g., relative to project root or absolute)

    print(f"Loading and enriching initial fire data from: {args.initial_fire_data}")
    try:
        # initial_fire_df = pd.read_csv(args.initial_fire_data) # OLD WAY
        initial_fire_df = load_and_enrich_data(
            csv_path=args.initial_fire_data,
            date_col='acq_date', # Assuming 'acq_date' from previous context, adjust if different
            time_col='acq_time',  # Assuming 'acq_time', adjust if different, or None if datetime is combined
            lat_col='latitude',
            lon_col='longitude'
        )
        if initial_fire_df is None:
            raise ValueError(f"Failed to load or enrich initial fire data from {args.initial_fire_data}")

        # Assuming 'ignition_time' is created or correctly named by load_and_enrich_data as 'date'
        # If 'ignition_time' is a specific column name expected later, ensure it's handled.
        # load_and_enrich_data standardizes to a 'date' column (datetime object).
        # Let's rename 'date' to 'ignition_time' if that's what downstream code expects for initial_fire_df.
        if 'date' in initial_fire_df.columns:
             initial_fire_df.rename(columns={'date': 'ignition_time'}, inplace=True)
        elif 'ignition_time' not in initial_fire_df.columns:
             raise ValueError("Missing 'ignition_time' or 'date' column after enriching initial_fire_df")
        else: # 'ignition_time' was already present and possibly a datetime
             initial_fire_df['ignition_time'] = pd.to_datetime(initial_fire_df['ignition_time'])


        if 'node_id' not in initial_fire_df.columns: initial_fire_df['node_id'] = initial_fire_df.index
        
        print(f"Loading and enriching all nodes data from: {args.all_nodes_data}")
        # all_nodes_df = pd.read_csv(args.all_nodes_data) # OLD WAY
        all_nodes_df = load_and_enrich_data(
            csv_path=args.all_nodes_data,
            date_col='acq_date', # Assuming 'acq_date', adjust if different
            time_col='acq_time',  # Assuming 'acq_time', adjust if different
            lat_col='latitude',
            lon_col='longitude'
        )
        if all_nodes_df is None:
            raise ValueError(f"Failed to load or enrich all nodes data from {args.all_nodes_data}")
        
        # For all_nodes_df, the 'date' column from enrichment is what we need for weather at each node's potential ignition time.
        # Ensure this 'date' column is present.
        if 'date' not in all_nodes_df.columns:
            raise ValueError("Missing 'date' column after enriching all_nodes_df, needed for environmental data.")


        if 'node_id' not in all_nodes_df.columns: all_nodes_df['node_id'] = all_nodes_df.index
        
        # Simple check for required columns in feature dataframes
        # These columns should now be present after load_and_enrich_data
        required_env_cols = ['latitude', 'longitude', 'windspeed', 'winddirection', 
                             'temperature', 'humidity', 'rainfall', 'ndvi', 'elevation']
        
        # Check initial_fire_df (which needs 'ignition_time' as its primary time ref)
        # Environmental data should be associated with its 'ignition_time'
        for col in required_env_cols:
            if col not in initial_fire_df.columns:
                raise ValueError(f"Missing enriched column '{col}' in data from '{args.initial_fire_data}'")
        if 'ignition_time' not in initial_fire_df.columns: # Double check ignition_time
             raise ValueError(f"Missing 'ignition_time' in data from '{args.initial_fire_data}' after enrichment")

        # Check all_nodes_df (which uses the 'date' column for its environmental context)
        for col in required_env_cols:
            if col not in all_nodes_df.columns:
                raise ValueError(f"Missing enriched column '{col}' in data from '{args.all_nodes_data}'")
        if 'date' not in all_nodes_df.columns: # Double check the generic 'date' column for all_nodes
            raise ValueError(f"Missing 'date' in data from '{args.all_nodes_data}' after enrichment")

    except FileNotFoundError as e: print(f"[ERROR] Data file not found during load_and_enrich: {e}. Exiting."); sys.exit(1)
    except ValueError as e: print(f"[ERROR] Data loading/enrichment: {e}. Exiting."); sys.exit(1)
    print("Data loaded successfully.")

    neighbor_max_dist = args.neighbor_max_dist if args.neighbor_max_dist is not None else 0.1
    def _neighbor_finder_wrapper(node_id, all_nodes, excluded_node_ids):
        return example_neighbor_finder(node_id, all_nodes, excluded_node_ids, max_dist_config=neighbor_max_dist)

    print(f"Starting prediction for {args.num_steps} steps...")
    predicted_df = predict_wildfire_spread(
        initial_fire_df,
        all_nodes_df,
        args.num_steps,
        #current_c_coeffs,
        spread_weight_calculator, # Pass the selected calculator function
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
    #parser.add_argument("--c_coeffs", type=str, help='Override C coefficients. Format: "c1:val1,c2:val2,...".')
    
    # --- Arguments for model selection and configuration ---
    parser.add_argument("--use_nn", action='store_true', help="Use the Neural Network model for prediction instead of the formula.")
    parser.add_argument("--model_path", type=str, default="models/wildfire_model.pth", help="Path to a pre-trained NN model state dictionary.")
    
    parser.add_argument("--c_coeffs", type=str, help='Override C coefficients for the formula-based model. Format: "c1:val1,c2:val2,...".')
    parser.add_argument("--num_steps", type=int, default=10, help="Number of new ignition events to predict.")
    parser.add_argument("--neighbor_max_dist", type=float, default=0.1, help="Max distance (degrees) for example neighbor finder.")
    args = parser.parse_args()
    main_prediction_workflow(args)
