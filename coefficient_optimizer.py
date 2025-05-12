import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Matern, ExpSineSquared, DotProduct
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib # For saving/loading models/scalers
import sys # For sys.path modification
import os # For path joining
import json # For saving coefficients

# --- Path Setup for Importing from Sibling 'prediction' Directory ---
# Assuming coefficient_optimizer.py is in the project root, and 'prediction' is a sibling directory.
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_script_path)
prediction_module_dir = os.path.join(project_root, 'prediction')
if prediction_module_dir not in sys.path:
    sys.path.append(prediction_module_dir)

# For data loading utilities
modules_dir = os.path.join(project_root, 'modules') 
if modules_dir not in sys.path:
    sys.path.append(modules_dir)
data_use_dir = os.path.join(project_root, 'data_use') # For data_path
if data_use_dir not in sys.path:
    sys.path.append(data_use_dir)
utiles_dir = os.path.join(project_root, 'utiles') # For estimate_fire_spread_times
if utiles_dir not in sys.path:
    sys.path.append(utiles_dir)

from prediction_utils import calculate_bearing, example_destination_calculator # Added example_destination_calculator
from data_load import load_and_enrich_data # Assuming it handles enrichment for all needed features
from data_path import load_data_path # To get the path for raw data
from estimate_time import estimate_fire_spread_times # To get 'ignition_time'

# --- Configuration & Constants ---
# Example: Path to the historical fire data with actual spread times
# This data needs to be prepared: each row representing a spread event (A -> B)
# with features (from A & B) and the target (actual_spread_duration_hours)
HISTORICAL_DATA_PATH = "data/processed_historical_fire_spread_data.csv" 
OPTIMIZED_COEFFS_PATH = os.path.join(prediction_module_dir, "optimized_c_coefficients.json")

# Define the features that will be extracted from source and target node data
# These should correspond to the terms in the calculate_spread_weight function
# (excluding the c_coefficients themselves)
# Order matters if we want to map optimized weights back to c1, c2, ...
FEATURE_TERMS_FUNCTIONS = {
    # 'term_name': lambda source_features, target_features, destination_metric: value
    # Example (assuming calculate_spread_weight will be refactored to expose these terms):
    # 'dest_metric_term': lambda sf, tf, dm: dm, # Corresponds to c1
    # 'wind_term': lambda sf, tf, dm: -sf['windspeed'] * np.cos(np.radians(calculate_bearing(sf['lat'], sf['lon'], tf['lat'], tf['lon'])) - np.radians(sf['winddir'])), # c2
    # 'temp_diff_term': lambda sf, tf, dm: -(sf['temperature'] - tf['temperature']), # c3 (note the sign based on previous discussion)
    # ... and so on for c4, c5, c6, c7
}

class CoefficientOptimizer:
    def __init__(self, feature_term_functions):
        self.feature_term_functions = feature_term_functions
        self.gp_model = None
        self.scaler_X = None
        self.scaler_y = None
        self.optimized_coeffs = None

    def _create_spread_event_dataset(self, enriched_df):
        """
        Transforms time-ordered fire ignition data into a dataset of
        (source_node_features, target_node_features, actual_spread_duration_hours) events.
        Assumes enriched_df is sorted by ignition_time and has all necessary features.
        """
        events = []
        if 'ignition_time' not in enriched_df.columns:
            print("[ERROR] 'ignition_time' column missing from enriched_df. Cannot create spread events.")
            return pd.DataFrame()
        
        # Ensure sorted by ignition time
        df_sorted = enriched_df.sort_values(by='ignition_time').reset_index(drop=True)

        required_features = ['latitude', 'longitude', 'windspeed', 'winddirection', 
                             'temperature', 'humidity', 'rainfall', 'ndvi', 'elevation', 'ignition_time', 'node_id']
                             # 'node_id' is for debugging/tracking, not a direct feature for GP terms

        for col in required_features:
            if col not in df_sorted.columns:
                print(f"[ERROR] Required column '{col}' for feature engineering is missing in input DataFrame.")
                return pd.DataFrame()

        for i in range(len(df_sorted) - 1):
            source_node = df_sorted.iloc[i]
            target_node = df_sorted.iloc[i+1]

            # Calculate actual spread duration
            duration = (target_node['ignition_time'] - source_node['ignition_time']).total_seconds() / 3600.0
            if duration <= 0: # Skip if target ignited before or at the same time as source (data issue or simultaneous)
                continue

            event = {}
            # Store all features for source and target for clarity, or select only needed ones
            for col_prefix, node_data in [('source_', source_node), ('target_', target_node)]:
                for feature_col in required_features:
                    event[f"{col_prefix}{feature_col}"] = node_data[feature_col]
            
            event['actual_spread_duration_hours'] = duration
            events.append(event)
        
        return pd.DataFrame(events)

    def _load_and_preprocess_data(self, spread_event_df):
        """
        Extracts features (terms of the spread weight formula) from spread_event_df 
        and target (actual spread duration). Scales features and target.
        """
        if spread_event_df.empty:
            print("[WARN] spread_event_df is empty. No data to process.")
            # Return empty arrays or handle as appropriate
            return np.array([]), np.array([]), []

        features_for_gp = []
        targets = []
        feature_names = list(self.feature_term_functions.keys())

        for _, row in spread_event_df.iterrows():
            # Reconstruct source_features (sf) and target_features (tf) dicts/Series from the row
            sf = pd.Series({k: row[f'source_{k}'] for k in ['latitude', 'longitude', 'windspeed', 'winddirection', 'temperature', 'humidity', 'rainfall', 'ndvi', 'elevation']})
            tf = pd.Series({k: row[f'target_{k}'] for k in ['latitude', 'longitude', 'windspeed', 'winddirection', 'temperature', 'humidity', 'rainfall', 'ndvi', 'elevation']})
            
            # Use imported example_destination_calculator for consistency
            dm = example_destination_calculator(sf, tf) 

            term_values = []
            valid_terms = True
            for term_name, func in self.feature_term_functions.items():
                try:
                    term_val = func(sf, tf, dm) 
                    # Check for NaN or Inf in term_val if functions can produce them
                    if pd.isna(term_val) or np.isinf(term_val):
                        print(f"[WARN] Invalid value (NaN/Inf) for term '{term_name}' in row. Skipping event.")
                        valid_terms = False
                        break
                    term_values.append(term_val)
                except Exception as e:
                    print(f"[WARN] Error calculating term '{term_name}' for an event: {e}. Skipping event.")
                    valid_terms = False
                    break
            
            if valid_terms:
                features_for_gp.append(term_values)
                targets.append(row['actual_spread_duration_hours'])
        
        if not features_for_gp:
            print("[WARN] No valid features could be extracted. Returning empty arrays.")
            return np.array([]), np.array([]), []
            
        X = np.array(features_for_gp)
        y = np.array(targets).reshape(-1, 1)

        if X.shape[0] == 0:
            print("[WARN] No data to process after feature engineering attempts. Returning empty arrays.")
            return np.array([]), np.array([]), []

        # Scale features and target
        self.scaler_X = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(X)
        
        self.scaler_y = StandardScaler()
        y_scaled = self.scaler_y.fit_transform(y)
        
        return X_scaled, y_scaled.ravel(), feature_names

    def _create_dummy_historical_data(self, num_rows=100):
        """Creates a dummy DataFrame for historical spread data."""
        # This is highly simplified. Real data would have source/target node IDs,
        # their detailed features, and observed spread times.
        data = {
            # Placeholder columns - actual columns will depend on your data schema
            'source_node_id': np.arange(num_rows),
            'target_node_id': np.arange(num_rows) + num_rows,
            # Features would ideally be more structured (e.g., nested dicts or separate columns)
            'source_temp': np.random.uniform(10, 30, num_rows),
            'target_temp': np.random.uniform(10, 30, num_rows),
            'source_windspeed': np.random.uniform(0, 20, num_rows),
            'source_winddirection': np.random.uniform(0, 360, num_rows),
            'source_lat': np.random.uniform(30, 40, num_rows),
            'source_lon': np.random.uniform(120, 130, num_rows),
            'target_lat': np.random.uniform(30, 40, num_rows),
            'target_lon': np.random.uniform(120, 130, num_rows),
            'target_humidity': np.random.uniform(20, 80, num_rows),
            'source_humidity': np.random.uniform(20, 80, num_rows),
            'target_rainfall': np.random.uniform(0, 5, num_rows),
            'source_rainfall': np.random.uniform(0, 5, num_rows),
            'target_ndvi': np.random.uniform(0.1, 0.8, num_rows),
            'source_ndvi': np.random.uniform(0.1, 0.8, num_rows),
            'target_elevation': np.random.uniform(0, 1000, num_rows),
            'source_elevation': np.random.uniform(0, 1000, num_rows),
            'destination_metric_example': np.random.uniform(0.01, 0.5, num_rows), # e.g., distance
            'actual_spread_duration_hours': np.random.uniform(0.5, 24, num_rows) # Target
        }
        return pd.DataFrame(data)

    def define_gp_model(self, kernel=None, alpha=1e-10, n_restarts_optimizer=10):
        """
        Defines the Gaussian Process Regressor model.
        The kernel is crucial for GP performance.
        """
        if kernel is None:
            # A common starting kernel: RBF (for smoothness) + WhiteKernel (for noise)
            # DotProduct kernel can be good for learning linear relationships (like coefficients)
            # For coefficients, we might want a linear model essentially, so DotProduct is interesting.
            # If we assume the terms ARE the features, and we want to find their weights (coeffs),
            # a simple DotProduct kernel might learn these weights directly.
            # If we assume the terms ARE the features, and we want to find their weights (coeffs),
            # a simple DotProduct kernel might learn these weights directly.
            # kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + \
            #          WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
            kernel = DotProduct(sigma_0=1.0, sigma_0_bounds="fixed") + WhiteKernel(noise_level=0.5)


        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha, # Value added to the diagonal of the kernel matrix during fitting
            n_restarts_optimizer=n_restarts_optimizer, # Number of restarts of the optimizer for finding the kernel's parameters
            random_state=42
        )
        print(f"GP Model defined with kernel: {self.gp_model.kernel_}")


    def train_model(self, X_train, y_train):
        """Trains the GP model."""
        if self.gp_model is None:
            self.define_gp_model()
        
        print("Training GP model...")
        self.gp_model.fit(X_train, y_train)
        print("GP Model training completed.")
        print(f"Learned kernel parameters: {self.gp_model.kernel_}")
        # If using DotProduct, the learned weights might be interpretable as coefficients.
        # For a simple DotProduct kernel, kernel_.k1.sigma_0 might not directly give coefficients.
        # If the kernel is just DotProduct(), then gp_model.kernel_.theta might be relevant,
        # or one might need to look at how sklearn handles weights for linear kernels.
        # Alternatively, if GP is used to predict spread_time, then a separate optimization
        # step is needed to find C-coeffs that make calculate_spread_weight match GP predictions.

        # For now, let's assume the GP model *is* the predictor for spread time,
        # and we'll need another way to get C-coeffs if we want to stick to the original formula.
        # A simpler approach: if the GP's mean prediction is f(x), and our formula is Sum(c_i * x_i),
        # then the 'c_i' are essentially the weights if f(x) is linear.
        # If using DotProduct kernel, gp.coef_ might exist after fitting, or can be derived.
        # For scikit-learn's GP with DotProduct, it's not as direct as linear regression's .coef_
        # A common way to get "coefficients" from a GP with a linear-like kernel is to check its behavior on unit vectors,
        # or to use it as a surrogate model for a separate coefficient optimization process.

        # For this initial setup, we'll just train the GP to predict scaled spread time.
        # Extracting C-coefficients will be a subsequent step or a different interpretation.


    def extract_coefficients_from_gp(self):
        """
        Placeholder for extracting/interpreting C-coefficients from the trained GP model.
        This is non-trivial and depends on the kernel and approach.
        If the kernel is a simple DotProduct, this might involve looking at kernel parameters.
        More generally, the GP model could be used as a surrogate to optimize C-coeffs
        for the original formula.
        
        For now, let's assume a simple (and possibly naive) interpretation if DotProduct kernel is used.
        If the GP model itself has learned the linear relationship, the 'weights' of this linear
        relationship are our coefficients. Scikit-learn's GP doesn't directly expose these like LinearRegression.
        
        A more robust method would be to use the trained GP to predict spread times for a range of
        feature inputs, and then fit a linear model (Sum ci*feature_i) to these GP predictions
        to find the ci.
        """
        if self.gp_model is None or self.gp_model.kernel is None: # Check if model is trained
            print("[WARN] GP model is not trained. Cannot extract coefficients.")
            self.optimized_coeffs = {f'c{i+1}': np.random.rand() for i in range(len(self.feature_term_functions) if self.feature_term_functions else 7)}
            print(f"[WARN] Placeholder coefficients generated due to untrained model: {self.optimized_coeffs}")
            return self.optimized_coeffs

        print(f"Attempting to interpret coefficients from GP kernel: {self.gp_model.kernel_}")

        # This is a placeholder. Actual extraction is complex.
        # If DotProduct kernel, the components might relate to feature weights.
        # For a kernel like DotProduct() (no ConstantKernel wrapper),
        # the learned parameters are not directly the linear coefficients.
        # if isinstance(self.gp_model.kernel_.k1, DotProduct): # Assuming k1 is DotProduct if wrapped
        #    # This is not how sklearn stores coefficients for DotProduct in GPR
        #    # For a simple DotProduct kernel, one might need to do more work.
        #    pass

        # For demonstration, if we had 'n' features, we'd need 'n' coefficients.
        # Let's return placeholder coefficients based on feature names for now.
        # This needs to be replaced with a proper method.
        
        # A simple approach for DotProduct: if X is scaled (mean 0, std 1), and y is scaled,
        # and the model is y_scaled = X_scaled @ w, then w are the scaled coefficients.
        # However, GPR with DotProduct kernel doesn't just give 'w'.

        # Let's assume for now that the features in FEATURE_TERMS_FUNCTIONS directly 
        # correspond to c1*term1, c2*term2 etc. in the original formula.
        # And the GP with DotProduct tries to learn these terms.
        # This part needs significant refinement.
        num_coeffs = len(self.feature_term_functions) if self.feature_term_functions else 7
        self.optimized_coeffs = {f'c{i+1}': np.random.rand() for i in range(num_coeffs)}
        print(f"[WARN] Placeholder coefficients generated: {self.optimized_coeffs}")
        print("[INFO] Proper coefficient extraction from GP needs further implementation.")
        return self.optimized_coeffs


    def save_coefficients(self, filepath=None):
        """Saves the optimized coefficients to a file."""
        if filepath is None: filepath = OPTIMIZED_COEFFS_PATH
        if self.optimized_coeffs is None:
            print("[WARN] No optimized coefficients to save.")
            return
        
        # Example: save as JSON
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.optimized_coeffs, f, indent=4)
        print(f"Optimized coefficients saved to {filepath}")

    def run_optimization(self, spread_event_df_for_training):
        """Runs the full optimization pipeline using pre-processed spread event data."""
        X_scaled, y_scaled, feature_names = self._load_and_preprocess_data(spread_event_df_for_training)
        
        if X_scaled.shape[0] == 0:
            print("[ERROR] No data available for training the GP model. Aborting optimization.")
            return
            
        self.train_model(X_scaled, y_scaled)
        self.extract_coefficients_from_gp()
        if self.optimized_coeffs: self.save_coefficients()

def main():
    print("--- Starting Coefficient Optimization using Gaussian Process ---")
    
    # 1. Load and prepare the full dataset (similar to Wildfire_spread_graph.py)
    #    This typically involves loading raw CSV, enriching with weather/NDVI/elevation.
    #    For now, we assume a single data_number for simplicity.
    data_number_for_training = 1 # Example data number
    print(f"Loading and enriching data for dataset_number: {data_number_for_training}")
    try:
        raw_csv_path = load_data_path(data_number_for_training) # Gets path like 'data/train/train_1.csv'
        
        # --- Caching logic for enriched data ---
        base, ext = os.path.splitext(raw_csv_path)
        # Define a specific cache name for data used by the optimizer
        # to avoid conflict with Wildfire_spread_graph.py's cache if structures differ slightly.
        enriched_cache_path = f"{base}_enriched_for_optimizer{ext}" 

        if os.path.exists(enriched_cache_path):
            print(f"[INFO] Loading enriched data from cache: {enriched_cache_path}")
            enriched_df_with_features = pd.read_csv(enriched_cache_path)
            # Ensure 'date' column is parsed if it exists and is needed by estimate_fire_spread_times
            if 'date' in enriched_df_with_features.columns:
                 enriched_df_with_features['date'] = pd.to_datetime(enriched_df_with_features['date'])
            # Also ensure other critical columns like latitude, longitude, acq_date, acq_time are correctly typed if read from CSV
            # For simplicity, assuming read_csv handles them or they are not immediately critical before estimate_fire_spread_times
        else:
            print(f"[INFO] Enriched cache not found. Loading and enriching from raw: {raw_csv_path}")
            enriched_df_with_features = load_and_enrich_data(
                csv_path=raw_csv_path,
                date_col='acq_date',
                time_col='acq_time',
                lat_col='latitude',
                lon_col='longitude'
            )
            if enriched_df_with_features is None or enriched_df_with_features.empty:
                 raise ValueError("Data loading and enrichment failed or returned empty DataFrame.")
            print(f"[INFO] Saving enriched data to cache: {enriched_cache_path}")
            enriched_df_with_features.to_csv(enriched_cache_path, index=False)

        if enriched_df_with_features.empty: # Added check for safety
            print("[ERROR] Enriched data is empty. Aborting."); return

        print("Estimating fire spread times (to get/refine ignition_time)...")
        full_processed_df = estimate_fire_spread_times(enriched_df_with_features.copy())
        if 'ignition_time' not in full_processed_df.columns:
            raise ValueError("'ignition_time' not found after estimate_fire_spread_times.")
        full_processed_df['ignition_time'] = pd.to_datetime(full_processed_df['ignition_time'])

    except Exception as e:
        print(f"[ERROR] Failed to load or process base data for training: {e}")
        return

    # Sort by ignition time and take the first 10% of *events*
    # An event is a transition from one fire point to the next.
    # So, if N points, there are N-1 potential events.
    full_processed_df = full_processed_df.sort_values(by='ignition_time').reset_index(drop=True)
    num_initial_points = len(full_processed_df)
    if num_initial_points < 2:
        print("[ERROR] Not enough data points (<2) to create spread events for training.")
        return
        
    # Calculate 10% of the data points to define the subset for creating training events
    # If we have P points, it creates P-1 events. We want 10% of these P-1 events.
    # So, we can take roughly the first 10% of points to generate these events.
    cutoff_index = max(1, int(0.1 * num_initial_points)) # Ensure at least 1 event can be formed if possible
    training_subset_df = full_processed_df.iloc[:cutoff_index + 1] # Need one extra point to form the last event in subset
    if len(training_subset_df) < 2:
        print(f"[WARN] After taking 10% ({cutoff_index+1} points), not enough data points (<2) to create spread events for training.")
        return

    print(f"Using first {len(training_subset_df)} data points (approx 10%) to generate training spread events.")

    feature_definitions = {
        'term_for_c1': lambda sf, tf, dm: dm, # Destination Metric
        'term_for_c2': lambda sf, tf, dm: -sf.get('windspeed',0) * np.cos(np.radians(calculate_bearing(sf['latitude'], sf['longitude'],tf['latitude'], tf['longitude'])) - np.radians(sf.get('winddirection',0))),
        'term_for_c3': lambda sf, tf, dm: -(sf.get('temperature',0) - tf.get('temperature',0)),
        'term_for_c4': lambda sf, tf, dm: (tf.get('humidity',0) - sf.get('humidity',0)), # Sign is positive here, c4 will be learned
        'term_for_c5': lambda sf, tf, dm: -(tf.get('rainfall',0) - sf.get('rainfall',0)),
        'term_for_c6': lambda sf, tf, dm: -(tf.get('ndvi',0) + sf.get('ndvi',0)),
        'term_for_c7': lambda sf, tf, dm: -(sf.get('elevation',0) + tf.get('elevation',0)),
    }

    optimizer = CoefficientOptimizer(feature_term_functions=feature_definitions)
    
    # Create spread event data from the 10% subset
    print("Creating spread event dataset for training...")
    spread_event_df_for_training = optimizer._create_spread_event_dataset(training_subset_df)

    if spread_event_df_for_training.empty:
        print("[ERROR] No spread events could be created from the initial 10% data. Aborting.")
        return
    
    print(f"Generated {len(spread_event_df_for_training)} spread events for training.")
    optimizer.run_optimization(spread_event_df_for_training)
    
    print("--- Coefficient Optimization Finished ---")

if __name__ == '__main__':
    # import math # Already imported at the top level if needed by calculate_bearing in prediction_utils
    main() 