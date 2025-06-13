import pandas as pd
import heapq

# Import necessary functions from other modules within the 'prediction' package
from .prediction_utils import calculate_spread_weight, example_destination_calculator, calculate_bearing
from .neighbor_definition import example_neighbor_finder

def predict_wildfire_spread(
    initial_fire_df: pd.DataFrame,
    all_nodes_df: pd.DataFrame,
    num_steps: int,
    spread_weight_calculator: callable,
    destination_calculator_func: callable = example_destination_calculator,
    neighbor_finder_func: callable = example_neighbor_finder,
    time_step_minutes: int = 15,
    spread_threshold: float = 0.9
) -> pd.DataFrame:
    """
    Predicts the spread of a wildfire over a given number of steps, allowing for multiple spreads per step.
    
    Args:
        initial_fire_df (pd.DataFrame): DataFrame of initially ignited nodes.
        all_nodes_df (pd.DataFrame): DataFrame of all possible nodes in the area.
        num_steps (int): The number of prediction steps to run.
        spread_weight_calculator (callable): Function to calculate spread weight between nodes.
        destination_calculator_func (callable): Function to calculate the distance/destination metric.
        neighbor_finder_func (callable): Function to find potential neighbors for a given node.
        time_step_minutes (int): The duration of each prediction step in minutes.
        spread_threshold (float): The minimum weight required for a spread to occur.
        
    Returns:
        pd.DataFrame: A DataFrame with the predicted spread path, including ignition times.
    """
    if 'ignition_time' not in initial_fire_df.columns or initial_fire_df['ignition_time'].isnull().any():
        raise ValueError("Initial fire DataFrame must have a valid 'ignition_time' column.")

    # Convert ignition_time to datetime if it's not already
    initial_fire_df['ignition_time'] = pd.to_datetime(initial_fire_df['ignition_time'])

    ignited_nodes_df = initial_fire_df.copy()
    ignited_node_ids = set(ignited_nodes_df['node_id'].unique())
    
    # This will store the full history of ignitions for the output
    prediction_history = [initial_fire_df]
    
    # We use a single, continuous timeline. The last known ignition time is our reference.
    last_ignition_time = ignited_nodes_df['ignition_time'].max()

    print(f"Starting prediction from {len(ignited_node_ids)} initial nodes for {num_steps} steps of {time_step_minutes} mins each.")
    print(f"Initial ignition time: {last_ignition_time.strftime('%Y-%m-%d %H:%M:%S')}")

    for step in range(1, num_steps + 1):
        potential_spreads = []
        
        # In each step, any currently ignited node is a potential source
        current_sources_df = all_nodes_df[all_nodes_df['node_id'].isin(ignited_node_ids)]

        for _, source_node in current_sources_df.iterrows():
            # Find neighbors that have not already been ignited
            neighbor_nodes_df = neighbor_finder_func(source_node['node_id'], all_nodes_df, excluded_node_ids=ignited_node_ids)
            
            if neighbor_nodes_df.empty:
                continue

            for _, target_node in neighbor_nodes_df.iterrows():
                destination_metric = destination_calculator_func(source_node, target_node)
                weight = spread_weight_calculator(source_node, target_node, destination_metric)
                potential_spreads.append({
                    'source_node_id': source_node['node_id'],
                    'node_id': target_node['node_id'],
                    'weight': weight
                })
        
        if not potential_spreads:
            print(f"Step {step}: No potential new spreads found from any of the {len(ignited_node_ids)} ignited nodes. Stopping prediction.")
            break
            
        spreads_df = pd.DataFrame(potential_spreads)
        
        # --- Select all spreads exceeding the threshold ---
        successful_spreads_df = spreads_df[spreads_df['weight'] > spread_threshold]
        
        if successful_spreads_df.empty:
            print(f"Step {step}: No spreads exceeded the threshold of {spread_threshold}. Stopping prediction.")
            break

        # --- CRITICAL: Advance time by the fixed step duration for all new ignitions in this step ---
        time_delta = pd.to_timedelta(time_step_minutes, unit='m')
        new_ignition_time = last_ignition_time + time_delta
        
        newly_ignited_nodes_this_step = set()

        for _, spread in successful_spreads_df.iterrows():
            newly_ignited_node_id = spread['node_id']
            
            # Avoid re-igniting a node that was just ignited in the same step by another source
            if newly_ignited_node_id in ignited_node_ids or newly_ignited_node_id in newly_ignited_nodes_this_step:
                continue

            source_of_ignition_id = spread['source_node_id']
            
            new_ignition_df = all_nodes_df[all_nodes_df['node_id'] == newly_ignited_node_id].copy()
            
            new_ignition_df['ignition_time'] = new_ignition_time
            new_ignition_df['source_node_id'] = source_of_ignition_id
            new_ignition_df['prediction_step_count'] = step

            prediction_history.append(new_ignition_df)
            newly_ignited_nodes_this_step.add(newly_ignited_node_id)
            
            print(f"  Step {step}: Spread from {int(source_of_ignition_id)} to {int(newly_ignited_node_id)} with weight {spread['weight']:.4f}. New ignition time: {new_ignition_time.strftime('%Y-%m-%d %H:%M:%S')}")

        if not newly_ignited_nodes_this_step:
            print(f"Step {step}: All potential spreads were to already ignited nodes. Stopping.")
            break

        # Add all unique new ignitions from this step to the main set
        ignited_node_ids.update(newly_ignited_nodes_this_step)
        
        # Update the last ignition time for the next step
        last_ignition_time = new_ignition_time

    if not prediction_history:
        return pd.DataFrame()

    final_predictions_df = pd.concat(prediction_history, ignore_index=True)
    return final_predictions_df