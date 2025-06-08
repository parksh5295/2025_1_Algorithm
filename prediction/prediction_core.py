import pandas as pd
import heapq

# Import necessary functions from other modules within the 'prediction' package
from .prediction_utils import calculate_spread_weight, calculate_bearing # calculate_bearing is used by c_s_w
# Assuming example_destination_calculator might also be in prediction_utils or passed directly
# from .prediction_utils import example_destination_calculator 
# Assuming neighbor_finder_func is passed as an argument and defined elsewhere (e.g., neighbor_definition.py)

def predict_wildfire_spread(initial_fire_nodes_df, all_nodes_df, 
                              prediction_steps,
                              #c_coeffs,
                              spread_weight_calculator_func, # Changed from c_coeffs
                              destination_calculator_func, 
                              neighbor_finder_func):
    """
    Simulates wildfire spread for a given number of steps.
    Uses a priority queue (min-heap) to manage fire spread events.
    """
    predicted_ignitions = []
    # burnt_nodes_times stores {node_id: actual_ignition_datetime}
    burnt_nodes_times = {}
    # event_queue_refined stores (potential_ignition_datetime, target_node_id, source_node_id, source_node_features_at_time_of_spread_calculation)
    event_queue_refined = [] 

    # Initialize with the starting fires and find their first potential spreads
    for _, initial_node_row in initial_fire_nodes_df.iterrows():
        node_id = initial_node_row['node_id']
        ignition_time = pd.to_datetime(initial_node_row['ignition_time'])
        
        if node_id in burnt_nodes_times:
            # This can happen if the same initial fire node is listed multiple times
            # or an earlier event already burned it. Prioritize the earliest known ignition.
            if ignition_time < burnt_nodes_times[node_id]:
                burnt_nodes_times[node_id] = ignition_time
                # Update its entry in predicted_ignitions if necessary (though more complex to track here)
            else:
                continue # Already have an earlier or same ignition time recorded
        else:
            burnt_nodes_times[node_id] = ignition_time
            predicted_ignitions.append({
                'node_id': node_id,
                'ignition_time': ignition_time,
                'source_node_id': None, # Initial fires have no preceding source
                'prediction_step_count': 0 
            })

        # For this initial fire, calculate spread to its neighbors
        neighbors_df = neighbor_finder_func(node_id, all_nodes_df, burnt_nodes_times.keys())
        for _, neighbor_node_row in neighbors_df.iterrows():
            neighbor_id = neighbor_node_row['node_id']
            if neighbor_id in burnt_nodes_times: 
                continue # Should be redundant if neighbor_finder_func is correct
                
            destination_metric = destination_calculator_func(initial_node_row, neighbor_node_row)
            # Use initial_node_row for source features as this is when it starts spreading
            '''
            spread_duration_hours = calculate_spread_weight(initial_node_row, neighbor_node_row,
                                                          destination_metric, c_coeffs)
            '''
            spread_duration_hours = spread_weight_calculator_func(initial_node_row, neighbor_node_row,
                                                                  destination_metric)
            
            potential_ignition_at = ignition_time + pd.to_timedelta(spread_duration_hours, unit='hours')
            # Store the features of the source node *at the time this potential spread was calculated*
            heapq.heappush(event_queue_refined, 
                           (potential_ignition_at, neighbor_id, node_id, initial_node_row.copy()))

    # Process events from the queue to predict new ignitions
    newly_ignited_count = 0
    while event_queue_refined and newly_ignited_count < prediction_steps:
        # Get the event with the earliest potential ignition time
        ignite_at, target_node_id, source_node_id, 
        source_node_features_when_event_created = heapq.heappop(event_queue_refined)
        
        # If target node already burnt by an earlier event, skip this path
        if target_node_id in burnt_nodes_times and burnt_nodes_times[target_node_id] <= ignite_at:
            continue
        
        # This is a new, confirmed ignition
        burnt_nodes_times[target_node_id] = ignite_at
        newly_ignited_count += 1
        predicted_ignitions.append({
            'node_id': target_node_id,
            'ignition_time': ignite_at,
            'source_node_id': source_node_id,
            'prediction_step_count': newly_ignited_count 
        })
        
        # print(f"Step {newly_ignited_count}: Node {target_node_id} predicted to ignite at {ignite_at} from {source_node_id}")

        # Now, this newly_ignited_node becomes a source for further spread.
        # Its features are needed. Get them from all_nodes_df using its ID.
        # We assume features in all_nodes_df are static or represent the state at a relevant time.
        newly_ignited_node_all_features = all_nodes_df[all_nodes_df['node_id'] == target_node_id]
        if newly_ignited_node_all_features.empty:
            # print(f"[WARN] Features for newly ignited node {target_node_id} not found in all_nodes_df. Cannot spread further from it.")
            continue
        newly_ignited_node_features = newly_ignited_node_all_features.iloc[0]

        # Find neighbors of the newly ignited node, excluding already burnt ones
        neighbors_df = neighbor_finder_func(target_node_id, all_nodes_df, burnt_nodes_times.keys())
        
        for _, neighbor_node_row in neighbors_df.iterrows():
            next_target_id = neighbor_node_row['node_id']
            # This check should ideally be handled by neighbor_finder_func too
            if next_target_id in burnt_nodes_times: 
                continue

            destination_metric = destination_calculator_func(newly_ignited_node_features, neighbor_node_row)
            # Use newly_ignited_node_features for source features in this new spread calculation
            '''
            spread_duration_hours = calculate_spread_weight(newly_ignited_node_features, neighbor_node_row,
                                                          destination_metric, c_coeffs)
            '''
            spread_duration_hours = spread_weight_calculator_func(newly_ignited_node_features, neighbor_node_row,
                                                                  destination_metric)
            
            # The new potential ignition time is relative to when the current source (newly_ignited_node) ignited
            potential_ignition_at = ignite_at + pd.to_timedelta(spread_duration_hours, unit='hours')
            heapq.heappush(event_queue_refined, 
                           (potential_ignition_at, next_target_id, target_node_id, newly_ignited_node_features.copy()))
            
    return pd.DataFrame(predicted_ignitions) 