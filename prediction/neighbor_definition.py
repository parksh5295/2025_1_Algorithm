import pandas as pd
import numpy as np

def example_neighbor_finder(node_id, all_nodes, excluded_node_ids, max_dist_config=0.1):
    """
    Example: Find neighbors within a certain lat/lon distance (in degrees),
    excluding already burnt ones (present in excluded_node_ids).
    Assumes all_nodes is a DataFrame with 'node_id', 'longitude', 'latitude'.
    max_dist_config is the threshold for distance (e.g., 0.1 degrees approx 10-11km).
    """
    # Get the current node's details
    current_node_series = all_nodes[all_nodes['node_id'] == node_id]
    if current_node_series.empty:
        # print(f"[WARN] Node ID {node_id} not found in all_nodes_df for neighbor search.")
        return pd.DataFrame() # Return empty DataFrame if node not found
    current_node = current_node_series.iloc[0]
    
    potential_neighbors = []
    for _, r_node in all_nodes.iterrows():
        # Skip self or already excluded nodes
        if r_node['node_id'] == node_id or r_node['node_id'] in excluded_node_ids:
            continue
        
        # Calculate distance (simple Euclidean on lat/lon degrees for example)
        dist = np.sqrt(
            (current_node['longitude'] - r_node['longitude'])**2 +
            (current_node['latitude'] - r_node['latitude'])**2
        )
        
        if dist < max_dist_config: # Check against the distance threshold
            potential_neighbors.append(r_node)
            
    neighbors_df = pd.DataFrame(potential_neighbors)
    # print(f"  [Debug] Node {node_id} found {len(neighbors_df)} neighbors (excluding {len(excluded_node_ids)} burnt) within {max_dist_config} dist.")
    return neighbors_df 