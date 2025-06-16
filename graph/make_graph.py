import networkx as nx
import numpy as np
import random
from geopy.distance import geodesic


def build_fire_graph(nodes_df, max_distance_km=10):
    G = nx.DiGraph()
    for i, node_i_series in nodes_df.iterrows():
        G.add_node(i, **node_i_series.to_dict())
    
    for i in G.nodes():
        node_i_attrs = G.nodes[i]
        for j in G.nodes():
            if i == j: continue
            node_j_attrs = G.nodes[j]
            
            if node_j_attrs['start_time'] <= node_i_attrs['start_time']:
                continue
            
            dist = geodesic((node_i_attrs['center_latitude'], node_i_attrs['center_longitude']),
                            (node_j_attrs['center_latitude'], node_j_attrs['center_longitude'])).km
            
            if dist <= max_distance_km:
                elev_i = node_i_attrs.get('avg_elevation', 0)
                elev_j = node_j_attrs.get('avg_elevation', 0)
                elev_diff = elev_j - elev_i
                weight = dist + max(0, -elev_diff) * 0.1
                G.add_edge(i, j, weight=weight)
    
    # === Mandatory connection for unconnected nodes ===
    sorted_nodes = sorted(G.nodes(), key=lambda x: G.nodes[x]['start_time'])
    
    for i, current_node in enumerate(sorted_nodes):
        if i == 0:  # first node has no previous node
            continue
            
        # check if there is an incoming edge to the current node
        has_incoming_edge = any(G.has_edge(prev_node, current_node) for prev_node in sorted_nodes[:i])
        
        # if there is no incoming edge, connect to the nearest previous node
        if not has_incoming_edge:
            current_attrs = G.nodes[current_node]
            previous_nodes = sorted_nodes[:i]  # all nodes before the current node
            
            # find the nearest previous node
            min_dist = float('inf')
            nearest_node = None
            
            for prev_node in previous_nodes:
                prev_attrs = G.nodes[prev_node]
                dist = geodesic((prev_attrs['center_latitude'], prev_attrs['center_longitude']),
                                (current_attrs['center_latitude'], current_attrs['center_longitude'])).km
                
                if dist < min_dist:
                    min_dist = dist
                    nearest_node = prev_node
            
            # connect to the nearest node
            if nearest_node is not None:
                elev_prev = G.nodes[nearest_node].get('avg_elevation', 0)
                elev_curr = current_attrs.get('avg_elevation', 0)
                elev_diff = elev_curr - elev_prev
                weight = min_dist + max(0, -elev_diff) * 0.1
                G.add_edge(nearest_node, current_node, weight=weight)
                print(f"[EDGE_DEBUG] Mandatory connection: node {current_node} to nearest node {nearest_node} (distance: {min_dist:.2f}km)")
    
    return G
