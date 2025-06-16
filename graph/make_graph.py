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
    
    # === Additional logic: Mandatory connection for unconnected nodes ===
    # sorted nodes
    sorted_nodes = sorted(G.nodes(), key=lambda x: G.nodes[x]['start_time'])
    
    for i, current_node in enumerate(sorted_nodes):
        if i == 0:  # first node has no previous node
            continue
            
        # check if there is an incoming edge to the current node
        has_incoming_edge = any(G.has_edge(prev_node, current_node) for prev_node in sorted_nodes[:i])
        
        # if there is no incoming edge, connect to 1~5 random previous nodes
        if not has_incoming_edge:
            current_attrs = G.nodes[current_node]
            previous_nodes = sorted_nodes[:i]  # all nodes before the current node
            
            # calculate distances to all previous nodes
            node_distances = []
            for prev_node in previous_nodes:
                prev_attrs = G.nodes[prev_node]
                dist = geodesic((prev_attrs['center_latitude'], prev_attrs['center_longitude']),
                                (current_attrs['center_latitude'], current_attrs['center_longitude'])).km
                node_distances.append((prev_node, dist))
            
            # sort by distance and select randomly 1~5 from the nearest nodes
            node_distances.sort(key=lambda x: x[1])
            available_nodes = min(5, len(node_distances))  # consider up to 5 nodes
            num_connections = random.randint(1, available_nodes)  # randomly select 1~available_nodes
            
            # randomly select from the nearest 5 nodes
            nearest_candidates = node_distances[:available_nodes]
            selected_nodes = random.sample(nearest_candidates, num_connections)
            
            print(f"[EDGE_DEBUG] No connections found for node {current_node}, connecting to {num_connections} random previous nodes")
            
            # connect to the selected nodes
            for nearest_node, min_dist in selected_nodes:
                elev_prev = G.nodes[nearest_node].get('avg_elevation', 0)
                elev_curr = current_attrs.get('avg_elevation', 0)
                elev_diff = elev_curr - elev_prev
                weight = min_dist + max(0, -elev_diff) * 0.1
                G.add_edge(nearest_node, current_node, weight=weight)
                print(f"[EDGE_DEBUG] Connected node {current_node} to node {nearest_node} (distance: {min_dist:.2f}km)")
            
            print(f"[EDGE_DEBUG] Total {num_connections} mandatory connections created for node {current_node}")
    
    return G
