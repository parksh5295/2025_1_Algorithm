import networkx as nx
import numpy as np
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
    return G
