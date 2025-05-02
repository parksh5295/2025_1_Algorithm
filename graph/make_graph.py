import networkx as nx
import numpy as np
from geopy.distance import geodesic


def build_fire_graph(nodes_df, max_distance_km=10):
    G = nx.DiGraph()
    for i, node_i in nodes_df.iterrows():
        G.add_node(i, **node_i.to_dict())
        for j, node_j in nodes_df.iterrows():
            if i == j: continue
            # Time order: Only consider nodes with later time
            if node_j['date'] <= node_i['date']:
                continue
            dist = geodesic((node_i['latitude'], node_i['longitude']),
                            (node_j['latitude'], node_j['longitude'])).km
            if dist <= max_distance_km:
                elev_diff = node_j['elevation'] - node_i['elevation']
                weight = dist + max(0, -elev_diff) * 0.1  # If elevation increases, it spreads less
                G.add_edge(i, j, weight=weight)
    return G
