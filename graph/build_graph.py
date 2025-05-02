# build a graph
from graph.point_clustering import cluster_fire_points, summarize_clusters
from graph.make_graph import build_fire_graph


def cluster_and_build_graph(df):
    df = cluster_fire_points(df)
    nodes_df = summarize_clusters(df)
    G = build_fire_graph(nodes_df)

    return df, nodes_df, G