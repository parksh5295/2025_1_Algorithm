# build a graph
from graph.point_clustering import cluster_fire_points, summarize_clusters
from graph.make_graph import build_fire_graph


def cluster_and_build_graph(df):
    # === DEBUGGING: Print input df to cluster_and_build_graph ===
    print(f"[C&B_DEBUG] Input df columns to cluster_and_build_graph: {df.columns.tolist()}")
    if not df.empty:
        print(f"[C&B_DEBUG] Input df head to cluster_and_build_graph:\n{df.head(2)}")
    # === DEBUGGING END ===

    df_after_clustering = cluster_fire_points(df.copy()) # Use .copy() to be safe

    # === DEBUGGING: Print df after cluster_fire_points ===
    print(f"[C&B_DEBUG] df columns after cluster_fire_points: {df_after_clustering.columns.tolist()}")
    if not df_after_clustering.empty:
        print(f"[C&B_DEBUG] df head after cluster_fire_points:\n{df_after_clustering.head(2)}")
    # === DEBUGGING END ===

    nodes_df = summarize_clusters(df_after_clustering) # Using df_after_clustering

    # === DEBUGGING: Print nodes_df after summarize_clusters ===
    print(f"[C&B_DEBUG] nodes_df columns after summarize_clusters: {nodes_df.columns.tolist()}")
    if not nodes_df.empty:
        print(f"[C&B_DEBUG] nodes_df head after summarize_clusters:\n{nodes_df.head(2)}")
    # === DEBUGGING END ===
    
    G = build_fire_graph(nodes_df)

    return df_after_clustering, nodes_df, G # Modified to return df_after_clustering