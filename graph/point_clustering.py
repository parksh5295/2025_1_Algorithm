# Location-based clustering -> Create a node

from sklearn.cluster import DBSCAN
import numpy as np


def cluster_fire_points(df, radius_km=3):
    """
    Latitude/Longitude-based DBSCAN clustering (3 km radius)
    Returned with a cluster_id column appended to each row
    """
    # Earth radius
    EARTH_RADIUS = 6371.0

    # Latitude/Longitude -> Radian conversion
    coords_rad = np.radians(df[['latitude', 'longitude']].values)

    # Convert distance unit (rad) -> DBSCAN eps must be in radians
    eps_rad = radius_km / EARTH_RADIUS

    db = DBSCAN(eps=eps_rad, min_samples=1, algorithm='ball_tree', metric='haversine')
    df['cluster_id'] = db.fit_predict(coords_rad)

    return df


def summarize_clusters(df):
    """
    Summarize the center coordinates, average elevation, and first time point of each cluster as a node
    The returned df contains 1 row per cluster
    """
    grouped = df.groupby('cluster_id')

    nodes = grouped.agg({
        'latitude': 'mean',
        'longitude': 'mean',
        'elevation': 'mean',
        'date': 'min'
    }).reset_index()

    nodes.rename(columns={
        'latitude': 'center_latitude',
        'longitude': 'center_longitude',
        'elevation': 'avg_elevation',
        'date': 'start_time'
    }, inplace=True)

    return nodes
