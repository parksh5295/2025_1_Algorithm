import numpy as np
import pandas as pd
from datetime import timedelta


# Latitude-longitude distance calculation functions
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    
    return R * 2 * np.arcsin(np.sqrt(a))


def estimate_fire_spread_times(df, spread_speed_kmph=1.5):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)

    # use the first point as the origin
    origin = df.iloc[0]
    origin_time = origin['date']

    def compute_estimated_time(row):
        distance_km = haversine_distance(origin['latitude'], origin['longitude'],
                                         row['latitude'], row['longitude'])
        # time = distance / speed
        hours_needed = distance_km / spread_speed_kmph
        return origin_time + timedelta(hours=hours_needed)

    df['estimated_time'] = df.apply(compute_estimated_time, axis=1)

    return df
