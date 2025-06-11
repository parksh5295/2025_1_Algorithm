import math
import numpy as np
import pandas as pd # Added if Series/DataFrame types are hinted or used internally

def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculates the initial bearing (angle) from point 1 to point 2.
    Result is in degrees, 0-360, clockwise from North.
    """
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    delta_lon = lon2_rad - lon1_rad

    y = math.sin(delta_lon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - \
        math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)
    
    initial_bearing_rad = math.atan2(y, x)
    initial_bearing_deg = math.degrees(initial_bearing_rad)
    
    # Normalize to 0-360
    compass_bearing_deg = (initial_bearing_deg + 360) % 360
    return compass_bearing_deg

def calculate_spread_weight(source_node_features, target_node_features, 
                              destination_metric, c_coeffs):
    """
    Calculates the spread weight from a source node to a target node.
    Args are assumed to be dict-like or pd.Series.
    """
    # Wind at source A
    windspeed_A = source_node_features['windspeed']
    phi_A_rad = math.radians(source_node_features['winddirection'])

    # Bearing from A to B
    theta_AB_rad = math.radians(calculate_bearing(source_node_features['latitude'], source_node_features['longitude'],
                                                target_node_features['latitude'], target_node_features['longitude']))

    # Temperature difference (A-B)
    temp_diff = source_node_features['temperature'] - target_node_features['temperature']

    # Humidity difference (B-A)
    humidity_diff = target_node_features['humidity'] - source_node_features['humidity']
    
    # Rainfall difference (B-A)
    rainfall_diff = target_node_features['rainfall'] - source_node_features['rainfall']

    # NDVI sum (B+A)
    ndvi_sum = target_node_features['ndvi'] + source_node_features['ndvi']

    # Elevation sum (A+B)
    elevation_sum = source_node_features['elevation'] + target_node_features['elevation']

    # --- Spread Weight Formula ---
    # A higher weight means a higher probability of spreading.
    # Positive terms increase spread likelihood, negative terms decrease it.
    weight = (
        # c1: Destination Metric (e.g., distance). Should be negative. Closer = more likely.
        -c_coeffs['c1'] * destination_metric +
        
        # c2: Wind Assistance. Should be positive. Wind blowing towards target = more likely.
        +c_coeffs['c2'] * windspeed_A * math.cos(theta_AB_rad - phi_A_rad) +
        
        # c3: Temperature Difference (Source - Target). Negative sign means spread is less likely if source is hotter.
        # This can be debated, but for now, we leave it. A fire spreading to a cooler area.
        -c_coeffs['c3'] * temp_diff +
        
        # c4: Humidity Difference (Target - Source). Should be negative. Higher humidity at target = less likely.
        -c_coeffs['c4'] * humidity_diff -
        
        # c5: Rainfall Difference (Target - Source). Negative sign is correct. More rain at target = less likely.
        -c_coeffs['c5'] * rainfall_diff +
        
        # c6: NDVI (Vegetation) Sum. Should be positive. More vegetation = more fuel = more likely.
        +c_coeffs['c6'] * ndvi_sum -
        
        # c7: Elevation Sum. Negative sign means spread is less likely at higher elevations. Plausible.
        -c_coeffs['c7'] * elevation_sum
    )
    return max(weight, 0.0001)

def example_destination_calculator(source_node_series, target_node_series):
    """
    Example: Using Euclidean distance in degrees as the '(Destination)' metric.
    Assumes source_node_series and target_node_series are dict-like or pd.Series
    with 'longitude' and 'latitude' keys.
    """
    dist = np.sqrt(
        (source_node_series['longitude'] - target_node_series['longitude'])**2 +
        (source_node_series['latitude'] - target_node_series['latitude'])**2
    )
    # print(f"  [Debug] Destination metric (dist) from {source_node_series['node_id']} to {target_node_series['node_id']}: {dist:.4f}")
    return dist 