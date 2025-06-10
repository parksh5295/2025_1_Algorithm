import torch
import torch.nn as nn
import math

# This model is designed to predict a spread weight based on various environmental features.
# It takes a flat feature vector as input, unlike the CNN models also in this directory.

class WildfireSpreadNet(nn.Module):
    """A neural network to predict wildfire spread weight."""
    def __init__(self, input_size=7): # Default input size based on feature preparation
        super(WildfireSpreadNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            # Using Sigmoid to constrain the output to a 0-1 range, 
            # representing a probability-like weight.
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# Helper function to calculate bearing, which is a required input feature
def _calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculates the initial bearing from point 1 to point 2."""
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(math.radians, [lat1, lon1, lat2, lon2])
    delta_lon = lon2_rad - lon1_rad
    y = math.sin(delta_lon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon)
    initial_bearing_rad = math.atan2(y, x)
    return (math.degrees(initial_bearing_rad) + 360) % 360

# This function prepares the input tensor for the WildfireSpreadNet model.
# It needs to be called before passing data to the model for training or inference.
def prepare_input_features(source_node_features, target_node_features, destination_metric):
    """Prepares a flat feature tensor for the neural network from source and target node data."""
    
    # Wind components
    windspeed_A = source_node_features.get('windspeed', 0)
    wind_dir_A = source_node_features.get('winddirection', 0)
    bearing_AB = _calculate_bearing(
        source_node_features['latitude'], source_node_features['longitude'],
        target_node_features['latitude'], target_node_features['longitude']
    )
    wind_factor = windspeed_A * math.cos(math.radians(bearing_AB) - math.radians(wind_dir_A))

    # Feature vector
    features = [
        destination_metric,
        wind_factor,
        source_node_features.get('temperature', 0) - target_node_features.get('temperature', 0),
        target_node_features.get('humidity', 0) - source_node_features.get('humidity', 0),
        target_node_features.get('precipitation', 0) - source_node_features.get('precipitation', 0),
        target_node_features.get('ndvi', 0) + source_node_features.get('ndvi', 0),
        source_node_features.get('elevation', 0) + source_node_features.get('elevation', 0),
    ]
    
    return torch.tensor(features, dtype=torch.float32)

def calculate_spread_weight_nn(model, source_node_features, target_node_features, destination_metric):
    """
    Calculates the spread weight using the trained neural network.
    This is the main function to be used as the `spread_weight_calculator_func`.
    """
    model.eval()
    with torch.no_grad():
        input_features = prepare_input_features(source_node_features, target_node_features, destination_metric)
        input_features = input_features.unsqueeze(0) # Add batch dimension
        predicted_weight = model(input_features)
    return predicted_weight.squeeze().item() 