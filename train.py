import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- Project-specific Imports ---
# Make sure the models and utilities are accessible
from models.wildfire_net import WildfireSpreadNet, prepare_input_features
from prediction.prediction_utils import example_destination_calculator

# --- PyTorch Dataset for Wildfire Data ---

class WildfireDataset(Dataset):
    """
    Custom Dataset for loading wildfire spread data.
    It expects a CSV file where each row represents a potential spread event
    from a source node to a target node.
    """
    def __init__(self, csv_path):
        """
        Args:
            csv_path (string): Path to the csv file with spread data.
        """
        print(f"Loading data from {csv_path}...")
        self.data_frame = pd.read_csv(csv_path)
        print("Data loaded successfully.")

        # For faster access, we can convert to records, but pandas .iloc is also fine for moderate sizes
        # self.records = self.data_frame.to_dict('records')

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the row corresponding to the index
        row = self.data_frame.iloc[idx]
        
        # The row should contain all necessary features for both source and target nodes.
        # We assume the CSV is structured like:
        # source_lat, source_lon, ..., target_lat, target_lon, ..., actual_spread (1 or 0)
        
        # Reconstruct source and target feature dictionaries/series from the flat row
        source_features = {
            'latitude': row['source_latitude'], 'longitude': row['source_longitude'],
            'windspeed': row['source_windspeed'], 'winddirection': row['source_winddirection'],
            'temperature': row['source_temperature'], 'humidity': row['source_humidity'],
            'precipitation': row['source_precipitation'], 'ndvi': row['source_ndvi'], 'elevation': row['source_elevation']
        }
        
        target_features = {
            'latitude': row['target_latitude'], 'longitude': row['target_longitude'],
            'temperature': row['target_temperature'], 'humidity': row['target_humidity'],
            'precipitation': row['target_precipitation'], 'ndvi': row['target_ndvi'], 'elevation': row['target_elevation']
        }
        
        # Calculate the destination metric
        destination_metric = example_destination_calculator(source_features, target_features)
        
        # Prepare the input tensor for the NN model
        input_tensor = prepare_input_features(source_features, target_features, destination_metric)
        
        # Get the label (whether the fire actually spread)
        label = torch.tensor(row['actual_spread'], dtype=torch.float32)
        
        return input_tensor, label


# --- Main Training Function ---

def train(args):
    """Main training and validation loop."""
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create Datasets and DataLoaders
    train_dataset = WildfireDataset(csv_path=args.train_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Initialize Model, Loss Function, and Optimizer
    model = WildfireSpreadNet().to(device)
    criterion = nn.BCELoss() # Binary Cross-Entropy Loss for binary classification (spread/no-spread)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("\n--- Starting Training ---")
    for epoch in range(args.epochs):
        model.train() # Set model to training mode
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # Print statistics
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss:.4f}")

    print("--- Finished Training ---")

    # Save the trained model
    save_path = args.save_path
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Wildfire Spread Prediction Model")
    
    # --- Paths and Directories ---
    parser.add_argument('--train_data', type=str, required=True, help="Path to the training data CSV file.")
    # parser.add_argument('--val_data', type=str, help="Path to the validation data CSV file.") # Optional validation set
    parser.add_argument('--save_path', type=str, default='models/wildfire_model.pth', help="Path to save the trained model.")
    
    # --- Training Hyperparameters ---
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate.")
    parser.add_argument('--epochs', type=int, default=200, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training.")
    
    args = parser.parse_args()
    
    train(args) 