import argparse
import pandas as pd
import folium
from pathlib import Path

# Assuming data_path.py is in data_use directory
from data_use.data_path import get_prediction_paths

def visualize_and_compare(data_number: int):
    """
    Generates a map visualizing the difference between actual and predicted wildfire spread
    and calculates the overlap percentage.
    """
    print(f"--- Starting Visualization for data_number: {data_number} ---")

    # 1. Get file paths
    try:
        paths = get_prediction_paths(data_number)
        actual_path = paths['all_nodes']
        predicted_path = paths['predicted']
    except ValueError as e:
        print(f"[ERROR] {e}. Please use a valid data number.")
        return

    # Check if files exist
    if not actual_path.exists():
        print(f"[ERROR] Actual fire data not found at: {actual_path}")
        return
    if not predicted_path.exists():
        print(f"[ERROR] Predicted fire data not found at: {predicted_path}")
        print("Please run the prediction script first to generate this file.")
        return

    print(f"Loading actual data from: {actual_path}")
    print(f"Loading predicted data from: {predicted_path}")

    # 2. Load data
    actual_df = pd.read_csv(actual_path)
    predicted_df = pd.read_csv(predicted_path)

    # Ensure latitude and longitude columns exist
    if 'latitude' not in actual_df.columns or 'longitude' not in actual_df.columns:
        print("[ERROR] Actual data must contain 'latitude' and 'longitude' columns.")
        return
    if 'latitude' not in predicted_df.columns or 'longitude' not in predicted_df.columns:
        print("[ERROR] Predicted data must contain 'latitude' and 'longitude' columns.")
        return

    # 3. Calculate Overlap
    # Create a unique identifier for each point to handle floating point inaccuracies
    actual_df['location'] = list(zip(actual_df['latitude'].round(6), actual_df['longitude'].round(6)))
    predicted_df['location'] = list(zip(predicted_df['latitude'].round(6), predicted_df['longitude'].round(6)))

    actual_points = set(actual_df['location'])
    predicted_points = set(predicted_df['location'])

    intersection = actual_points.intersection(predicted_points)
    union = actual_points.union(predicted_points)

    if not union:
        print("No fire points found in either dataset.")
        return

    overlap_percentage = (len(intersection) / len(union)) * 100
    print("\n--- Overlap Analysis ---")
    print(f"Actual fire points: {len(actual_points)}")
    print(f"Predicted fire points: {len(predicted_points)}")
    print(f"Common points (Intersection): {len(intersection)}")
    print(f"Total unique points (Union): {len(union)}")
    print(f"Overlap Percentage (Intersection / Union): {overlap_percentage:.2f}%")

    # 4. Create Visualization
    if not union:
        print("Cannot create map as there are no data points.")
        return

    # Determine map center
    map_center_lat = pd.concat([actual_df['latitude'], predicted_df['latitude']]).mean()
    map_center_lon = pd.concat([actual_df['longitude'], predicted_df['longitude']]).mean()

    m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=10)

    # Points only in actual data (Red)
    for loc in actual_points - intersection:
        folium.CircleMarker(
            location=loc,
            radius=5,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.4
        ).add_to(m)

    # Points only in predicted data (Blue)
    for loc in predicted_points - intersection:
        folium.CircleMarker(
            location=loc,
            radius=5,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.4
        ).add_to(m)

    # Overlapping points (Purple)
    for loc in intersection:
        folium.CircleMarker(
            location=loc,
            radius=5,
            color='purple',
            fill=True,
            fill_color='purple',
            fill_opacity=0.6
        ).add_to(m)

    # 5. Save map
    output_filename = f'spread_visualization_{data_number}.html'
    m.save(output_filename)
    print(f"\nMap has been saved to: {Path(output_filename).resolve()}")
    print("--- Visualization Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualize and compare actual vs. predicted wildfire spread.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data_number",
        type=int,
        required=True,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="The data number to process for visualization."
    )
    args = parser.parse_args()
    visualize_and_compare(args.data_number) 