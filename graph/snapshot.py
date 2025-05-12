import os
import matplotlib.pyplot as plt
import networkx as nx
import requests
from io import BytesIO
from PIL import Image


def draw_graph_snapshot(G, filenumber, sequence_id, latlon_bounds=None):
    # Get the current script directory (e.g., /home/work/code/graph)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to reach the project root (e.g., /home/work)
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Construct the base data directory (e.g., /home/work/data)
    base_data_dir = os.path.join(project_root, 'data')
    
    # Construct the target directory for frames 
    # (e.g., /home/work/data/graph/frame/frame_XXXX)
    data_graph_dir = os.path.join(base_data_dir, 'graph')
    frame_dir = os.path.join(data_graph_dir, 'frame')
    frame_folder_name = f"frame_{str(filenumber).zfill(4)}"
    target_dir = os.path.join(frame_dir, frame_folder_name)

    # Create directories if they don't exist
    os.makedirs(target_dir, exist_ok=True)

    # Define the filename using sequence_id within the target directory
    snapshot_filename = f"snapshot_{str(sequence_id).zfill(4)}.png"
    filename = os.path.join(target_dir, snapshot_filename)

    plt.figure(figsize=(10, 8))

    # Automatically download background maps (OpenStreetMap Static)
    if latlon_bounds is not None:
        lat_min, lat_max, lon_min, lon_max = latlon_bounds
        # Use ESRI World Imagery (satellite) tiles for background
        # We'll fetch a static image using the static-maps.yandex.ru API as a workaround for easy satellite imagery
        # (Google Static Maps requires API key and billing)
        # For more robust solution, use a tile server and stitch tiles, but here we use a simple static image
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2
        # Yandex Static Maps API (satellite): https://static-maps.yandex.ru/1.x/?ll=lon,lat&z=zoom&l=sat&size=650,450
        # Note: size is in pixels, max 650x450 for free
        yandex_url = (
            f"https://static-maps.yandex.ru/1.x/?ll={center_lon},{center_lat}"
            f"&z=10&l=sat&size=650,450"
        )
        try:
            response = requests.get(yandex_url, timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                plt.imshow(img, extent=[lon_min, lon_max, lat_min, lat_max], alpha=0.7, zorder=0)
            else:
                print(f"[WARN] Failed to download satellite background image: {response.status_code}")
        except Exception as e:
            print(f"[WARN] Exception during satellite background download: {e}")

    # Use the correct attribute names for node positions
    pos = {n: (G.nodes[n].get('center_longitude', 0), 
                 G.nodes[n].get('center_latitude', 0)) 
           for n in G.nodes()}
    
    nx.draw(G, pos, with_labels=True, node_color='red', edge_color='red', width=1.5, node_size=300, font_size=8)
    plt.title(f"Dataset {filenumber} - Snapshot #{sequence_id}")

    # Fix xlim/ylim
    if latlon_bounds is not None:
        lat_min, lat_max, lon_min, lon_max = latlon_bounds
        plt.xlim(lon_min, lon_max)
        plt.ylim(lat_min, lat_max)
        # Add ticks for latitude and longitude
        lon_ticks = list(plt.MaxNLocator(nbins=8).tick_values(lon_min, lon_max))
        lat_ticks = list(plt.MaxNLocator(nbins=8).tick_values(lat_min, lat_max))
        plt.xticks(lon_ticks)
        plt.yticks(lat_ticks)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.axis('on') # Ensure axes are on
    plt.grid(True) # Add a grid for better readability

    plt.savefig(filename)
    plt.close()
