import os
import matplotlib.pyplot as plt
import networkx as nx
import requests
from io import BytesIO
from PIL import Image


def draw_graph_snapshot(G, filenumber, sequence_id, latlon_bounds=None):
    # Get the current script directory (graph folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to reach the root directory (code's parent)
    root_dir = os.path.dirname(os.path.dirname(script_dir))
    # Create path to data directory
    data_dir = os.path.join(root_dir, 'data')
    
    # Create frame directory structure
    frame_dir = os.path.join(data_dir, 'frame')
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
        # OpenStreetMap Static Map API (via third-party, e.g. staticmap.openstreetmap.de)
        osm_url = (
            f"https://staticmap.openstreetmap.de/staticmap.php?"
            f"center={(lat_min+lat_max)/2},{(lon_min+lon_max)/2}"
            f"&zoom=10&size=800x640&maptype=mapnik"
            f"&markers={lat_min},{lon_min},lightblue1"
        )
        try:
            response = requests.get(osm_url, timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                plt.imshow(img, extent=[lon_min, lon_max, lat_min, lat_max], alpha=0.3, zorder=0)
            else:
                print(f"[WARN] Failed to download OSM background image: {response.status_code}")
        except Exception as e:
            print(f"[WARN] Exception during OSM background download: {e}")

    # Use the correct attribute names for node positions
    pos = {n: (G.nodes[n].get('center_longitude', 0), 
                 G.nodes[n].get('center_latitude', 0)) 
           for n in G.nodes()}
    
    nx.draw(G, pos, with_labels=True, node_color='red', edge_color='gray', node_size=300, font_size=8)
    plt.title(f"Dataset {filenumber} - Snapshot #{sequence_id}")

    # Fix xlim/ylim
    if latlon_bounds is not None:
        lat_min, lat_max, lon_min, lon_max = latlon_bounds
        plt.xlim(lon_min, lon_max)
        plt.ylim(lat_min, lat_max)

    plt.savefig(filename)
    plt.close()
