import os
import matplotlib.pyplot as plt
import networkx as nx


def draw_graph_snapshot(G, filenumber, sequence_id):
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
    # Use the correct attribute names for node positions
    # Ensure these attributes exist in G.nodes[n]
    pos = {n: (G.nodes[n].get('center_longitude', 0), 
                 G.nodes[n].get('center_latitude', 0)) 
           for n in G.nodes()}
    
    nx.draw(G, pos, with_labels=True, node_color='red', edge_color='gray', node_size=300, font_size=8)
    plt.title(f"Dataset {filenumber} - Snapshot #{sequence_id}")
    plt.savefig(filename)
    plt.close()
