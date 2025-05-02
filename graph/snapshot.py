import os
import matplotlib.pyplot as plt
import networkx as nx


def draw_graph_snapshot(G, filenumber, sequence_id):
    # Calculate the target directory path based on filenumber
    # Get the directory containing the current script (snapshot.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go one level up to the parent directory (presumably 'graph')
    # parent_dir = os.path.dirname(script_dir) # This might be too high if 'graph' is the current dir
    # Assuming snapshot.py is directly in the 'graph' directory
    graph_dir = script_dir
    frame_folder_name = f"frame_{str(filenumber).zfill(4)}"
    target_dir = os.path.join(graph_dir, frame_folder_name)

    # Check and create the specific directory for this frame
    os.makedirs(target_dir, exist_ok=True)

    # Define the filename using sequence_id within the target directory
    # Pad sequence_id similarly to filenumber for consistent sorting (e.g., 4 digits)
    # Assuming sequence_id is an integer counter
    snapshot_filename = f"snapshot_{str(sequence_id).zfill(4)}.png"
    filename = os.path.join(target_dir, snapshot_filename)

    plt.figure(figsize=(10, 8))
    pos = {n: (G.nodes[n]['lon'], G.nodes[n]['lat']) for n in G.nodes()}
    
    nx.draw(G, pos, with_labels=True, node_color='red', edge_color='gray', node_size=300, font_size=8)
    plt.title(f"Dataset {filenumber} - Snapshot #{sequence_id}")
    plt.savefig(filename)
    plt.close()
