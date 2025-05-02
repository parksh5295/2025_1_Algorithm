import os
import imageio
import glob
import re


def generate_gif_for_dataset(
    filenumber,
    frame_base_dir='.',
    output_gif_name=None,
    duration=0.5,
    frame_image_pattern='*.png'
):
    """
    Finds all the PNG frames in the folder corresponding to a specific dataset, identified by filenumber, and creates a GIF animation.
    PNG frames in a folder corresponding to a specific dataset, identified by filenumber, to create a single GIF animation.
    """
    # 1. configure the frame folder path corresponding to a specific filenumber
    filenumber_padded = str(filenumber).zfill(4)
    target_frame_folder = os.path.join(frame_base_dir, f"frame_{filenumber_padded}")

    # 2. verify the existence of the destination folder
    if not os.path.isdir(target_frame_folder):
        print(f"[ERROR] Target frame folder not found: {target_frame_folder}")
        return

    print(f"[INFO] Processing dataset {filenumber} from folder: {target_frame_folder}")

    # 3. locate and sort image files within the destination folder
    image_search_pattern = os.path.join(target_frame_folder, frame_image_pattern)
    # Finding files that fit a pattern with glob.glob
    image_files_unordered = glob.glob(image_search_pattern)

    if not image_files_unordered:
        print(f"[ERROR] No image files matching pattern '{frame_image_pattern}' found in {target_frame_folder}")
        return

    # --- Extract numbers from filenames and sort them accurately --- Sort numbers correctly
    def sort_key(filepath):
        # Find the number part at the end of the filename (e.g. snapshot_001.png -> 1, frame_10.png -> 10)
        match = re.search(r'(\d+)\.png$', os.path.basename(filepath), re.IGNORECASE)
        # If a number is found, return it as an integer, otherwise return infinity to move it to the end
        return int(match.group(1)) if match else float('inf')

    image_files = sorted(image_files_unordered, key=sort_key)
    # ----------------------------------------------------

    print(f"[INFO] Found {len(image_files)} frames for dataset {filenumber}.")
    if len(image_files) < 5: # Too few frames, print the sorted result for verification
        print(f"   (Sorted files sample: {image_files[:5]})")


    # 4. Configure the GIF save path and create the 'gif' folder
    #    Create the 'gif' folder under frame_base_dir (same level as frame_xxxx folders)
    gif_output_dir = os.path.join(frame_base_dir, 'gif')
    os.makedirs(gif_output_dir, exist_ok=True)

    # Determine the output GIF file name
    if output_gif_name is None:
        output_gif_name = f"animation_{filenumber_padded}.gif" # Default name format

    output_path = os.path.join(gif_output_dir, output_gif_name)
    print(f"[INFO] Output GIF will be saved to: {output_path}")

    # 5. Read images and create/save GIF
    try:
        images = [imageio.imread(f) for f in image_files]
        print(f"[INFO] Creating and saving GIF...")
        # Add loop=0 to make it loop indefinitely (typical GIF behavior)
        imageio.mimsave(output_path, images, duration=duration, loop=0)
        print(f"[INFO] GIF for dataset {filenumber} saved successfully!")
    except FileNotFoundError as e:
        print(f"[ERROR] Could not read image file: {e}. Check file paths and permissions.")
    except Exception as e:
        print(f"[ERROR] Failed to create or save GIF: {e}")


# --- Example usage ---
# Assume this script is in the 'utiles' folder and
# frame_xxxx folders are in the 'graph' folder
if __name__ == '__main__':
    print("\n--- Running GIF Generation Example for a specific dataset ---")

    # Configure the graph folder path (assume the parent folder of utiles)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) # The parent folder of utiles
    graph_folder_path = os.path.join(project_root, 'graph')

    # Example: Create a GIF for dataset 5 ('graph/frame_0005' folder)
    target_dataset_number = 5
    # Configure to find all png files in the target folder ('graph/frame_0005')
    generate_gif_for_dataset(
        filenumber=target_dataset_number,
        frame_base_dir=graph_folder_path, # Specify 'graph' folder as the base path
        duration=0.2,
        frame_image_pattern='*.png' # Use all png files in the 'frame_0005' folder
        # output_gif_name='custom_name_for_5.gif' # Specify the output name if needed
    )