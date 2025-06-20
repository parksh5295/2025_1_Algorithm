from pathlib import Path
import sys


# Find the project root based on the location of this file
# Assuming data_path.py is inside the 'code/data_use' folder, and the project root is three levels up.
try:
    # Handle cases where __file__ is not defined (e.g., REPL)
    if '__file__' not in locals():
        raise NameError("__file__ is not defined. Cannot determine relative path.")

    # Absolute path of data_path.py
    current_file_path = Path(__file__).resolve()
    # Project root path (three levels up from data_path.py)
    if len(current_file_path.parents) < 3:
         raise FileNotFoundError("Cannot go up three parent directories from script location.")
    PROJECT_ROOT = current_file_path.parents[2] # Three levels up
    DATA_DIR = PROJECT_ROOT / 'data'

    # Check if the data directory exists (optional)
    if not DATA_DIR.is_dir():
        print(f"[WARN] Data directory not found at expected location: {DATA_DIR}")
        # Handle error if needed

except NameError as e:
    print(f"[ERROR] Could not determine project root path: {e}")
    # If the project root cannot be found, path creation is not possible, so exit or handle differently
    # sys.exit("Critical error: Could not determine project path.")
    # Temporary fallback (high likelihood of error) - This part may also need modification
    DATA_DIR = Path("../../..") / "data" # Three levels up
except FileNotFoundError as e:
    print(f"[ERROR] Problem finding or accessing project root path: {e}")
    DATA_DIR = Path("../../..") / "data"
except Exception as e:
    print(f"[ERROR] Unexpected error determining paths: {e}")
    # sys.exit("Critical error: Could not determine paths.")
    DATA_DIR = Path("../../..") / "data"


def load_data_path(data_number):
    """
    Returns a Path object for the data file corresponding to the given data_number.
    The path is dynamically calculated based on the location of the data_path.py file.
    """
    if data_number == 1:
        file_path = DATA_DIR / 'DL_FIRE_SV-C2_608350(250314-05)' / 'fire_nrt_SV-C2_608350.csv'
    elif data_number == 2:
        file_path = DATA_DIR / 'DL_FIRE_SV-C2_608316(230402-12)' / 'fire_archive_SV-C2_608316.csv'
    elif data_number == 3:
        file_path = DATA_DIR / 'DL_FIRE_J2V-C2_37482(2025-California)' / 'fire_nrt_J2V-C2_37482.csv'
    elif data_number == 4:
        file_path = DATA_DIR / 'DL_FIRE_SV-C2_37483(2019-Australia)' / 'fire_archive_SV-C2_37483.csv'
    elif data_number == 5:
        file_path = DATA_DIR / 'DL_FIRE_SV-C2_608318(220528-03)' / 'fire_archive_SV-C2_608318.csv'
    elif data_number == 6:
        file_path = DATA_DIR / 'DL_FIRE_SV-C2_608319(220405-12)' / 'fire_archive_SV-C2_608319.csv'
    elif data_number == 7:
        file_path = DATA_DIR / 'DL_FIRE_SV-C2_608320(220304-14)' / 'fire_archive_SV-C2_608320.csv'
    else:
        raise ValueError("Invalid data number")

    # Check if the file exists and return (optional, depending on debugging or logic)
    # if not file_path.is_file():
    #     print(f"[WARN] Data file not found for data_number {data_number} at: {file_path}")
    #     # return None
    
    return file_path # Return Path object

def get_prediction_paths(data_number):
    """
    Returns a dictionary of paths required for prediction for a given data_number.
    - 'all_nodes': The main dataset, used as the field of all possible nodes.
    - 'initial': A small subset of the main data, used as the starting fire points.
    - 'predicted': The path where the output of a prediction should be saved.
    """
    # 'all_nodes' is the original dataset from load_data_path
    all_nodes_path = load_data_path(data_number)

    # The output and initial files will be in the same directory as the 'all_nodes' file.
    output_dir = all_nodes_path.parent

    # Define the path for the initial fire data, assuming it's in the same folder
    initial_path = output_dir / 'initial_fire.csv'

    # The predicted file is saved in the same directory with a "similar_" prefix.
    predicted_path = output_dir / f"similar_{all_nodes_path.name}"
    
    # Note: This function assumes that the 'initial_fire.csv' either exists or will be created.
    # A utility to create this initial file from the 'all_nodes' data might be needed.
    # For now, we just return the conventional path.
    
    return {
        'all_nodes': all_nodes_path,
        'initial': initial_path,
        'predicted': predicted_path
    }

# Example usage (in another module):
# from data_path import load_data_path
# path_obj = load_data_path(1)
# print(f"Path for data 1: {path_obj}")
# print(f"Does it exist? {path_obj.is_file()}")
# # If needed, convert to string for use in pandas, etc.
# # df = pd.read_csv(str(path_obj)) 