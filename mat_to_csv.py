import os
import glob
import mat73
import numpy as np

# 1. THE ACTUAL CONVERSION FUNCTION
def mat73_to_npz(mat_filepath, npz_filepath):
    try:
        # Load the v7.3 .mat file
        data_dict = mat73.loadmat(mat_filepath)
        flat_data = {}
        
        # Flatten the nested cells
        def extract_arrays(nested_dict, prefix=''):
            if isinstance(nested_dict, dict):
                for key, value in nested_dict.items():
                    new_key = f"{prefix}_{key}" if prefix else key
                    extract_arrays(value, new_key)
            elif isinstance(nested_dict, list):
                for i, item in enumerate(nested_dict):
                    new_key = f"{prefix}_cell{i}"
                    extract_arrays(item, new_key)
            elif isinstance(nested_dict, np.ndarray):
                flat_data[prefix] = nested_dict.flatten()
            else:
                flat_data[prefix] = np.array([nested_dict])

        extract_arrays(data_dict)
        
        # Save to compressed npz
        np.savez_compressed(npz_filepath, **flat_data)
        print(f"  -> Successfully converted and saved: {npz_filepath}")
        
    except Exception as e:
        print(f"  -> ERROR converting {mat_filepath}: {e}")

# 2. THE BATCH PROCESSING LOGIC
print("Searching for files...")
mat_files = glob.glob('Animal*_Day*.mat')

# Failsafe: Did we actually find anything?
if len(mat_files) == 0:
    print("\nSTOPPING: Found 0 files.")
    print("Please check that this script is in the exact same folder as your .mat files,")
    print("and that the files match the pattern 'AnimalX_DayX.mat' (Case Sensitive!).")
else:
    print(f"Found {len(mat_files)} MAT files to process.\n")
    
    for mat_file in mat_files:
        npz_file = mat_file.replace('.mat', '.npz')
        
        if os.path.exists(npz_file):
            print(f"Skipping {mat_file}, NPZ already exists.")
            continue
            
        print(f"Processing {mat_file}...")
        mat73_to_npz(mat_file, npz_file)

    print("\nBatch conversion complete!")