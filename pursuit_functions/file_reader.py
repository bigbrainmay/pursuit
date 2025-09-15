from pathlib import Path
import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib as plt

# load region files 
def load_region_files(data_dir: Path, prefix: str):
    region_mats = {}
    for file in data_dir.glob(f"{prefix}*.mat"):
        print(f"Loading: {file}")
        mat_data = sio.loadmat(file, spmatrix=False, squeeze_me=True)

        struct_dict = {}
        for key in mat_data['SL'].dtype.names:
            value = mat_data['SL'][key].squeeze()
            struct_dict[key] = value

        struct_name = file.stem
        region_mats[struct_name] = struct_dict
        
    return region_mats


#load pursuit files
def load_session_files (data_dir: Path, suffix: str = 'pursuitRoot.mat', ignore_keys=None, include_files: list[str] = None):
    if ignore_keys is None:
        ignore_keys = ['ind', 'cells', 'spkTimes', 'Events']
        
    pursuit_matrices = {}

    include_files = include_files or []
    for file in data_dir.glob(f"*{suffix}"):
        if file.name not in include_files:
            print(f"Skipping: {file}")
            continue

        print(f"Loading: {file}")
        try:
            mat_data = sio.loadmat(file, spmatrix=False, squeeze_me=False)
        except:
            print(f"Unable to read {file}, skipping")
            continue
        
        struct_dict = {}
        if 'session' not in mat_data:
            print(f"''session' key missing in {file}, skipping.")
            continue
            
        for key in mat_data['session'].dtype.names:
            if key not in ignore_keys:
                value = mat_data['session'][0,0][key].squeeze()
        
            # Handle different data types
                if isinstance(value, np.ndarray):
                    if value.ndim == 1:
                        value = value.astype(float).tolist()  # Convert 1D int array to float before list conversion
                    elif value.ndim == 2:  # Expand 2D array into multiple columns
                        value = value.astype(float).tolist()  # Convert to a list of lists
                    else:
                        raise ValueError(f"Unsupported array shape {value.shape} for key: {key}")
                else:
                    value = [float(value)] if isinstance(value, int) else [value]  # Convert scalars to float
                
                struct_dict[key] = value  # Store in dictionary
        
        # Find the max length of any value
        max_len = max(len(v) if isinstance(v, list) else 1 for v in struct_dict.values())

    
        # Expand 2D arrays into separate columns dynamically
        expanded_dict = {}
        for key, value in struct_dict.items():
            if isinstance(value[0], list):  # Check if first element is a list (2D structure)
                for i in range(len(value[0])):  # Iterate over columns of 2D array
                    expanded_dict[f"{key}_{i+1}"] = [
                        float(row[i]) if len(row) > i else np.nan for row in value
                    ]
            else:
                # Convert integers to float before padding
                value = np.array(value, dtype=float)
                known_list_keys = ['spkTable']
                if key in known_list_keys:
                    key = f'{key}_1'
                expanded_dict[key] = np.pad(value, (0, max_len - len(value)), constant_values=np.nan)
        
        struct_name = file.name
        pursuit_matrices[struct_name] = expanded_dict
        
    return pursuit_matrices


