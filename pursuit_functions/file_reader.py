# load region files 
from pathlib import Path
import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib as plt

def load_region_files(data_dir: Path, suffix: str):
    region_mats = {}
    for file in data_dir.glob(f"*{suffix}"):
        print(f"Loading: {file}")
        mat_data = sio.loadmat(file, spmatrix=False, squeeze_me=True)

        struct_dict = {}
        for key in mat_data['SL'].dtype.names:
            value = mat_data['SL'][key].squeeze()
            struct_dict[key] = value

        struct_name = file.stem
        region_mats[struct_name] = struct_dict

    return region_mats
#getting the start and end indicies for pursuit trials
def get_first_element(x):
    first_value = x.iloc[0]
    return first_value[0] if isinstance(first_value, (list, tuple, pd.Series)) else first_value
    
