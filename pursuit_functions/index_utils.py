import pandas as pd
import numpy as np

#getting the start and end indices for pursuit trials
def get_first_element(x):
    first_value = x.iloc[0]
    return first_value[0] if isinstance(first_value, (list, tuple, pd.Series)) else first_value
    

#function for indexing values
def get_block_rows(start, end, max_length):
    if np.isnan(start) or np.isnan(end):
        return None, None
    start, end = int(start), int(end)
    start = max(0, min(start, max_length - 1))
    end = max(start, min(end, max_length - 1))
    return start, end
