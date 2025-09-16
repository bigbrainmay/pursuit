from pathlib import Path
import scipy.io as sio
import pandas as pd
import numpy as np
import matplotlib as plt
import pyarrow as pa
import re

#unpack block indices
#changes region directory file block indices from list of lists to individual vals
def unpack_block_indices(df):
    blocks = ["FE1", "pursuit", "FE2"]

    for i, blocks in enumerate(blocks):
        df[f"{blocks}_start"] = df["blocks"].apply(lambda x: x[i][0] if len(x) > i else None)
        df[f"{blocks}_end"]   = df["blocks"].apply(lambda x: x[i][1] if len(x) > i else None)
    
    return df


#convert extracted pursuit session file data into dataframes
def convert_sessions_to_dfs(session_files_df):
    session_dfs = {
        filename: pd.DataFrame(file_data).convert_dtypes(dtype_backend="pyarrow")
        for filename, file_data in session_files_df.items()
    }
    return session_dfs 

#filter for valid cells 
def filter_spkTables(region_df, session_dfs):

    filtered_dfs = {}

    grouped = region_df.groupby("sessFile")["cellIndex"].apply(list) 

    for session, df in session_dfs.items():
        if session not in grouped:
            continue
        
        valid_cells = [str(c) for c in grouped[session]]
        
        keep_spk_cols = [
            col for col in df.columns
            if col.startswith("spkTable") and any(re.fullmatch(rf"spkTable_?{cell}", col) for cell in valid_cells)
        ]
        
        non_spk_cols = [c for c in df.columns if not c.startswith("spkTable")]
        keep_cols = non_spk_cols + keep_spk_cols
        filtered_dfs[session] = df[keep_cols].copy()
    
    return filtered_dfs

#build region dfs 
#finds max cell index and pads spkTables so each session df has the same number of columns
#keeps session idx and assigns sessfile names/blocks
def build_region_df(region_df, session_dfs):
    max_cell = region_df["cellIndex"].max()
    print("Highest cellIndex:", max_cell)

    all_spk_cols = [f"spkTable_{i}" for i in range (1, max_cell + 1)]

    dfs = [] 

    for sessFile, df in session_dfs.items():
        df = df.copy()

        for col in all_spk_cols:
            if col not in df.columns:
                df[col] = np.nan

        non_spk_cols = [c for c in df.columns if not c.startswith("spkTable")]
        df = df[non_spk_cols + all_spk_cols]

        
        df["sessIdx"] = df.index
        df["sessFile"] = sessFile

        dfs.append(df)

        session_row = region_df.loc[region_df["sessFile"] == sessFile].iloc[0]

        blocks = np.full(len(df),  np.nan, dtype=object)
        for block_id in ["FE1", "pursuit", "FE2"]:
            start_val = session_row[f"{block_id}_start"]
            end_val = session_row[f"{block_id}_end"]

            if pd.notna(start_val) and pd.notna(end_val):
                start, end = int(start_val), int(end_val)
                mask = (df["sessIdx"] >= start) & (df["sessIdx"] <= end)
                blocks[mask] = block_id

        df["block"] = blocks

        dfs.append(df)

    big_df = pd.concat(dfs, axis=0)
    return big_df

#count neurons to verify that we have the right number of session, spkTable column combos in the region df
def count_neurons(region_df):
    grand_total = 0
    groups = region_df.groupby("sessFile")
    
    for sessFile, group in groups:
        spk_cols = [col for col in group.columns if col.startswith("spkTable")]
        spk_cols = [col for col in spk_cols if group[col].notna().any()]
        spk_count = len(spk_cols)
        print(f"{sessFile}: {spk_count} spkTable columns")
        grand_total += spk_count
    
    print(f"\nGrand total: {grand_total} (session, column) combos")

#save files to parquet
def save_to_parquet(df, filename: str = "output") -> None:

    dtype_map = {
        "time": "float64",
        "sessIdx": "int32",
        "sessFile": "category",
        "block": "category",
    }

    dtype_map.update({col: "Int16" for col in df.columns if col.startswith("spkTable")})

    for col in df.columns:
        if col not in dtype_map and df[col].dtype.kind in ("f", "i"):
            dtype_map[col] = "float64"

    df = df.astype(dtype_map)
    df.to_parquet(f"{filename}.parquet", engine="pyarrow", compression="snappy")
