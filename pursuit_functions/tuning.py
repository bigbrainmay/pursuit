import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit
from scipy import stats
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages



# normalize points and find circle boundaries
#the circle boundaries calculated per session will be used later for distance to boundary calculations for each sessFile
#get all coordinate values below 99th percentile and normalize points 

def normalize_points(dataframe, rat_x="ratPos_1", rat_y="ratPos_2", laser_x="laserPos_1", laser_y="laserPos_2"):

    normalized_data = []
    
    for sessFile in dataframe["sessFile"].unique():
        session = dataframe[dataframe["sessFile"] == sessFile].dropna(subset=[rat_x, rat_y, laser_x, laser_y]).copy()

        #region = dataframe['region'][0]

        rat_x_vals = session[rat_x].values.astype("float64")
        rat_y_vals = session[rat_y].values.astype("float64")
        laser_x_vals = session[laser_x].values.astype("float64")
        laser_y_vals = session[laser_y].values.astype("float64")

        x = np.concatenate((rat_x_vals, laser_x_vals))
        y = np.concatenate((rat_y_vals, laser_y_vals))

        #identify 99th percentile x, y boundaries
        x_low, x_high = np.percentile(x, [1, 99])
        y_low, y_high = np.percentile(y, [1, 99])

        #filter the data so we only get the data under the 99th percentile
        filter = (x >= x_low) & (x <= x_high) & (y >= y_low) & (y <= y_high)

        x_filtered = x[filter]
        y_filtered = y[filter]

        #normalize the points to the origin
        x_normalized = x_filtered - x_low
        y_normalized = y_filtered - y_low

        for xn, yn in zip(x_normalized, y_normalized):
            normalized_data.append({
                "sessFile": sessFile,
                #"region": region,
                "x_normalized": xn,
                "y_normalized": yn,
                "x_low": x_low,
                "x_high": x_high, 
                "y_low": y_low, 
                "y_high": y_high
            })
            
    normalized_df = pd.DataFrame(normalized_data)
    return normalized_df

#boundary approach: find center of arena and radius to find boundaries for all normalized data points
#find the mean center and overall radius of the arena for all normalized data points
#you can specify the percentile value to be considered for the overall radius; default is 95th percentile
#calculates the individual center point for each session

def fit_circle_bounds(dataframe, x_norm="x_normalized", y_norm="y_normalized", radius_percentile=95):
    
    results = []

    all_x_vals = dataframe[x_norm].values.astype("float64")
    all_y_vals = dataframe[y_norm].values.astype("float64")

    avg_center_x = np.mean(all_x_vals)
    avg_center_y = np.mean(all_y_vals)

    dist_to_avg_center = np.sqrt((all_x_vals - avg_center_x)**2 + (all_y_vals - avg_center_y)**2)

    overall_radius = np.percentile(dist_to_avg_center, radius_percentile)

    for sessFile in dataframe["sessFile"].unique():
        session = dataframe[dataframe["sessFile"] == sessFile]
        #region = dataframe['region'][0]

        session_x_vals = session[x_norm].values.astype("float64")
        session_y_vals = session[y_norm].values.astype("float64")

        x_min = np.min(session_x_vals)
        x_max = np.max(session_x_vals)
        y_min = np.min(session_y_vals)
        y_max = np.max(session_y_vals)

        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        results.append({
            "sessFile": sessFile,
            #"region": region,
            "center_x": center_x,
            "center_y": center_y,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "radius": overall_radius
        })

    
    center_boundary_vals = pd.DataFrame(results)

    return center_boundary_vals, overall_radius

#find circumference points for plotting
def circumference(dataframe, center_x="center_x", center_y="center_y", radius="radius"):
    
    results = []

    for sessFile in dataframe["sessFile"].unique():
        session = dataframe[dataframe["sessFile"] == sessFile]

        x_val = session[center_x].values.astype("float64")
        y_val = session[center_y].values.astype("float64")
        r = session[radius].values.astype("float64")

        num_points = 360

        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

        x_points = x_val + r * np.cos(angles)
        y_points = y_val + r * np.sin(angles)

        results.append({
            "sessFile": sessFile,
            "x_bounds": x_points,
            "y_bounds": y_points,
            "radius": r
        })
    
    session_circumference_points = pd.DataFrame(results)


    return session_circumference_points

#plot normalized concatenated laser and rat paths with center point and boundary

def plot_arena_bounds(normalized_df, circle_boundaries_df, circumference_df):

    for sessFile in normalized_df["sessFile"].unique():
        plt.figure(figsize=(6,6))

        #plot normalized trajectory
        subset_sessFile = normalized_df[normalized_df["sessFile"] == sessFile]
        plt.plot(subset_sessFile["x_normalized"], subset_sessFile["y_normalized"], color='purple', label=f'Session {sessFile}')
        
        #plot the center point
        subset_center = circle_boundaries_df[circle_boundaries_df["sessFile"] == sessFile]
        plt.scatter(subset_center["center_x"], subset_center["center_y"], color='black', zorder=5)

        #plot the boundary circle
        subset_bounds = circumference_df[circumference_df["sessFile"] == sessFile]
        plt.plot(subset_bounds["x_bounds"].iloc[0], subset_bounds["y_bounds"].iloc[0], 'b-', zorder=10, label='Circle Boundary')
    
        plt.axis('equal')
        plt.title(f"Arena Bounds: {sessFile}")
        plt.legend()
        plt.tight_layout()
        plt.show()

#clean data: after pulling region-specific sessions, drop NA vals 
# drop NA vals 
def drop_NA_vals(dataframe):
    cleaned_data = dataframe.dropna(subset=['ratPos_1', 'ratPos_2', 'laserPos_1', 'laserPos_2'])
    return cleaned_data

#bootstrapping pre-process functions: calculate relative time

# there's some data compression that collapses the 120hz recording time points as the recordings get longer. 
# first we collapse the time points to whole seconds
# we need to normalize the time so that the first observation starts from 0 seconds
# we then calculate the normalized minute each observation belongs to

def normalize_time(dataframe):
    df = dataframe.copy()
    df["rel_time"] = df.groupby("sessFile")["time"].transform(lambda x: x - x.min())
    df["norm_sec"] = df["rel_time"].astype(int)
    df["norm_min"] = df["norm_sec"] // 60
    
    return df


#bootstrapping pre-process functions: assign epochs to dataframes normalized time for bootstrapping and tuning
# # to split each session into epochs, we separate them by first/second half of the recording or odd/even minutes
# for epoch_half, True:1, False:2 will result in the first half labeled as 1 (<= cutoff) and the second half labeled as 2
# for epoch_odd_even, we take the minutes divided by 2 and find the remainder. If the remainder is 1 (odd), it will be labeled as 1 and if the remainder is 0 (even), it will be labeled as 2 

def assign_epochs(dataframe):
    df = dataframe.copy()

    # separate epochs by half

    def label_half(group):
        mins = group["norm_min"].unique()
        mins.sort()
        cutoff = mins[len(mins) // 2]
        return group["norm_min"] <= cutoff
    
    df["epoch_half"] = (
        df.groupby("sessFile", group_keys=False)
        .apply(label_half, include_groups=False)
        .map({True: 1, False: 2})
    )

    # separate epochs by odd/even minutes

    df["epoch_odd_even"] = df["norm_min"] % 2
    df["epoch_odd_even"] = df["epoch_odd_even"].map({1: 1, 0: 2})

    return df

#bootstrapping pre-process functions: make a new df with only sessFile, laser, epoch, spike, and relative time data for bootstrapping.
def epoch_laser_spks(dataframe, laser_x="laserPos_1", laser_y="laserPos_2"):

    epoch_laser_spks_data = []

    spk_columns = [col for col in dataframe.columns if "spkTable" in col]

    time_column = [col for col in dataframe.columns if "rel_time" in col.lower()]
    
    for sessFile in dataframe["sessFile"].unique():

            session = dataframe[dataframe["sessFile"] == sessFile].copy()
            
            laser_x_vals = session[laser_x].astype("float64")
            laser_y_vals = session[laser_y].astype("float64")
    
            #identify 99th percentile x, y boundaries
            x_low, x_high = np.percentile(laser_x_vals, [0, 99])
            y_low, y_high = np.percentile(laser_y_vals, [0, 99])

            #filter the data so we only get the data under the 99th percentile
            filter = (
                (laser_x_vals >= x_low) & (laser_x_vals <= x_high) & 
                (laser_y_vals >= y_low) & (laser_y_vals <= y_high)
            )

            filtered_session = session[filter].copy()

            #normalize the points to the origin
            x_normalized = filtered_session[laser_x].astype("float64") - float(x_low)
            y_normalized = filtered_session[laser_y].astype("float64") - float(y_low)

            #grab epoch data 
            epoch_half = filtered_session["epoch_half"].values
            epoch_odd_even = filtered_session["epoch_odd_even"].values

            #make a dataframe containing normalized data
            normalized_df = pd.DataFrame({
                "sessFile": sessFile,
                "laser_x_normalized": x_normalized.values,
                "laser_y_normalized": y_normalized.values,
                "epoch_half": epoch_half,
                "epoch_odd_even": epoch_odd_even
            })

            #grab spike data using the normalized data mask
            spk_df = filtered_session[spk_columns].reset_index(drop=True)

            #grab time data using the normalized data mask
            time_df = filtered_session[time_column].reset_index(drop=True)

            #make a combined dataframe
            combined_df = pd.concat([normalized_df.reset_index(drop=True), spk_df, time_df], axis=1)

            #append dataframe to the list
            epoch_laser_spks_data.append(combined_df)

        #make a giant dataframe by concatenating all the dataframes in the list        
    epoch_laser_spks_df = pd.concat(epoch_laser_spks_data, ignore_index=True)

    return epoch_laser_spks_df


#bootstrapping pre-process functions: function for pulling epochs from each session along with associated sessFile, laser, and spike data

def pull_epochs(dataframe, 
                spk_prefix="spkTable"):

    epoch_first_half = []
    epoch_second_half = []
    epoch_odd_min = []
    epoch_even_min = []

    for sessFile in dataframe["sessFile"].unique():
        
        session = dataframe[dataframe["sessFile"] == sessFile].copy()

        # mapping first half and second epochs for each session 
        first_half = session[session["epoch_half"] == 1]
        second_half = session[session["epoch_half"] ==2]
        # mapping odd and even epochs for each session
        odd_min = session[session["epoch_odd_even"] == 1]
        even_min = session[session["epoch_odd_even"] == 2]

        spk_cols = [col for col in session.columns if spk_prefix in col and not session[col].isna().all()]

        #function for grabbing sessFile, laser x, laser y, and spk columns for each epoch
        def build_epoch_df(epoch):
            return pd.concat([
                epoch[["sessFile", "laser_x_normalized", "laser_y_normalized", "rel_time"]].reset_index(drop=True),
                epoch[spk_cols].reset_index(drop=True)
            ], axis=1)

        epoch_first_half.append(build_epoch_df(first_half))
        epoch_second_half.append(build_epoch_df(second_half))
        epoch_odd_min.append(build_epoch_df(odd_min))
        epoch_even_min.append(build_epoch_df(even_min))

    return (
        pd.concat(epoch_first_half, ignore_index=True),
        pd.concat(epoch_second_half, ignore_index=True),
        pd.concat(epoch_odd_min, ignore_index=True),
        pd.concat(epoch_even_min, ignore_index=True),
    )
    
#bootstrapping functions: get bin ids and bin edges

# this is making a matrix the shape of the length of distance bin edges and bin assignments
# for the bin assignments it's grabbing the bin id

@njit
def make_assignment_matrix(bin_assignments, dist_bin_edges):
    bin_ids = np.digitize(bin_assignments, dist_bin_edges, right=False) - 1
    bin_mask = (bin_ids >= 0) & (bin_ids < len(dist_bin_edges) -1)

    B = len(dist_bin_edges) - 1
    T = len(bin_assignments)
    M = np.zeros((B, T))
    for t in range(T):
        if bin_mask[t]:
            M[bin_ids[t], t] += 1
    return M

#bootstrapping functions: calculating tuning by using the M bin matrix to grab spike info and get spike occupancy per bin
@njit
def tuning_calc(M, spk_array):
    spk_sum = M @ spk_array # (B, N)
    occupancy = M.sum(axis=1) # (B,)

    tuning = np.zeros_like(spk_sum)

    for b in range(spk_sum.shape[0]):
            if occupancy[b] != 0:
                for n in range(spk_sum.shape[1]):
                    tuning[b, n] = spk_sum[b, n] / occupancy[b]    
    
    return tuning, occupancy

#bootstrapping functions: find timepoints to shift 
def make_shift_idx(rel_time, num_shifts=1000):
    rel_time_np = rel_time.to_numpy().flatten()
    shift_points = np.where(np.diff(np.floor(rel_time_np)) > 0)[0] + 1
    return shift_points[:num_shifts].tolist()

#bootstrapping functions: calculate tuning and get spear correlation for a shift
def process_session(epoch_df1, epoch_df2, center_df, sessFile, spk_cols, num_shifts, rel_time_col="rel_time"):
        
    all_results = []
    valid_spk_cols = []
    for col in spk_cols:
        all_nulls_1 = epoch_df1[col].isna().all() 
        all_nulls_2 = epoch_df2[col].isna().all() 

        if not all_nulls_1 and not all_nulls_2:
            valid_spk_cols.append(col)

    # calculate distance to bounds
    dist1 = dist_to_bounds(epoch_df1, center_df)["bound_dist"].values
    dist2 = dist_to_bounds(epoch_df2, center_df)["bound_dist"].values

    # get spike arrays
    spk1 = epoch_df1[valid_spk_cols].to_numpy(dtype=np.float64)
    spk2 = epoch_df2[valid_spk_cols].to_numpy(dtype=np.float64)

    # get bin edges
    dist_bin_edges = np.linspace(min(dist1.min(), dist2.min()),
                                max(dist1.max(), dist2.max()), 21)                                   

    # make bin assignment matrices 
    M1 = make_assignment_matrix(dist1, dist_bin_edges)
    M2 = make_assignment_matrix(dist2, dist_bin_edges)

    # calculate tuning
    tune1, _ = tuning_calc(M1, spk1)
    tune2, _ = tuning_calc(M2, spk2)

    # spearman corr- iter: 0 (True corr)
    for j, ncol in enumerate(valid_spk_cols):
        r, p = stats.spearmanr(tune1[:, j], tune2[:, j])
        all_results.append({
            "sessFile": sessFile,
            "neuron": ncol,
            "bootstrap_iter": 0,
            "spearman_r": float(r),
            "p_val": float(p)
        })

    #bootstrapping with shift indices from rel_time

    rel_time = epoch_df1[rel_time_col]
    shift_points = make_shift_idx(rel_time, num_shifts=num_shifts)

    for i, shift_idx in enumerate(shift_points, start=1):
        spk1_shifted = np.roll(spk1, shift_idx, axis=0) 
        tune1_rolled, _ = tuning_calc(M1, spk1_shifted)

        for j, ncol in enumerate(valid_spk_cols):
            r, p = stats.spearmanr(tune1_rolled[:, j], tune2[:, j])
            all_results.append({
                "sessFile": sessFile,
                "neuron": ncol,
                "bootstrap_iter": i,
                "spearman_r": float(r),
                "p_val": float(p)
                })

    return all_results

#bootstrapping functions: do all shifts and calculate tuning
def bootstrap_all_sessions(epoch_df1, epoch_df2, center_df, spk_prefix="spkTable", rel_time_col="rel_time", num_shifts=1000):
    spk_cols = [col for col in epoch_df1.columns if spk_prefix in col]
    sessions = epoch_df1["sessFile"].unique()

    all_bootstrap_results = []

    for sessFile in tqdm(sessions, desc="Sessions"):
        df1 = epoch_df1[epoch_df1["sessFile"] == sessFile]
        df2 = epoch_df2[epoch_df2["sessFile"] == sessFile]

        results = process_session(df1, df2, center_df, sessFile=sessFile, spk_cols=spk_cols, num_shifts=num_shifts, rel_time_col=rel_time_col)
        all_bootstrap_results.extend(results)

    return pd.DataFrame(all_bootstrap_results)


#identify significant cells!

# find cells with corr values over the 95th and 99th percentiles and compare them to the "true" tuning
def plot_null_dist(df, neurons=None, percentile_lines=(95, 99), max_neurons=None):
    grouped = df.groupby(["sessFile", "neuron"])

    if neurons is None:
        neurons = list(grouped.groups.keys())
        if max_neurons is not None:
            neurons = neurons[:max_neurons]

    for sessFile, neuron in neurons:
        sub_df = grouped.get_group((sessFile, neuron))

        null_df = sub_df[sub_df["bootstrap_iter"] > 0]
        true_df = sub_df[sub_df["bootstrap_iter"] == 0]
        
        fig, ax = plt.subplots(figsize=(8,5))

        sns.rugplot(
            x=null_df["spearman_r"],
            height=0.05,
            hue=null_df["bootstrap_iter"],
            ax=ax,
            lw=0.05
        )

        sns.histplot(
            null_df["spearman_r"],
            bins = 30,
            stat="count",
            kde=False,
            color='lightblue',
            edgecolor='white',
            ax=ax,
            alpha=0.6
        )

        true_r = true_df["spearman_r"].values[0]
        ax.axvline(true_r, color='red', linestyle='--', label=f"True r ={true_r:.3f}")

        for p in percentile_lines:
            threshold = np.percentile(null_df["spearman_r"], p)
            ax.axvline(threshold, color='gray', linestyle=':', label=f"{p}th ={threshold:.3f}")

        ax.set_title(f"Null dist — {sessFile} / {neuron}")
        ax.set_xlabel("Spearman r")
        ax.set_ylabel("Bootstrap count")
        ax.legend()
        plt.tight_layout()
        plt.show()

# identify significant cells: we're going to get only the significant cells now

def get_significant_cells(df, percentile=95):
    significant_cells = []

    grouped = df.groupby(["sessFile", "neuron"])

    for (sessFile, neuron), group in grouped:
        true_r = group.loc[group["bootstrap_iter"] == 0, "spearman_r"].values
        true_r = true_r[0]

        boot_r = group.loc[group["bootstrap_iter"] > 0, "spearman_r"]
        threshold = np.percentile(boot_r, percentile)

        if true_r > threshold:
            significant_cells.append((sessFile, neuron))

    return significant_cells    

#plot heatmap tuning curves for significant cells
# returns df_sorted (peak sorted data ready to plot in heatmap) and df_binned (bins, spk_ct, laser_occ, tuning)
def sig_cells_heatmap(raw_data, center_df, sig_cells_list, percentile=95,
                                 smoothing_window=3, smoothing_std=1, exclude_r=0, plot_title=None):
    
    df_sig_cells = pd.DataFrame(sig_cells_list, columns=["sessFile", "neuron"])

   
    df_laser_spks = norm_laser_get_spks(raw_data)

   
    df_bounds = dist_to_bounds(df_laser_spks, center_df, exclude_r=exclude_r)

    
    df_binned = bin_spikes_laser(df_bounds, center_df)

    
    df_tuning = calculate_tuning(df_binned)

   
    df_tuning_filtered = df_tuning.merge(df_sig_cells, on=["sessFile", "neuron"])

   
    #df_z = z_score_norm(df_tuning_filtered)
    df_max_norm = max_norm_score(df_tuning_filtered)
   
    #df_smoothed = pivot_smooth(df_z, window_size=smoothing_window, std=smoothing_std)
    df_smoothed = pivot_smooth(df_max_norm, window_size=smoothing_window, std=smoothing_std)

   
    df_sorted = peak_sort(df_smoothed)

   
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_sorted, cmap="viridis", annot=False, fmt=".2f", yticklabels=False)
    plt.title(plot_title or f"Max-scored Spike Activity (> {percentile}th percentile)")
    #plt.xlabel("Boundary Distance (bin midpoint)")
    plt.xlabel("Distance to Center (bin midpoints)")
    plt.ylabel("Neurons (peak sorted)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return df_sorted, df_binned  


#plot individual significant cell tuning curves
def plot_sig_tuning_curves(raw_data, center_df, sig_cells_list, percentile=95,
                                 smoothing_window=3, smoothing_std=1, exclude_r=0, plot_title=None):
    
    df_sig_cells = pd.DataFrame(sig_cells_list, columns=["sessFile", "neuron"])

   
    df_laser_spks = norm_laser_get_spks(raw_data)

   
    df_bounds = dist_to_bounds(df_laser_spks, center_df, exclude_r=exclude_r)

    
    df_binned = bin_spikes_laser(df_bounds, center_df)

    
    df_tuning = calculate_tuning(df_binned)

    
    df_tuning_filtered = df_tuning.merge(df_sig_cells, on=["sessFile", "neuron"])


    grouped = df_tuning_filtered.groupby(["sessFile", "neuron"])

    for (sessFile, neuron), sub_df in grouped:
        
        pivoted = (sub_df.pivot(index="neuron", columns="bin_midpoint", values="tuning").fillna(0))

        for neuron in pivoted.index:
            plt.plot(pivoted.columns, pivoted.loc[neuron], marker='o', linestyle='-', label=f"{neuron}")

        #plt.xlabel("Boundary Distance (bin midpoint)")
        plt.xlabel("Distance to Center (bin midpoints)")
        plt.ylabel("Tuning (spike count / laser occupancy)")
        plt.title(f"Tuning Curve — {sessFile} / {neuron}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()




#helper functions for calculating tuning below

#normalize laser points and make a dataframe containing spike data using the normalized data mask
def norm_laser_get_spks(dataframe, laser_x="laserPos_1", laser_y="laserPos_2"):

    normalized_laser_data = []

    spk_columns = [col for col in dataframe.columns if "spkTable" in col]
    
    for sessFile in dataframe["sessFile"].unique():

        session = dataframe[dataframe["sessFile"] == sessFile].copy()

        laser_x_vals = session[laser_x].astype("float64")
        laser_y_vals = session[laser_y].astype("float64")
   
        #identify 99th percentile x, y boundaries
        x_low, x_high = np.percentile(laser_x_vals, [0, 99])
        y_low, y_high = np.percentile(laser_y_vals, [0, 99])

        #filter the data so we only get the data under the 99th percentile
        filter = (
            (laser_x_vals >= x_low) & (laser_x_vals <= x_high) & 
            (laser_y_vals >= y_low) & (laser_y_vals <= y_high)
        )

        filtered_session = session[filter].copy()

        #normalize the points to the origin
        x_normalized = filtered_session[laser_x].astype("float64") - float(x_low)
        y_normalized = filtered_session[laser_y].astype("float64") - float(y_low)

        #make a dataframe containing normalized data
        normalized_df = pd.DataFrame({
            "sessFile": sessFile,
            "laser_x_normalized": x_normalized.values,
            "laser_y_normalized": y_normalized.values
        })

        #make a dataframe containing spike data using the normalized data mask
        spk_df = filtered_session[spk_columns].reset_index(drop=True)

        #make a combined dataframe
        combined_df = pd.concat([normalized_df.reset_index(drop=True), spk_df], axis=1)

        #append dataframe to the list
        normalized_laser_data.append(combined_df)

    #make a giant dataframe by concatenating all the dataframes in the list        
    laser_spks_df = pd.concat(normalized_laser_data, ignore_index=True)

    return laser_spks_df

#finds laser x,y coordinates distance to boundary per session
def dist_to_bounds(laser_df, center_df, laser_x="laser_x_normalized", laser_y="laser_y_normalized", center_x="center_x", center_y="center_y", radius="radius", exclude_r=0):

    session_df = [] 
    
    for sessFile in laser_df["sessFile"].unique():
        session = laser_df[laser_df["sessFile"] == sessFile].copy()

        laser_x_val = session[laser_x].values.astype("float64")
        laser_y_val = session[laser_y].values.astype("float64")

        subset_center = center_df[center_df["sessFile"] == sessFile]

        center_x_val = subset_center[center_x].iloc[0]
        center_y_val = subset_center[center_y].iloc[0]
        r = subset_center[radius].iloc[0]

        session["center_dist"] = np.sqrt((laser_x_val - center_x_val)**2 + (laser_y_val - center_y_val)**2)

        if exclude_r > 0:
            session = session[session["center_dist"] >= exclude_r]

        session["bound_dist"] = np.abs(session["center_dist"] - r)

        session_df.append(session)

    combined_df = pd.concat(session_df, ignore_index=True)
    return combined_df

#put raw spike counts and laser coords into bins calculated from the overall min and max bound_dist values
#function takes laser_spks_bounds dataframes 

def bin_spikes_laser(dist_df,
                     center_df,
                     spk_prefix="spkTable", 
                     #bound_dist_col="bound_dist",
                     radius="radius",
                     center_dist_col="center_dist",
                     num_bins=20, 
                     bin_edges=None):
    
    
    if bin_edges is None:
        overall_min = dist_df[center_dist_col].min()
        overall_max = center_df[radius].iloc[0]
        bin_edges = np.linspace(overall_min, overall_max, num_bins+1)

    intervals = pd.IntervalIndex.from_breaks(bin_edges)
    bin_midpoints = (intervals.left + intervals.right) / 2

    bin_mid_lookup = dict(zip(intervals, bin_midpoints))
    
    rows = []

    for sessFile in dist_df["sessFile"].unique():
        
        session = dist_df[dist_df["sessFile"] == sessFile].copy()

        #session["bound_bin"] = pd.cut(session[dist_col], bins=intervals, include_lowest=True)
        session["center_bin"] = pd.cut(session[center_dist_col], bins=intervals, include_lowest=True)

        #laser_occupancy = session["bound_bin"].value_counts().reindex(intervals, fill_value=0)
        laser_occupancy = session["center_bin"].value_counts().reindex(intervals, fill_value=0)

        spk_cols = [col for col in session.columns if spk_prefix in col and not session[col].isna().all()]

        for neuron in spk_cols:
            #spks_by_bin = session.groupby("bound_bin", observed=False)[neuron].sum().reindex(intervals, fill_value=0)
            spks_by_bin = session.groupby("center_bin", observed=False)[neuron].sum().reindex(intervals, fill_value=0)

            for i in intervals:
                rows.append({
                    "sessFile": sessFile,
                    "neuron": neuron,
                    "bin_midpoint": round(bin_mid_lookup[i], 2),
                    "spike_count": int(spks_by_bin[i]),
                    "laser_occupancy": int(laser_occupancy[i])
                })

    return pd.DataFrame(rows)

#normalize spike counts by laser occupancy using bins calculated from the overall min and max bound_dist values 

def calculate_tuning(laser_spikes_binned_df):

    laser_spikes_binned_df["tuning"] = laser_spikes_binned_df["spike_count"] / laser_spikes_binned_df["laser_occupancy"]

    return laser_spikes_binned_df

#plotting function for tuning curves

def plot_tuning_curves(dataframe):
    
    plt.figure(figsize=(12,8))

    for sessFile in dataframe["sessFile"].unique():

        session = dataframe[dataframe["sessFile"] == sessFile]
        
        pivoted = (session.pivot(index="neuron", columns="bin_midpoint", values="tuning").fillna(0))

        for neuron in pivoted.index:
            plt.plot(pivoted.columns, pivoted.loc[neuron], marker='o', linestyle='-', label=f"{neuron}")

    plt.xlabel("Boundary Distance (bin midpoint)")
    plt.ylabel("Tuning (spike count / laser occupancy)")
    plt.title(f"All Neuron Tuning Curves")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#max norm binned normalized data
def max_norm_score(dataframe):

    sessions = dataframe.copy()

    def max_norm(x):
        x_max = x.max()
        x_norm = x / x_max
        return x_norm
    
    sessions["max_norm_score"] = (
        sessions.groupby(["sessFile", "neuron"])["tuning"].transform(max_norm)
    )

    return sessions

#z-score binned normalized data

def z_score_norm(dataframe):
    
    sessions = dataframe.copy()
        
    def norm_z(x):
        std = x.std()
        mean = x.mean()
        z = (x - mean) / std if std > 0 else x*0

        z_min, z_max = z.min(), z.max()
        if z_max > z_min:
            return (z - z_min) / (z_max - z_min)
        else:
            return z * 0

        
    sessions["z_score"] = (
        sessions.groupby(["sessFile", "neuron"])["tuning"].transform(norm_z)
    )

    return sessions

#pivot table and apply gaussian smoothing for plotting

def pivot_smooth(dataframe, window_size=3, window_type='gaussian', std=1):

    pivoted_df = dataframe.pivot_table(
        index=["sessFile", "neuron"], 
        columns="bin_midpoint", 
        values="max_norm_score"
        #values="z_score"
        ).fillna(0)

    smoothed_df = pivoted_df.apply(
        lambda row: row.rolling(
            window=window_size, 
            win_type=window_type, 
            center=True).mean(std=std), 
            axis=1
        ).fillna(0)
    
    return smoothed_df

#peak sort the neurons

def peak_sort(smoothed_df):

    peak_bin = smoothed_df.idxmax(axis=1)

    ordered_index = peak_bin.sort_values().index

    return smoothed_df.loc[ordered_index]

#plot heatmap for z-scored data

def heatmap(dataframe):

    plt.figure(figsize=(12,8))

    heatmap = sns.heatmap(dataframe, cmap="viridis", annot=False, fmt=".2f", yticklabels=False)

    x_labels = heatmap.get_xticklabels()

    rounded_labels = [f"{float(label.get_text()):.2f}" if label.get_text() != "" else "" for label in x_labels]

    heatmap.set_xticklabels(rounded_labels, rotation=45, ha="right")


    plt.title(f"Max-scored Spike Activity by Boundary Distance")
    plt.xlabel("Boundary Distance (bin midpoint)")
    plt.ylabel("Neurons")
    plt.show()





#for sanity checks

#plot at total laser occupancy per bin per session (and save to pdf bc there's a lot of plots)
def occ_dist_to_pdf(df, output_path = "RSC_sessions_laser_occ.pdf"):
    
    grouped = df.groupby(["sessFile", "bin_midpoint"], observed=True)["laser_occupancy"].first().reset_index()

    occ_stats = (
        grouped.groupby("bin_midpoint", observed=True)["laser_occupancy"]
        .agg(["mean"])
        .reset_index()
        .rename(columns={"mean": "mean_occ"})
    )

    merged = grouped.merge(occ_stats, on="bin_midpoint", how="left")

    with PdfPages(output_path) as pdf:

        for sessFile, group in merged.groupby("sessFile"):
            fig, ax = plt.subplots(figsize=(10, 6))

            sns.barplot(
                x=group["bin_midpoint"].astype(str),
                y=group["laser_occupancy"],
                ax=ax,
                color='lightblue',
                edgecolor='black',
                label="Session Laser Occupancy"
            )

            ax.errorbar(
                x=np.arange(len(group)),
                y=group["mean_occ"],
                fmt='o',
                color='black',
                capsize=4,
                label="Global Mean"
            )

            ax.set_title(f"Laser Occupancy by Bin - {sessFile}")
            ax.set_xlabel("Boundary Distance (bin midpoint)")
            ax.set_ylabel("Laser Occupancy")
            plt.xticks(rotation=45, ha="right")
            ax.legend()
            plt.tight_layout()
            #plt.show()
            
            pdf.savefig(fig)
            plt.close(fig)


#count neurons in dataframes pre-bootstrapping/tuning
def count_neurons(dataframe, spk_prefix="spkTable"):

    # Create an empty dictionary to hold the counts for each session.
    session_neuron_counts = {}

    # Iterate over unique sessions.
    for sess in dataframe["sessFile"].unique():
        # Select only rows for that session.
        session = dataframe[dataframe["sessFile"] == sess]
        
        # Identify spkTable columns that are not entirely NaN for this session.
        spk_cols = [col for col in session.columns 
                    if col.startswith(spk_prefix) and not session[col].isna().all()
        ]
        
        # Store the count for this session.
        session_neuron_counts[sess] = len(spk_cols)

    # Sum the counts over all sessions
    total_neurons = sum(session_neuron_counts.values())

    return session_neuron_counts, total_neurons

#count neurons in dataframe post-bootstrapping/tuning
def count_bootstrap_neurons(dataframe, spk_prefix="spkTable"):

    group = dataframe.groupby(["sessFile", "neuron"])

    total_neurons = group.size()

    return total_neurons

#count the cell count differences between each region's epoch
def find_diff_cells(list1, list2):
    first_second = set(list1)
    odd_even = set(list2)

    union = first_second | odd_even
    first_second_only = first_second - odd_even
    odd_even_only = odd_even - first_second

    return len(union), len(first_second_only), len(odd_even_only)

#find the tuning value mean for each distance bin. takes the df_sorted df from the heatmap function
def bin_means(df):
    float_cols = [col for col in df.columns if isinstance(col, float)]
    return df[float_cols].mean(axis=0)

#plot the bin means after you find them. takes an array you get from def bin means.
def plot_bin_means(means, title=None):
    plt.plot(means.index, means.values, marker='o')
    plt.xlabel("Bins")
    plt.ylabel("Mean value")
    plt.title(title)
    plt.grid(True)
    plt.show()

#plot peaks by bin
def plot_peaks_by_bin(df1, df2, title=str):

    df1_bin_cols = [col for col in df1.columns if isinstance(col, float)]
    df2_bin_cols = [col for col in df2.columns if isinstance(col, float)]

    rebinned_df1 = (
        df1[df1_bin_cols]
        .groupby(np.arange(len(df1_bin_cols)) // 2, axis=1)
        .sum()
        )
    
    rebinned_df2 = (
        df2[df2_bin_cols]
        .groupby(np.arange(len(df2_bin_cols)) // 2, axis=1)
        .sum()
    )
    
    new_bin_names = [
        np.mean(df1_bin_cols[i*2:(i+1)*2])
        for i in range(rebinned_df1.shape[1])
    ]

    rebinned_df1.columns = new_bin_names
    rebinned_df2.columns = new_bin_names

    df1_peak_bins = rebinned_df1.idxmax(axis=1)
    df2_peak_bins = rebinned_df2.idxmax(axis=1)

    df1_peak_counts = df1_peak_bins.value_counts().sort_index()
    df2_peak_counts = df2_peak_bins.value_counts().sort_index()

    df1_peak_counts = df1_peak_counts.reindex(new_bin_names, fill_value=0)
    df2_peak_counts = df2_peak_counts.reindex(new_bin_names, fill_value=0)

    df1_peak_counts = df1_peak_counts / len(df1)
    df2_peak_counts = df2_peak_counts / len(df2)


    fig, ax = plt.subplots()
    width=np.diff(sorted(new_bin_names)).mean()

    ax.bar(df1_peak_counts.index, df1_peak_counts.values,
           width=width, color="#8068A3", label="Peak bins (PPC)")
    
    ax.plot(df2_peak_counts.index, df2_peak_counts.values,
            color="#8BA368", marker="o", label="Peak bins (RSC)")

    ax.set_xlabel("Bin midpoint")
    ax.set_ylabel("Proportion of Neuron Peaks/Bin")
    ax.legend()
    ax.set_title(title)
    plt.show()

    return df1_peak_counts, df2_peak_counts

# find unique time entries
def unique_time_entries(dataframe):
    all_sessions = []

    for sessFile in dataframe["sessFile"].unique():
        session = dataframe[dataframe["sessFile"] == sessFile]
        session_times = session["time"]

        times_unique = session_times.astype("float64").nunique()
        times_count = session_times.astype("float64").value_counts().to_dict()

        all_sessions.append({
            "sessFile": sessFile,
            "times_unique": times_unique,
            "unique_times_count": times_count
        })

    return pd.DataFrame(all_sessions)