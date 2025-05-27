import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# drop NA vals 
def drop_NA_vals(dataframe):
    cleaned_data = dataframe.dropna(subset=['ratPos_1', 'ratPos_2', 'laserPos_1', 'laserPos_2'])
    return cleaned_data

#get all coordinate values below 99th percentile and normalize points 

def normalize_points(dataframe, rat_x="ratPos_1", rat_y="ratPos_2", laser_x="laserPos_1", laser_y="laserPos_2"):

    normalized_data = []
    
    for sessFile in dataframe["sessFile"].unique():
        session = dataframe[dataframe["sessFile"] == sessFile].dropna(subset=[rat_x, rat_y, laser_x, laser_y]).copy()

        region = dataframe['region'][0]

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
                "region": region,
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
        region = dataframe['region'][0]

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
            "region": region,
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


def dist_to_bounds(laser_df, center_df, laser_x="laser_x_normalized", laser_y="laser_y_normalized", center_x="center_x", center_y="center_y", radius="radius"):

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

        session["bound_dist"] = np.abs(session["center_dist"] - r)

        session_df.append(session)

    combined_df = pd.concat(session_df, ignore_index=True)
    return combined_df

#put raw spike counts and laser coords into bins calculated from the overall min and max bound_dist values
#function takes laser_spks_bounds dataframes 

def bin_spikes_laser(dataframe, 
                     spk_prefix="spkTable", 
                     dist_col="bound_dist",
                     num_bins=20, 
                     bin_edges=None):
    
    
    if bin_edges is None:
        overall_min = dataframe[dist_col].min()
        overall_max = dataframe[dist_col].max()
        bin_edges = np.linspace(overall_min, overall_max, num_bins+1)

    intervals = pd.IntervalIndex.from_breaks(bin_edges)
    bin_midpoints = (intervals.left + intervals.right) / 2

    bin_mid_lookup = dict(zip(intervals, bin_midpoints))
    
    rows = []

    for sessFile in dataframe["sessFile"].unique():
        
        session = dataframe[dataframe["sessFile"] == sessFile].copy()

        session["bound_bin"] = pd.cut(session[dist_col], bins=intervals, include_lowest=True)

        laser_occupancy = session["bound_bin"].value_counts().reindex(intervals, fill_value=0)

        spk_cols = [col for col in session.columns if spk_prefix in col and not session[col].isna().all()]

        for neuron in spk_cols:
            spks_by_bin = session.groupby("bound_bin", observed=False)[neuron].sum().reindex(intervals, fill_value=0)

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
        values="z_score"
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


    plt.title(f"Z-scored Spike Activity by Boundary Distance")
    plt.xlabel("Boundary Distance (bin midpoint)")
    plt.ylabel("Neurons")
    plt.show()





#count number of neurons by sessFile and region dataframe

def count_neurons(dataframe, spk_prefix="spkTable"):

    # Create an empty dictionary to hold the counts for each session.
    session_neuron_counts = {}

    # Iterate over unique sessions.
    for sess in dataframe["sessFile"].unique():
        # Select only rows for that session.
        session = dataframe[dataframe["sessFile"] == sess]
        
        # Identify spkTable columns that are not entirely NaN for this session.
        spk_cols = [col for col in session.columns 
                    if col.startswith(spk_prefix) and not sess_df[col].isna().all()
        ]
        
        # Store the count for this session.
        session_neuron_counts[sess] = len(spk_cols)

    # Sum the counts over all sessions
    total_neurons = sum(session_neuron_counts.values())

    return session_neuron_counts, total_neurons