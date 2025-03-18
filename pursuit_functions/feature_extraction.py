import numpy as np
import pandas as pd

def calculate_curvature(all_regions_data):
    velocity_x = np.gradient(all_regions_data["ratPos_1"])
    velocity_y = np.gradient(all_regions_data["ratPos_2"])
    acceleration_x = np.gradient(velocity_x)
    acceleration_y = np.gradient(velocity_y)

    speed_squared = velocity_x**2 + velocity_y**2
    curvature = np.abs(velocity_x * acceleration_y - velocity_y * acceleration_x) / (speed_squared + 1e-6)**(3/2)

    #avoid issues if there are NaN or infinite values
    curvature[np.isnan(curvature)] = 0
    curvature[np.isinf(curvature)] = 0

    return curvature



def path_efficiency(all_regions_data):
    all_regions_data = all_regions_data.dropna(subset=["ratPos_1", "ratPos_2"])
    all_regions_data = all_regions_data[~np.isinf(all_regions_data["ratPos_1"])]
    all_regions_data = all_regions_data[~np.isinf(all_regions_data["ratPos_2"])]

    if all_regions_data.empty:
        all_regions_data['path_efficiency'] = 0
        return all_regions_data
    
    actual_distance = np.sqrt(np.diff(all_regions_data["ratPos_1"])**2 + np.diff(all_regions_data["ratPos_2"])**2).sum()
    direct_distance = np.sqrt((all_regions_data["ratPos_1"].iloc[-1] - all_regions_data["ratPos_1"].iloc[0])**2 + (all_regions_data["ratPos_2"].iloc[-1] - all_regions_data["ratPos_2"].iloc[0])**2)

    path_efficiency = 0
    if actual_distance > 0:
        path_efficiency = direct_distance / actual_distance
    all_regions_data['path_efficiency'] = path_efficiency
    return all_regions_data

