import numpy as np
import pandas as pd

def calculate_curvature(dataframe, x_col, y_col):
    velocity_x = np.gradient(dataframe[x_col].astype("float64"))
    velocity_y = np.gradient(dataframe[y_col].astype("float64"))
    acceleration_x = np.gradient(velocity_x)
    acceleration_y = np.gradient(velocity_y)

    speed_squared = velocity_x**2 + velocity_y**2
    curvature = np.abs(velocity_x * acceleration_y - velocity_y * acceleration_x) / (speed_squared + 1e-6)**(3/2)

    #avoid issues if there are NaN or infinite values
    curvature[np.isnan(curvature)] = 0
    curvature[np.isinf(curvature)] = 0

    x = dataframe[x_col].values
    y = dataframe[y_col].values

    dx = np.diff(x)
    dy = np.diff(y)

    displacements = np.sqrt(dx**2 + dy**2)

    path_distance = np.sum(displacements)

    mean_curvature = np.sum(curvature) / path_distance if path_distance > 0 else 0
    
    return mean_curvature


def path_efficiency(dataframe, x_pos, y_pos):

    if dataframe.empty:
        dataframe['path_efficiency'] = 0
        return dataframe
    
    x_positions = dataframe[x_pos].astype("float64")
    y_positions = dataframe[y_pos].astype("float64")
    actual_distance = np.sqrt(np.diff(x_positions)**2 + np.diff(y_positions)**2).sum()
    direct_distance = np.sqrt((x_positions.iloc[-1] - x_positions.iloc[0])**2 + (y_positions.iloc[-1] - y_positions.iloc[0])**2)
    path_efficiency = 0
    if actual_distance > 0:
        path_efficiency = direct_distance / actual_distance
    dataframe.loc[:, 'path_efficiency'] = path_efficiency
    return path_efficiency


def time_to_target(dataframe):
    times = np.array(dataframe['time'])
    time_to_target = times[-1] - times[0]
    return time_to_target


def head_direction_change(dataframe):
    head_direction = np.array(dataframe['laserBearingHD'])
    head_diff = np.diff(head_direction)
    head_diff = np.concatenate(([np.nan], head_diff))
    return head_diff


def movement_direction_change(dataframe):
    movement_direction = np.array(dataframe['laserBearingMD'])
    movement_diff = np.diff(movement_direction)
    movement_diff = np.concatenate(([np.nan], movement_diff))
    return movement_diff


def total_path_distance(dataframe, x_col, y_col):
    
    x = dataframe[x_col].values
    y = dataframe[y_col].values

    dx = np.diff(x)
    dy = np.diff(y)

    total_path_distance = np.sqrt(dx**2 + dy**2).sum()
    return total_path_distance


def mean_squared_value(dataframe, column):

    column_values = dataframe[column].astype("float64").values

    mean_squared_value = np.mean(column_values ** 2)

    return mean_squared_value


def path_ratio(dataframe, rat_x = "ratPos_1", rat_y = "ratPos_2", laser_x = "laserPos_1", laser_y = "laserPos_2"):

    total_rat_path = np.sqrt(np.diff(dataframe[rat_x])**2 + np.diff(dataframe[rat_y])**2).sum()
    total_laser_path = np.sqrt(np.diff(dataframe[laser_x])**2 + np.diff(dataframe[laser_y])**2).sum()

    return total_rat_path / total_laser_path if total_laser_path > 0 else np.nan