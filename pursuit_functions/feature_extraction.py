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
    actual_distance = total_path_distance(dataframe, x_pos, y_pos)
    direct_distance = np.sqrt((x_positions.iloc[-1] - x_positions.iloc[0])**2 + (y_positions.iloc[-1] - y_positions.iloc[0])**2)
    path_efficiency = 0
    if actual_distance > 0:
        path_efficiency = direct_distance / actual_distance
    dataframe.loc[:, 'path_efficiency'] = path_efficiency
    return path_efficiency


def time_to_target(dataframe):
    times = np.array(dataframe['time'])
    time_to_target = times[-1] - times[0]
    return time_to_target.astype("float64")


def total_path_distance(dataframe, x_col, y_col):
    
    x = dataframe[x_col].values
    y = dataframe[y_col].values

    dx = np.diff(x)
    dy = np.diff(y)

    total_path_distance = np.sqrt(dx**2 + dy**2).sum()
    return total_path_distance.astype("float64")


def mean_squared_value(dataframe, column):

    column_values = dataframe[column].astype("float64").values

    mean_squared_value = np.mean(column_values ** 2)

    return mean_squared_value.astype("float64")

# now define rat laser features for shortcut identification model
def path_ratio(dataframe, rat_x = "ratPos_1", rat_y = "ratPos_2", laser_x = "laserPos_1", laser_y = "laserPos_2"):

    '''
    calcuates the efficiency of the path the rat takes compared to the laser path
    '''
    rat_path_dist = total_path_distance(df, rat_x, rat_y)
    laser_path_dist = total_path_distance(df, laser_x, laser_y)
    return (rat_path_dist / laser_path_dist).astype("float64")

def rat_angle_shifts():
    '''
    identifies instances when the rat abruptly changes direction 
    '''
    
    
    pass


