import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from . import plot
import matplotlib.pyplot as plt

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

    values = dataframe[column].astype("float64").dropna().values

    if len(values) == 0:
        return 0
    mean_squared_value = np.mean(values ** 2)

    return mean_squared_value.astype("float64")


def old_extract_laser_feats(df, id_col = 'pursuit_task_id', columns = ['pursuit_task_id', 'curvature', 'path_efficiency', 'time_to_target', 'total_path_distance', 'path_ratio', 'avsq_laserHD', 'avsq_laserMD', 'avsq_laserMoveDir'],
                            laser_pos_x = 'laserPos_1', laser_pos_y = 'laserPos_2', rat_pos_x = 'ratPos_1', rat_pos_y = 'ratPos_2', laser_bearing_hd = 'laserBearingHD', laser_bearing_md = 'laserBearingMD', laser_move_dir = 'laserMoveDir'):
    features_data = []
    for pursuit_task_id, p_task_df in df.groupby(id_col):
        features_list = [pursuit_task_id,
        calculate_curvature(p_task_df, laser_pos_x, laser_pos_y),
        path_efficiency(p_task_df, rat_pos_x, rat_pos_y),
        time_to_target(p_task_df),
        total_path_distance(p_task_df, laser_pos_x, laser_pos_y),
        path_ratio(p_task_df),
        mean_squared_value(p_task_df, laser_bearing_hd),
        mean_squared_value(p_task_df, laser_bearing_md),
        mean_squared_value(p_task_df, laser_move_dir)
        ]
        features_data.append(features_list)
    return pd.DataFrame(features_data, columns = columns)

def extract_laser_feats(df, id_col = 'pursuit_task_id', columns = ['pursuit_task_id', 'curvature', 'path_efficiency', 'total_path_distance', 'avsq_laserMoveDir', 'avsq_laserJerk'],
                            laser_pos_x = 'laserPos_1', laser_pos_y = 'laserPos_2', rat_pos_x = 'ratPos_1', rat_pos_y = 'ratPos_2', laser_bearing_hd = 'laserBearingHD', laser_bearing_md = 'laserBearingMD', laser_move_dir = 'laserMoveDir'):
    features_data = []
    for pursuit_task_id, p_task_df in df.groupby(id_col):
        features_list = [pursuit_task_id,
        calculate_curvature(p_task_df, laser_pos_x, laser_pos_y),
        path_efficiency(p_task_df, laser_pos_x, laser_pos_y),
        total_path_distance(p_task_df, laser_pos_x, laser_pos_y),
        mean_squared_value(p_task_df, laser_move_dir),
        laser_jerk(p_task_df)
        ]
        features_data.append(features_list)
    return pd.DataFrame(features_data, columns = columns)

# now define rat laser features for shortcut identification model
def path_ratio(df, rat_x = "ratPos_1", rat_y = "ratPos_2", laser_x = "laserPos_1", laser_y = "laserPos_2"):

    '''
    calcuates the efficiency of the path the rat takes compared to the laser path
    '''
    rat_path_dist = total_path_distance(df, rat_x, rat_y)
    laser_path_dist = total_path_distance(df, laser_x, laser_y)
    # if laser_path_dist == 0:
    #     print(f' error in path ratio calc for {df['pursuit_task_id']}')
    if laser_path_dist == 0:
        return 0
    return (rat_path_dist / laser_path_dist).astype("float64") 


def sliding_scale_sum(df, col, window_size = 20):
    '''

    '''
    pass


def laser_jerk(df, laser_acc = 'laserAcc', time_col = 'time', dt = 0.0167,
                     x_col="laserPos_1", y_col="laserPos_2"):
    '''
    calculates the jerk of the laser
    '''
    laser_acc = df[laser_acc].astype("float64")

    # time got mangled
    # time = np.arange(0, len(df) * dt, dt)  # Starts at 0, increments by dt

    jerk = np.gradient(laser_acc, dt)

    jerk[np.isinf(jerk)] = np.nan
    jerk = np.nan_to_num(jerk)  #  nan to 0

    # msj jerk to get rid of sign and higher weight to higher jerk
    msj = np.mean(jerk**2)

    # norm by total path dist 
    path_distance = total_path_distance(df, x_col, y_col)
    normalized_msj = msj / path_distance if path_distance > 0 else msj

    return normalized_msj


# one feature: were a rat was close to the laser, then further, then closer again 

# 

def rat_angle_shifts(df, dist_col = 'laserDist'):
    '''
    measures the times when the rat changes direction abruptly
    '''
    
    # then need to find times when the rat's direction changes abruptly
    


    # now figure out if a time has both of these things happening at the same time

    # could also do when shortest path from rat to laser decreases then increases again
    # or when the direction of the rat vs the direction to the laser diverges, but the 

    pass





def rat_laser_distance_shifts(df, dist_col = 'laserDist', window_size = 20, threshold = 10):

    # first track the distance between the rat and the laser
    rat_laser_dist = df[dist_col].astype("float64")

    # now find times when the distance goes from close to far for a long ish time 
    # this needs to be some sort of sliding threshold that takes the dist at the 
    # previous ~.3 seconds and compares it to the dist at the next ~.3 seconds (.3 seconds is about 20 rows)
    # if the dist increases by a certain amount then we count it as a shift
    # need a sliding scale summation function
    

def gather_groundtruth(pursuit_ids, df):

    labels_dict = {}

    
    for i, pursuit_id in enumerate(pursuit_ids):
        # print amount done
        print(f"{i} out of {len(pursuit_ids)} done")
        pursuit_df = df[df['pursuit_task_id'] == pursuit_id]
        
        # graph it 
        plot.plot_trajectory(pursuit_df)
        plt.show()
        # get the user input
        label = input("is it a line (l), charactaristic (c), random (r) or unknown (u)? ")

        labels_dict[pursuit_id] = label

    return labels_dict




