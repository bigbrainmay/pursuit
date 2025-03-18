import matplotlib.pyplot as plt



def plot_trajectory(df, ax = None, figsize = None, rat_x_col = 'ratPos_1', rat_y_col = 'ratPos_2', 
                   laser_x_col = 'laserPos_1', laser_y_col = 'laserPos_2', rat_color = 'blue', 
                   laser_color = 'red'):

    '''
    Plots the trajectory of the rat and the laser.

    '''
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    
    ax.plot(df[rat_x_col], df[rat_y_col], color = rat_color)
    ax.plot(df[laser_x_col], df[laser_y_col], color = laser_color)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")  