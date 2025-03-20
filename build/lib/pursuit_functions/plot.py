import matplotlib.pyplot as plt



def plot_trajectory(df, ax = None, figsize = None, rat_x_col = 'ratPos_1', rat_y_col = 'ratPos_2', 
                   laser_x_col = 'laserPos_1', laser_y_col = 'laserPos_2', rat_color = 'blue', 
                   laser_color = 'red', legend = False, legend_labels = ['mouse path', 'laser path', 'start','end'],
                   plot_start_end = True):

    '''
    Plots the trajectory of the rat and the laser

    '''
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    
    ax.plot(df[rat_x_col], df[rat_y_col], color = rat_color)
    ax.plot(df[laser_x_col], df[laser_y_col], color = laser_color)

    if plot_start_end:
        ax.scatter(df[rat_x_col].iloc[0], df[rat_y_col].iloc[0], color="green", s=100, zorder=3)
        ax.scatter(df[rat_x_col].iloc[-1], df[rat_y_col].iloc[-1], color="red", s=100, zorder=3)

        ax.scatter(df[laser_x_col].iloc[0], df[laser_y_col].iloc[0], color="green", s=100, zorder=3)
        ax.scatter(df[laser_x_col].iloc[-1], df[laser_y_col].iloc[-1], color="red", s=100, zorder=3)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")  
    ax.legend(labels = legend_labels)