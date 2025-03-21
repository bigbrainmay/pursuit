import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import numpy as np

from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score, auc



def plot_trajectory(df, ax = None, figsize = None, rat_x_col = 'ratPos_1', rat_y_col = 'ratPos_2', 
                   laser_x_col = 'laserPos_1', laser_y_col = 'laserPos_2', rat_color = 'blue', 
                   laser_color = 'red', legend = False, legend_labels = ['rat path', 'laser path', 'start','end'],
                   plot_start_end = True, title = ''):

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
    ax.set_title(title)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")  
    ax.legend(labels = legend_labels)


from IPython.display import HTML

def animate_trajectory(df, time_col='time', rat_x_col='ratPos_1', rat_y_col='ratPos_2', 
                       laser_x_col='laserPos_1', laser_y_col='laserPos_2',
                       rat_color='blue', laser_color='red', speed_factor=1.0):
    """
    Creates an animated plot showing the movement of the rat and laser over time.
    """

    fig, ax = plt.subplots(figsize=(6,6))


    df = df.copy()
    df[[rat_x_col, laser_x_col, rat_y_col, laser_y_col]] = df[[rat_x_col, laser_x_col, rat_y_col, laser_y_col]].astype("float64")

    # Compute time differences 
    time_diffs = np.diff(df[time_col].to_numpy(), prepend=df[time_col].iloc[0])  
    time_diffs = np.clip(time_diffs, 1e-3, None)  

    
    time_diffs = (time_diffs / speed_factor) * 1000  # convert to ms

    
    ax.set_xlim(df[[rat_x_col, laser_x_col]].min().min(), df[[rat_x_col, laser_x_col]].max().max())
    ax.set_ylim(df[[rat_y_col, laser_y_col]].min().min(), df[[rat_y_col, laser_y_col]].max().max())

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Rat and Laser Movement")

    
    rat_traj, = ax.plot([], [], color=rat_color, alpha=0.5, label="Rat Path")
    laser_traj, = ax.plot([], [], color=laser_color, alpha=0.5, label="Laser Path")

   
    rat_marker, = ax.plot([], [], 'o', color=rat_color, markersize=6, label="Rat Position")
    laser_marker, = ax.plot([], [], 'o', color=laser_color, markersize=6, label="Laser Position")

    ax.legend()

    def update(frame):
        """
        Animation update function.
        - Runs in real time based on timestamps.
        """
        if frame >= len(df):
            return rat_traj, laser_traj, rat_marker, laser_marker  

       
        rat_x_vals = df[rat_x_col].iloc[:frame+1].to_numpy()
        rat_y_vals = df[rat_y_col].iloc[:frame+1].to_numpy()
        laser_x_vals = df[laser_x_col].iloc[:frame+1].to_numpy()
        laser_y_vals = df[laser_y_col].iloc[:frame+1].to_numpy()

        rat_traj.set_data(rat_x_vals, rat_y_vals)
        laser_traj.set_data(laser_x_vals, laser_y_vals)

        
        rat_marker.set_data([rat_x_vals[-1]], [rat_y_vals[-1]])
        laser_marker.set_data([laser_x_vals[-1]], [laser_y_vals[-1]])

        return rat_traj, laser_traj, rat_marker, laser_marker

    ani = animation.FuncAnimation(fig, update, frames=len(df), interval=np.mean(time_diffs), blit=False)
    plt.rcParams["animation.embed_limit"] = 10

    return HTML(ani.to_jshtml())  



def plot_pr_curves(models, X, y, class_labels, numeric_label_dict = {0:'Characteristic', 1:'Linear', 2:'Pseudorandom'},
                    palette = 'tab10'):
    '''
    plots precision recall curves for each class for a list of models
    '''
    # first binarize the label because some models need that 
    y_binarized = label_binarize(y, classes=class_labels)
    num_classes = len(class_labels)

    colors = cm.get_cmap(palette, len(models))

    # create a separate plot for each class
    for class_idx, class_label in enumerate(class_labels):
        plt.figure(figsize=(6, 6))
        for model_idx, (model_name, model) in enumerate(models.items()):
            # all models we made have this but just in case 
            if hasattr(model, "predict_proba"):
                y_scores = model.predict_proba(X)[:, class_idx]
            else:
                print('missing predict_proba method')
                return 
            # get precision recall curve
            precision, recall, _ = precision_recall_curve(y_binarized[:, class_idx], y_scores)
            avg_precision = average_precision_score(y_binarized[:, class_idx], y_scores)
            auc_pr = auc(recall, precision)

            # plot the curve
            plt.plot(recall, precision, marker='.', label=f'{model_name} (avg precision: {avg_precision:.2f}, AUC = {auc_pr:.2f}) ', color=colors(model_idx))

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall curve for {numeric_label_dict[class_label]}')
        plt.legend()
