import argparse
import math
import numpy as np

from . import load_all
from . import show
from . import Reader
from . import metrics
from .interactions import *
import matplotlib.pyplot as plt
from . import kalman

def non_linear(scene):
    primary_prediction, _ = kalman.predict(scene)[0] 
    score = metrics.final_l2(scene[0], primary_prediction)
    return score > 2.0

def scene_plots(input_file, args):
    n_int = 0
    reader = Reader(input_file, scene_type='paths')
    scenes = [s for _, s in reader.scenes()]
    for scene in scenes:
        rows = reader.paths_to_xy(scene)
    # for primary_ped, rows in load_all(input_file):
        path = rows[:, 0]
        neigh_path = rows[:, 1:]
        interaction_matrix = get_interaction_matrix(rows, args, output='matrix')
        # "Shape": PredictionLength x Number of Neighbours
        interaction_index = interaction_length(interaction_matrix, length=5)
        neigh = neigh_path[:,interaction_index]
        ## n Examples of interactions ##
        if (np.sum(interaction_index) == 1) & (np.linalg.norm(path[-1] - path[0]) > 4.0):
            if non_linear(scene):    
                n_int += 1  
                if (n_int < args.n):
                    with show.interaction_path(path, neigh):
                        pass
                #     with show.interaction_path(path, neigh_path):
                #         pass
                    # show.makeDynamicPlot(rows.transpose(1, 0, 2), np=True)
    print("Number of Instances: ", n_int) 

def interaction_length(interaction_matrix, length=1):
    interaction_sum = np.sum(interaction_matrix, axis=0)
    return interaction_sum >= length

def distribution_plots(input_file, args):
    ## Distributions of interactions
    n_theta, vr_n, dist_thresh, choice = args.n_theta, args.vr_n, args.dist_thresh, args.choice
    distr = np.zeros((n_theta, vr_n))
    def fill_grid(theta_vr):
        theta, vr, sign = theta_vr
        theta = theta*(2*np.pi)/360
        # theta[np.where(sign < 0)] = (2*np.pi - 1/n_theta) - theta[np.where(sign < 0)]
        thetap = np.floor(theta * distr.shape[0] / (2*np.pi)).astype(int)
        vrp = np.floor(vr * distr.shape[1] / dist_thresh).astype(int)
        distr[thetap, vrp] += 1
    
    unbinned_vr = [[] for _ in range(n_theta)]
    def fill_unbinned_vr(theta_vr):
        theta, vr, sign = theta_vr
        theta = theta*(2*np.pi)/360
        # theta[np.where(sign < 0)] = (2*np.pi - 1/n_theta) - theta[np.where(sign < 0)]
        thetap = np.floor(theta * len(unbinned_vr) / (2*np.pi)).astype(int)
        for th in range(len(thetap)):
            unbinned_vr[thetap[th]].append(vr[th])
    vr_max = dist_thresh

    hist = []
    def fill_hist(vel):
        hist.append(vel)

    #run
    for primary_ped, rows in load_all(input_file):
        interaction_matrix, chosen_true, sign_true, dist_true = \
        get_interaction_matrix(rows, args)

        fill_grid((chosen_true, dist_true, sign_true))
        fill_unbinned_vr((chosen_true, dist_true, sign_true)) 
        fill_hist(chosen_true)
        # if np.any(chosen_true):     
        #     print("MAX, MIN: ", max(chosen_true), min(chosen_true))
        #     print("Shape: ", chosen_true.shape)
    with show.canvas(input_file + '.' + choice + '.png', figsize=(4, 4), subplot_kw={'polar': True}) as ax:
        r_edges = np.linspace(0, vr_max, distr.shape[1] + 1)
        theta_edges = np.linspace(0, 2*np.pi, distr.shape[0] + 1)
        thetas, rs = np.meshgrid(theta_edges, r_edges)
        ax.pcolormesh(thetas, rs, distr.T, vmin=0, vmax=None, cmap='Blues')

        median_vr = np.array([np.median(vrs) if len(vrs) > 5 else np.nan
                              for vrs in unbinned_vr])
        center_thetas = np.linspace(0.0, 2*np.pi, len(median_vr) + 1)
        center_thetas = 0.5 * (center_thetas[:-1] + center_thetas[1:])
        # close loop
        center_thetas = np.hstack([center_thetas, center_thetas[0:1]])
        median_vr = np.hstack([median_vr, median_vr[0:1]])
        # plot median radial velocity
        # ax.plot(center_thetas, median_vr, label='median $d_r$ [m/s]', color='orange')

        ax.grid(linestyle='dotted')
        ax.legend()    

    with show.canvas(input_file + '.' + choice + '_hist.png', figsize=(4, 4)) as ax:
        ax.hist(np.hstack(hist), bins=n_theta)

def group_plots(input_file, args, dist_thresh=0.8, std_thresh=0.1):
    ## Identify and Visualize Groups
    ## dist_thresh: Distance threshold to be withinin a group
    ## std_thresh: Std deviation threshold for variation of distance
    
    n_groups = 0
    n_statn_groups = 0
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for primary_ped, rows in load_all(input_file):
        path, group, flag = check_group(rows, args, dist_thresh, std_thresh)        
        if flag:
            n_groups += 1
            if np.linalg.norm(path[-1] - path[0]) < 1.0:
                n_statn_groups += 1
            if n_groups < args.n:
                with show.group_path(path, group):
                    pass

    print("Number of Groups: ", n_groups)
    print("Number of Stationary Groups: ", n_statn_groups)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_files', nargs='+',
                        help='Trajnet dataset file(s).')
    parser.add_argument('--pos_angle', type=int, default=90,
                        help='axis angle of position cone (in deg)')
    parser.add_argument('--vel_angle', type=int, default=0,
                        help='relative velocity centre (in deg)')
    parser.add_argument('--pos_range', type=int, default=45,
                        help='range of position cone (in deg)')
    parser.add_argument('--vel_range', type=int, default=20,
                        help='relative velocity span (in rsdeg)')
    parser.add_argument('--dist_thresh', type=int, default=4,
                        help='threshold of distance (in m)')
    parser.add_argument('--n_theta', type=int, default=72,
                        help='number of segments in polar plot radially')
    parser.add_argument('--vr_n', type=int, default=10,
                        help='number of segments in polar plot linearly')
    parser.add_argument('--choice', default='pos',
                        help='choice of interaction')
    parser.add_argument('--n', type=int, default=10,
                        help='number of plots')
    args = parser.parse_args()

    print('{dataset:>60s} |     N'.format(dataset=''))
    for dataset_file in args.dataset_files:
        print('{dataset:>60s} | {N:>5}'.format(
            dataset=dataset_file,
            N=sum(1 for _ in load_all(dataset_file)),
        ))

    for dataset_file in args.dataset_files:
        # pass

        ## Interaction
        # scene_plots(dataset_file, args)

        ## Position Global 
        # distribution_plots(dataset_file, args) 

        ## Grouping
        group_plots(dataset_file, args)

if __name__ == '__main__':
    main()

# # Positions for Agents moving in same direction
# dataset_plots(dataset_file, pos_angle, pos_range, 180, 10, dist_thresh, n_theta, vr_max, vr_n, 'bothpos')

# # Positions for Agents moving in opposite direction
# dataset_plots(dataset_file, pos_angle, pos_range, 0, 10, dist_thresh, n_theta, vr_max, vr_n, 'bothpos')

# # Velocity Global
# dataset_plots(dataset_file, pos_angle, pos_range, vel_angle, vel_range, dist_thresh, n_theta, vr_max, vr_n, 'vel')

# # Velocity for Agents in Front
# dataset_plots(dataset_file, 0, 10, vel_angle, vel_range, dist_thresh, n_theta, vr_max, vr_n, 'bothvel')

    # for vel_angle in [0, 180]:
    #     print("VEL Angle: ", vel_angle)
    #     for dist_thresh in range(1, 5):
    #         print("Dist Thresh:", dist_thresh)
    #         multimodality_plot(dataset_file, args)


