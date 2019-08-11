import argparse
import math
import numpy as np

from . import load_all
from . import show
import matplotlib.pyplot as plt

## Observation Length and Total Length
T_OBS = 9
T_SEQ = 21

## Time to observe interaction
T_INT = 9 
## Time stride to compute velocity 
T_STR = 3

def compute_velocity_interaction(path, neigh_path):
    ## Computes the angle between velocity of neighbours and velocity of pp

    prim_vel = path[T_INT:T_SEQ] - path[T_INT-T_STR:T_SEQ-T_STR]
    theta1 = np.arctan2(prim_vel[:,1], prim_vel[:,0])
    neigh_vel = neigh_path[T_INT:T_SEQ] - neigh_path[T_INT-T_STR:T_SEQ-T_STR]
    # print(prim_vel)
    # print(theta1 * 180 / np.pi)
    # print(rel_dist.shape)
    vel_interaction = np.zeros(neigh_vel.shape[0:2])
    sign_interaction = np.zeros(neigh_vel.shape[0:2])

    # print(theta_interaction.shape)
    for n in range(neigh_vel.shape[1]):
        theta2 = np.arctan2(neigh_vel[:, n, 1], neigh_vel[:, n, 0])
        sign_interaction[:, n] = np.sign(theta2 - theta1 - np.pi)
        vel_interaction[:, n] = np.abs((theta2 - theta1 - np.pi)* 180 / np.pi)
    return vel_interaction, sign_interaction

def compute_theta_interaction(path, neigh_path):
    ## Computes the angle between line joining pp to neighbours and velocity of pp

    prim_vel = path[T_INT:T_SEQ] - path[T_INT-T_STR:T_SEQ-T_STR]
    theta1 = np.arctan2(prim_vel[:,1], prim_vel[:,0])
    rel_dist = neigh_path[T_INT:T_SEQ] - path[T_INT:T_SEQ][:, np.newaxis, :]
    # print(prim_vel)
    # print(theta1 * 180 / np.pi)
    # print(rel_dist.shape)
    theta_interaction = np.zeros(rel_dist.shape[0:2])
    sign_interaction = np.zeros(rel_dist.shape[0:2])

    # print(theta_interaction.shape)
    for n in range(rel_dist.shape[1]):
        theta2 = np.arctan2(rel_dist[:, n, 1], rel_dist[:, n, 0])
        sign_interaction[:, n] = np.sign(theta2 - theta1)
        theta_interaction[:, n] = np.abs((theta2 - theta1)* 180 / np.pi)
    return theta_interaction, sign_interaction

def compute_dist_rel(path, neigh_path):
    ## Distance between pp and neighbour 
    dist_rel = np.linalg.norm((neigh_path[T_INT:T_SEQ] - path[T_INT:T_SEQ][:, np.newaxis, :]), axis=2)
    return dist_rel

def compute_interaction(theta_rel, dist_rel, angle, dist_thresh):
    ## Interaction is defined as 
    ## 1. distance < threshold and 
    ## 2. angle between velocity of pp and line joining pp to neighbours
    theta_bool = (theta_rel < angle)
    dist_bool = (dist_rel < dist_thresh)
    interaction_matrix = (theta_rel < angle) & (dist_rel < dist_thresh) == 1

    return interaction_matrix

def compute_theta_vr(path):
    ## Computes the angle between velocity of pp at t_obs and velocity of pp at t_pred
    row1, row2, row3, row4 = path[5], path[8], path[17], path[20]
    diff1 = np.array([row2[0] - row1[0], row2[1] - row1[1]])
    diff2 = np.array([row4[0] - row3[0], row4[1] - row3[1]])
    theta1 = np.arctan2(diff1[1], diff1[0])
    theta2 = np.arctan2(diff2[1], diff2[0])
    vr1 = np.linalg.norm(diff1) / (3 * 0.4)
    vr2 = np.linalg.norm(diff2) / (3 * 0.4)
    if vr1 < 0.1:
        return 0, 0
    return theta2 - theta1, vr2

def theta_rotation(xy, theta):
    # rotates scene by theta
    ct = math.cos(theta)
    st = math.sin(theta)

    r = np.array([[ct, st], [-st, ct]])
    return np.einsum('ptc,ci->pti', xy, r)


def center_scene(path, neigh_path):
    # normalizes scene around t_obs of pp
    center = path[T_INT, :]
    path = path - center
    neigh_path = neigh_path - center

    k = T_INT - T_STR
    while sum(path[k, :]==0)==2 and k > 0:
        k -= 1

    if k > 0:
        thet = np.pi + np.arccos((path[k, 1])/(np.linalg.norm([0, -1])*np.linalg.norm(path[k, :])))
        if path[k, 0] < 0:
            thet = -thet
        # rot_mat = np.array([[np.cos(thet),-np.sin(thet)],[np.sin(thet),np.cos(thet)]])
        # print(path[:, np.newaxis, :].shape)
        norm_path = theta_rotation(path[:, np.newaxis, :], thet)
        norm_neigh_path = theta_rotation(neigh_path, thet)
        return norm_path[:, 0], norm_neigh_path

    else:
        return path, neigh_path


def multimodality_plot(input_file, pos_angle=4, vel_angle=4, dist_thresh=5, n_theta=15, vr_max=2.5, vr_n=10):
	## Multimodality of interactions
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    heatmap = False

    # run
    i = 0
    x_values = []
    y_values = []
    for primary_ped, rows in load_all(input_file):
        path = rows[:, 0]
        neigh_path = rows[:, 1:]
        theta_interaction, sign_interaction = compute_theta_interaction(path, neigh_path)
        vel_interaction, sign_vel_interaction = compute_velocity_interaction(path, neigh_path)
        dist_rel = compute_dist_rel(path, neigh_path)

        interaction_matrix = compute_interaction(vel_interaction, dist_rel, vel_angle, dist_thresh) \
        					 & compute_interaction(theta_interaction, dist_rel, pos_angle, dist_thresh)
        # interaction_matrix1 = compute_interaction(vel_interaction, dist_rel, angle, dist_thresh)
        # if not np.any(interaction_matrix1 ^ interaction_matrix):
        #     print("Same")
        # else:
        #     print("NOT SAME")
        # if np.sum(vel_interaction - theta_interaction) == 0:
        #     print("BUG")
        # print(interaction_matrix.shape)
        # print(interaction_matrix)
        interaction_index = np.any(interaction_matrix, axis=0)

        if np.sum(interaction_index) < 3:
            i += 1
            neigh = neigh_path[:,interaction_index]
            path, neigh = center_scene(path, neigh)
            # ax.plot(path[:9, 0], path[:9, 1])
            ax.plot(path[T_OBS:, 0], path[T_OBS:, 1])
            # ax.plot(neigh[T_OBS:, 0, 0], neigh[T_OBS:, 0, 1])
            ## PP
            # x_values = np.concatenate((x_values, path[T_OBS:, 0]))
            # y_values = np.concatenate((y_values, path[T_OBS:, 1]))
            ## Neighbours 
            # x_values = np.concatenate((x_values, neigh[T_OBS:, 0, 0]))
            # y_values = np.concatenate((y_values, neigh[T_OBS:, 0, 1]))
            # ax.plot(path[0, 0], path[0, 1], color='g', marker='o', label = 'start point')
            # ax.plot(path[-1, 0], path[-1, 1], color='r', marker='x', label = 'end point')
            # ax.plot(neigh[:, 0, 0], neigh[:, 0, 1])
            # ax.plot(neigh[0, 0, 0], neigh[0, 0, 1], color='g', marker='o', label = 'start point')
            # ax.plot(neigh[-1, 0, 0], neigh[-1, 0, 1], color='r', marker='x', label = 'end point')
            ax.set_xlim([-6, 6])
            ax.set_ylim([-6, 6])

    if heatmap:
	    ax.clear()
	    heatmap, xedges, yedges = np.histogram2d(y_values, x_values, bins=[50, 50], range=[[-3,3],[-3,3]])
	    thres = 10
	    heatmap[np.where(heatmap >= thres)] = thres
	    plt.imshow(heatmap, interpolation='none', origin='lower')
	    plt.colorbar()

    fig.show()
    plt.close(fig)


def dataset_plots(input_file, pos_angle=4, vel_angle=4, dist_thresh=5, n_theta=360, vr_max=2.5, vr_n=10, choice='pos'):
	## Distributions of interactions

	## choice : Choice of angle to be thresholded 
			# 'pos': Angle between Velocity of PP and Line Joining PP to N
			# 'vel': Angle between Velocity of PP and Velocity of N 
			# 'pos_vel': Both pos and vel 

    distr = np.zeros((n_theta, vr_n))
    def fill_grid(theta_vr):
        theta, vr, sign = theta_vr
        theta = theta*(2*np.pi)/360
        theta[np.where(sign < 0)] = (2*np.pi - 1/n_theta) - theta[np.where(sign < 0)]
        thetap = np.floor(theta * distr.shape[0] / (2*np.pi)).astype(int)
        vrp = np.floor(vr * distr.shape[1] / dist_thresh).astype(int)
        distr[thetap, vrp] += 1
    
    unbinned_vr = [[] for _ in range(n_theta)]
    def fill_unbinned_vr(theta_vr):
        theta, vr, sign = theta_vr
        theta = theta*(2*np.pi)/360
        theta[np.where(sign < 0)] = (2*np.pi - 1/n_theta) - theta[np.where(sign < 0)]
        thetap = np.floor(theta * len(unbinned_vr) / (2*np.pi)).astype(int)
        for th in range(len(thetap)):
            unbinned_vr[thetap[th]].append(vr[th])

    vr_max = dist_thresh
    # run
    i = 0
    for primary_ped, rows in load_all(input_file):
        # print("PP", primary_ped)
        path = rows[:, 0]
        neigh_path = rows[:, 1:]
        theta_interaction, sign_interaction = compute_theta_interaction(path, neigh_path)
        vel_interaction, sign_vel_interaction = compute_velocity_interaction(path, neigh_path)
        dist_rel = compute_dist_rel(path, neigh_path)
        
        ## str choice
        if choice == 'pos':
        	chosen_interaction = theta_interaction
        	angle_thres = pos_angle
        elif choice == 'vel':
        	chosen_interaction = vel_interaction
        	sign_interaction = sign_vel_interaction
        	angle_thres = vel_angle
        else:
        	raise NotImplementedError 

        interaction_matrix = compute_interaction(chosen_interaction, dist_rel, angle_thres, dist_thresh)
        chosen_true = chosen_interaction[interaction_matrix]
        sign_true = sign_interaction[interaction_matrix]
        dist_true = dist_rel[interaction_matrix]
        fill_grid((chosen_true, dist_true, sign_true))
        fill_unbinned_vr((chosen_true, dist_true, sign_true))

        # interaction_matrix = compute_interaction(theta_interaction, dist_rel, angle, dist_thresh)
        # theta_true = theta_interaction[interaction_matrix]
        # sign_true = sign_interaction[interaction_matrix]
        # dist_true = dist_rel[interaction_matrix]
        # fill_grid((theta_true, dist_true, sign_true))
        # fill_unbinned_vr((theta_true, dist_true, sign_true))


        # interaction_matrix = compute_interaction(vel_interaction, dist_rel, angle, dist_thresh)
        # vel_true = vel_interaction[interaction_matrix]
        # sign_true = sign_vel_interaction[interaction_matrix]
        # dist_true = dist_rel[interaction_matrix]
        # fill_grid((vel_true, dist_true, sign_true))
        # fill_unbinned_vr((vel_true, dist_true, sign_true))        


    with show.canvas(input_file + '.' + choice + '.png', figsize=(4, 4), subplot_kw={'polar': True}) as ax:
        r_edges = np.linspace(0, vr_max, distr.shape[1] + 1)
        theta_edges = np.linspace(0, 2*np.pi, distr.shape[0] + 1)
        thetas, rs = np.meshgrid(theta_edges, r_edges)
        ax.pcolormesh(thetas, rs, distr.T, cmap='Blues')

        median_vr = np.array([np.median(vrs) if len(vrs) > 5 else np.nan
                              for vrs in unbinned_vr])
        center_thetas = np.linspace(0.0, 2*np.pi, len(median_vr) + 1)
        center_thetas = 0.5 * (center_thetas[:-1] + center_thetas[1:])
        # close loop
        center_thetas = np.hstack([center_thetas, center_thetas[0:1]])
        median_vr = np.hstack([median_vr, median_vr[0:1]])
        # plot median radial velocity
        ax.plot(center_thetas, median_vr, label='median $d_r$ [m/s]', color='orange')

        ax.grid(linestyle='dotted')
        ax.legend()    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_files', nargs='+',
                        help='Trajnet dataset file(s).')
    args = parser.parse_args()

    print('{dataset:>60s} |     N'.format(dataset=''))
    for dataset_file in args.dataset_files:
        print('{dataset:>60s} | {N:>5}'.format(
            dataset=dataset_file,
            N=sum(1 for _ in load_all(dataset_file)),
        ))

    pos_angle = 20
    vel_angle = 20
    dist_thresh = 5
    n_theta = 36
    vr_max = 2.5
    vr_n = 10
    for dataset_file in args.dataset_files:
        # multimodality_plot(dataset_file, pos_angle, vel_angle, dist_thresh, n_theta, vr_max, vr_n)
        dataset_plots(dataset_file, pos_angle, vel_angle, dist_thresh, n_theta, vr_max, vr_n, 'pos')


if __name__ == '__main__':
    main()
