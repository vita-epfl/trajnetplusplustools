import argparse
import math
import numpy as np

from . import load_all
from . import show
import matplotlib.pyplot as plt


def compute_theta_interaction(path, neigh_path):
    prim_vel = path[9:21] - path[6:18]
    theta1 = np.arctan2(prim_vel[:,1], prim_vel[:,0])
    rel_dist = neigh_path[9:21] - path[9:21][:, np.newaxis, :]
    # print(prim_vel)
    # print(theta1 * 180 / np.pi)
    # print(rel_dist.shape)
    theta_interaction = np.zeros(rel_dist.shape[0:2])
    # print(theta_interaction.shape)
    for n in range(rel_dist.shape[1]):
        theta2 = np.arctan2(rel_dist[:, n, 1], rel_dist[:, n, 0])
        theta_interaction[:, n] = np.abs((theta2 - theta1)* 180 / np.pi)
    return theta_interaction

def compute_dist_rel(path, neigh_path):
    dist_rel = np.linalg.norm((neigh_path[9:21] - path[9:21][:, np.newaxis, :]), axis=2)
    # print(dist_rel.shape)
    return dist_rel

def compute_interaction(theta_rel, dist_rel, angle=15, dist_thresh=5):
    theta_bool = (theta_rel < angle)
    dist_bool = (dist_rel < dist_thresh)
    interaction_matrix = np.logical_and(theta_bool, dist_bool)
    return interaction_matrix

def compute_theta_vr(path):
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
    # theta = random.random() * 2.0 * math.pi
    ct = math.cos(theta)
    st = math.sin(theta)

    r = np.array([[ct, st], [-st, ct]])
    return np.einsum('ptc,ci->pti', xy, r)


def center_scene(path, neigh_path):
    center = path[8, :]
    path = path - center
    neigh_path = neigh_path - center

    k = 6
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
        
def dataset_plots(input_file, n_theta=64, vr_max=2.5, vr_n=10):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # run
    i = 0
    x_values = []
    y_values = []
    for primary_ped, rows in load_all(input_file):
        # print("PP", primary_ped, 'Rows', rows.shape)
        path = rows[:, 0]
        neigh_path = rows[:, 1:]
        theta_interaction = compute_theta_interaction(path, neigh_path)
        dist_rel = compute_dist_rel(path, neigh_path)
        # print(dist_rel)
        # print(dist_rel.shape)
        interaction_matrix = compute_interaction(theta_interaction, dist_rel)
        # print(interaction_matrix.shape)
        # print(interaction_matrix)
        interaction_index = np.any(interaction_matrix, axis=0)
        # print(interaction_index.shape)
        # print(interaction_index)
        if np.sum(interaction_index) > 5:
            i += 1
            # print(interaction_index)
            neigh = neigh_path[:,interaction_index]
            path, neigh = center_scene(path, neigh)
            # ax.plot(path[:9, 0], path[:9, 1])
            ax.plot(path[9:, 0], path[9:, 1])
            x_values = np.concatenate((x_values, path[9:, 0]))
            y_values = np.concatenate((y_values, path[9:, 1]))
            # ax.plot(path[0, 0], path[0, 1], color='g', marker='o', label = 'start point')
            ax.plot(path[-1, 0], path[-1, 1], color='r', marker='x', label = 'end point')
            # ax.plot(neigh[:, 0, 0], neigh[:, 0, 1])
            # ax.plot(neigh[0, 0, 0], neigh[0, 0, 1], color='g', marker='o', label = 'start point')
            # ax.plot(neigh[-1, 0, 0], neigh[-1, 0, 1], color='r', marker='x', label = 'end point')
            ax.set_xlim([-6, 6])
            ax.set_ylim([-6, 6])
    
    # plt.show()
    # print(x_values.shape)
    # print(i)

    ax.clear()

    heatmap, xedges, yedges = np.histogram2d(y_values, x_values, bins=[50, 50], range=[[-3,3],[-3,3]])
    thres = 20
    heatmap[np.where(heatmap >= thres)] = thres
    # print(xedges)
    plt.imshow(heatmap, interpolation='none', origin='lower')
    plt.colorbar()
    plt.show()
    ax.clear()

        # print(neigh_path[:,interaction_index].shape)

        # print(theta_interaction)
        # print(theta_interaction < 15)
        # t_vr = compute_theta_vr(path)
        # fill_grid(t_vr)
        # fill_unbinned_vr(t_vr)

    # with show.canvas(input_file + '.theta.png', figsize=(4, 4), subplot_kw={'polar': True}) as ax:
    #     r_edges = np.linspace(0, vr_max, distr.shape[1] + 1)
    #     theta_edges = np.linspace(0, 2*np.pi, distr.shape[0] + 1)
    #     thetas, rs = np.meshgrid(theta_edges, r_edges)
    #     ax.pcolormesh(thetas, rs, distr.T, cmap='Blues')

    #     median_vr = np.array([np.median(vrs) if len(vrs) > 5 else np.nan
    #                           for vrs in unbinned_vr])
    #     center_thetas = np.linspace(0.0, 2*np.pi, len(median_vr) + 1)
    #     center_thetas = 0.5 * (center_thetas[:-1] + center_thetas[1:])
    #     # close loop
    #     center_thetas = np.hstack([center_thetas, center_thetas[0:1]])
    #     median_vr = np.hstack([median_vr, median_vr[0:1]])
    #     # plot median radial velocity
    #     ax.plot(center_thetas, median_vr, label='median $v_r$ [m/s]', color='orange')

    #     ax.grid(linestyle='dotted')
    #     ax.legend()

    # # histogram of radial velocities
    # with show.canvas(input_file + '.speed.png', figsize=(4, 4)) as ax:
    #     ax.hist([vr for theta_bin in unbinned_vr for vr in theta_bin],
    #             bins=20, range=(0.0, vr_max))
    #     ax.set_xlabel('$v_r$ [m/s]')


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

    for dataset_file in args.dataset_files:
        dataset_plots(dataset_file)


if __name__ == '__main__':
    main()
