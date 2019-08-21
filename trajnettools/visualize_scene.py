import argparse
import math
import numpy as np

from . import load_all
from . import show

import matplotlib.pyplot as plt

x_list = []
y_list = []
v_list = []
def dataset_plots(input_file):
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    # run
    for primary_ped, rows in load_all(input_file):
        traj = rows[:, 0]

        ## Trajectory
        # ax.plot(traj[:, 0], traj[:, 1])

        # ax.plot(traj[0, 0], traj[0, 1], color='g', marker='o', label = 'start point')
        x_list.append(traj[0, 0])
        y_list.append(traj[0, 1])
        v_list = np.arctan2(traj[3, 1] - traj[0, 1], traj[3, 0] - traj[0, 0])
        # ax.plot(traj[-1, 0], traj[-1, 1], color='r', marker='o', label = 'end point')
        # for i in range(row    s.shape[1]):
        #     traj = rows[:, i]
        #     ax.plot(traj[:, 0], traj[:, 1])

    # plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_files', nargs='+',
                        help='Trajnet dataset file(s).')
    args = parser.parse_args()

    print('{dataset:>60s} |     N'.format(dataset=''))
    for dataset_file in args.dataset_files:
        print('{dataset:>60s} | {N:>5}'.format(
            dataset=dataset_file,
            N=sum(1 for _ in load_all(dataset_file))
        ))
        # print('{dataset:>60s} | {N:>5}'.format(
        #     dataset=dataset_file,
        #     N=sum(rows.shape[1] for primary_ped, rows in load_all(dataset_file))
        # ))

    for dataset_file in args.dataset_files:
        dataset_plots(dataset_file)

    H, x_bins, y_bins = np.histogram2d(x_list, y_list, bins=[50,50])
    print(H.shape)
    print(x_bins.shape)
    # print("Xedges: ", xedges)
    # x_bin_midpoints = x_bins[:-1] + np.diff(x_bins)/2
    # y_bin_midpoints = y_bins[:-1] + np.diff(y_bins)/2
    # cdf = np.cumsum(H.ravel())
    # cdf = cdf / cdf[-1]

    # values = np.random.rand(200)
    # value_bins = np.searchsorted(cdf, values)
    # x_idx, y_idx = np.unravel_index(value_bins,
    #                                 (len(x_bin_midpoints),
    #                                  len(y_bin_midpoints)))
    # random_from_cdf = np.column_stack((x_bin_midpoints[x_idx],
    #                                    y_bin_midpoints[y_idx]))
    # new_x, new_y = random_from_cdf.T
    # neigh_array = np.column_stack((new_x, new_y))
    # print("neigh_array:", len(neigh_array))
    # unique_neigh_array = np.unique(neigh_array, axis=0)
    # print("Unique_array:", len(unique_neigh_array))
    # # plt.subplot(211)
    # plt.hist2d(x_list, y_list, bins=(50, 50))
    # plt.show()
    # plt.close()
    # # plt.subplot(212)
    # plt.hist2d(new_x, new_y, bins=(50, 50))
    # plt.show()
    # plt.close()  

    # H = H.T
    print(H.shape)
    H_crop = H[20:30, 20:30]
    print(H_crop.shape)
    x_bins_crop = x_bins[20:31]
    print(x_bins_crop.shape)
    y_bins_crop = y_bins[20:31]
    plt.imshow(H[20:30, 20:30].T, interpolation='nearest', origin='low')
    plt.show()
    plt.close()  

    x_bin_midpoints = x_bins_crop[:-1] + np.diff(x_bins_crop)/2
    y_bin_midpoints = y_bins_crop[:-1] + np.diff(y_bins_crop)/2
    cdf = np.cumsum(H_crop.ravel())
    cdf = cdf / cdf[-1]

    values = np.random.rand(10000)
    value_bins = np.searchsorted(cdf, values)
    x_idx, y_idx = np.unravel_index(value_bins,
                                    (len(x_bin_midpoints),
                                     len(y_bin_midpoints)))
    random_from_cdf = np.column_stack((x_bin_midpoints[x_idx],
                                       y_bin_midpoints[y_idx]))
    new_x, new_y = random_from_cdf.T
    # neigh_array = np.column_stack((new_x, new_y))
    # print("neigh_array:", len(neigh_array))
    # unique_neigh_array = np.unique(neigh_array, axis=0)
    # print("Unique_array:", len(unique_neigh_array))
    # plt.subplot(211)
    # plt.hist2d(x_list, y_list, bins=(50, 50))
    # plt.show()
    # plt.close()
    # plt.subplot(212)
    plt.hist2d(new_x, new_y)
    plt.show()
    plt.close()  


if __name__ == '__main__':
    main()
