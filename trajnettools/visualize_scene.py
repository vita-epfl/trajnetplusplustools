import argparse
import math
import numpy as np

from . import load_all
from . import show

import matplotlib.pyplot as plt


def dataset_plots(input_file):
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    # run
    for primary_ped, rows in load_all(input_file):
        traj = rows[:, 0]
        # print(traj)
        ax.plot(traj[:, 0], traj[:, 1])
        # ax.plot(traj[0, 0], traj[0, 1], color='g', marker='o', label = 'start point')
        # ax.plot(traj[-1, 0], traj[-1, 1], color='r', marker='o', label = 'end point')
        # for i in range(rows.shape[1]):
        #     traj = rows[:, i]
        #     ax.plot(traj[:, 0], traj[:, 1])
    plt.show()

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


if __name__ == '__main__':
    main()
