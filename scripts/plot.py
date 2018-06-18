from contextlib import contextmanager
import math
import matplotlib.pyplot as plt
import numpy as np
import pysparkling
import trajnettools


@contextmanager
def show(fig_file=None, **kwargs):
    fig, ax = plt.subplots(**kwargs)

    yield ax

    fig.set_tight_layout(True)
    if fig_file:
        fig.savefig(fig_file, dpi=300)
    fig.show()
    plt.close(fig)


@contextmanager
def show2(fig_file=None, **kwargs):
    fig = plt.figure(**kwargs)
    ax1 = fig.add_subplot(1, 2, 1, polar=True)
    ax2 = fig.add_subplot(1, 2, 2)

    yield ax1, ax2

    fig.set_tight_layout(True)
    if fig_file:
        fig.savefig(fig_file, dpi=300)
    fig.show()
    plt.close(fig)


def theta_r(row1, row2, row3):
    diff1 = np.array([row2.x - row1.x, row2.y - row1.y])
    diff2 = np.array([row3.x - row2.x, row3.y - row2.y])
    theta1 = np.arctan2(diff1[1], diff1[0])
    theta2 = np.arctan2(diff2[1], diff2[0])
    r1 = np.linalg.norm(diff1)
    r2 = np.linalg.norm(diff2)
    if r1 < 0.1:
        return 0, 0
    return theta2 - theta1, r2


def dataset_plots(input_files, output, n_theta=64, vr_max=2.5, vr_n=10):
    sc = pysparkling.Context()
    scenes = (sc
              .wholeTextFiles(input_files)
              .mapValues(trajnettools.readers.trajnet_marked)
              .mapValues(lambda path: [r for r in path if r.mark])
              .mapValues(lambda path: theta_r(path[7], path[10], path[19]))
              .cache())

    distr = np.zeros((n_theta, vr_n))
    def fill_grid(theta_r):
        theta, r = theta_r
        thetap = math.floor(theta * distr.shape[0] / (2*np.pi))
        vrp = math.floor(r * distr.shape[1] / 3.6 / vr_max)
        if vrp >= distr.shape[1]:
            vrp = distr.shape[1] - 1
        if vrp < 0.01:
            return
        distr[thetap, vrp] += 1

    scenes.values().foreach(fill_grid)

    unbinned_vr = [[] for _ in range(n_theta)]
    def fill_unbinned_vr(theta_r):
        theta, r = theta_r
        thetap = math.floor(theta * len(unbinned_vr) / (2*np.pi))
        vr = r / 3.6
        unbinned_vr[thetap].append(vr)
    scenes.values().foreach(fill_unbinned_vr)
    median_vr = np.array([np.median(vrs) for vrs in unbinned_vr])
    median_vr[median_vr < 0.1] = np.nan
    print(median_vr)

    with show2(output + '_theta_speed.png', figsize=(8, 4)) as (ax1, ax2):
        r_edges = np.linspace(0, vr_max, distr.shape[1] + 1)
        theta_edges = np.linspace(0, 2*np.pi, distr.shape[0] + 1)
        thetas, rs = np.meshgrid(theta_edges, r_edges)
        ax1.pcolormesh(thetas, rs, distr.T, cmap='Blues')

        center_thetas = np.linspace(0.0, 2*np.pi, len(median_vr) + 1)
        center_thetas = 0.5 * (center_thetas[:-1] + center_thetas[1:])
        # close loop
        center_thetas = np.hstack([center_thetas, center_thetas[0:1]])
        median_vr = np.hstack([median_vr, median_vr[0:1]])
        # plot median radial velocity
        ax1.plot(center_thetas, median_vr, label='median $v_r$ [m/s]', color='orange')

        ax1.grid(linestyle='dotted')
        ax1.legend()

        # histogram of radial velocities
        ax2.hist([vr for theta_bin in unbinned_vr for vr in theta_bin],
                 bins=20, range=(0.0, vr_max))
        ax2.set_xlabel('$v_r$ [m/s]')


if __name__ == '__main__':
    dataset_plots('data/train/biwi_hotel/*.txt', 'output/biwi_hotel')
    dataset_plots('data/train/crowds_students001/*.txt', 'output/crowds_students001')
    dataset_plots('data/train/crowds_students003/*.txt', 'output/crowds_students003')
    dataset_plots('data/train/crowds_zara02/*.txt', 'output/crowds_zara02')
    dataset_plots('data/train/crowds_zara03/*.txt', 'output/crowds_zara03')
    dataset_plots('data/train/mot_pets2009_s2l1/*.txt', 'output/mot_pets2009_s2l1')  #, n_theta=8, vr_max=4.0, vr_n=5)
