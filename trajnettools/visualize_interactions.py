import argparse
import numpy as np

from . import load_all
from . import show
from . import Reader
from . import metrics
from .interactions import get_interaction_matrix, check_group
from . import kalman

def non_linear(scene):
    primary_prediction, _ = kalman.predict(scene)[0]
    score = metrics.final_l2(scene[0], primary_prediction)
    return score > 1.0, primary_prediction

def interaction_length(interaction_matrix, length=1):
    interaction_sum = np.sum(interaction_matrix, axis=0)
    return interaction_sum >= length

def leader_follower(rows, args):
    interaction_matrix = get_interaction_matrix(rows, args, output='matrix')
    interaction_index = interaction_length(interaction_matrix, length=5)
    return interaction_index

def collision_avoidance(rows, args):
    interaction_matrix = get_interaction_matrix(rows, args, output='matrix')
    interaction_index = interaction_length(interaction_matrix, length=1)
    return interaction_index

def group(rows, args, dist_thresh=0.8, std_thresh=0.2):
    interaction_index = check_group(rows, args, dist_thresh, std_thresh)
    return interaction_index

def interaction_plots(input_file, interaction_type, args):
    n_instances = 0
    reader = Reader(input_file, scene_type='paths')
    scenes = [s for _, s in reader.scenes()]
    for scene in scenes:
        rows = reader.paths_to_xy(scene)

        if interaction_type == 'lf':
            interaction_index = leader_follower(rows, args)
        elif interaction_type == 'ca':
            interaction_index = collision_avoidance(rows, args)
        elif interaction_type == 'group':
            interaction_index = group(rows, args)
        else:
            interaction_matrix = get_interaction_matrix(rows, args, output='matrix')
            # "Shape": PredictionLength x Number of Neighbours
            interaction_index = interaction_length(interaction_matrix, length=1)

        path = rows[:, 0]
        neigh_path = rows[:, 1:]
        neigh = neigh_path[:, interaction_index]
        num_interactions = np.any(interaction_index)

        ## Calculate and display
        if num_interactions & (np.linalg.norm(path[-1] - path[0]) > 1.0):
            nl_tag, kf = non_linear(scene)
            kf = reader.paths_to_xy([kf])
            if nl_tag:
                n_instances += 1
                ## n Examples of interactions ##
                if n_instances <= args.n:
                    output = '{}_{}_{}.png'.format(input_file, interaction_type, n_instances)
                    with show.interaction_path(path, neigh, kalman=kf, output_file=output):
                        pass
                    output = '{}_{}_{}_full.png'.format(input_file, interaction_type, n_instances)
                    with show.interaction_path(path, neigh_path, kalman=kf, output_file=output):
                        pass
    print("Number of Instances: ", n_instances)

def distribution_plots(input_file, args):
    ## Distributions of interactions
    n_theta, vr_n, dist_thresh, choice = args.n_theta, args.vr_n, args.dist_thresh, args.choice
    distr = np.zeros((n_theta, vr_n))
    def fill_grid(theta_vr):
        theta, vr, _ = theta_vr
        theta = theta*(2*np.pi)/360
        thetap = np.floor(theta * distr.shape[0] / (2*np.pi)).astype(int)
        vrp = np.floor(vr * distr.shape[1] / dist_thresh).astype(int)
        distr[thetap, vrp] += 1

    unbinned_vr = [[] for _ in range(n_theta)]
    def fill_unbinned_vr(theta_vr):
        theta, vr, _ = theta_vr
        theta = theta*(2*np.pi)/360
        thetap = np.floor(theta * len(unbinned_vr) / (2*np.pi)).astype(int)
        for th, _ in enumerate(thetap):
            unbinned_vr[thetap[th]].append(vr[th])
    vr_max = dist_thresh

    hist = []
    def fill_hist(vel):
        hist.append(vel)

    #run
    for _, rows in load_all(input_file):
        _, chosen_true, sign_true, dist_true = \
        get_interaction_matrix(rows, args)

        fill_grid((chosen_true, dist_true, sign_true))
        fill_unbinned_vr((chosen_true, dist_true, sign_true))
        fill_hist(chosen_true)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_files', nargs='+',
                        help='Trajnet dataset file(s).')
    parser.add_argument('--pos_angle', type=int, default=0,
                        help='axis angle of position cone (in deg)')
    parser.add_argument('--vel_angle', type=int, default=0,
                        help='relative velocity centre (in deg)')
    parser.add_argument('--pos_range', type=int, default=15,
                        help='range of position cone (in deg)')
    parser.add_argument('--vel_range', type=int, default=15,
                        help='relative velocity span (in rsdeg)')
    parser.add_argument('--dist_thresh', type=int, default=5,
                        help='threshold of distance (in m)')
    parser.add_argument('--n_theta', type=int, default=72,
                        help='number of segments in polar plot radially')
    parser.add_argument('--vr_n', type=int, default=10,
                        help='number of segments in polar plot linearly')
    parser.add_argument('--choice', default='bothpos',
                        help='choice of interaction')
    parser.add_argument('--n', type=int, default=5,
                        help='number of plots')
    parser.add_argument('--interaction_type', default='lf',
                        help='type of interaction')
    args = parser.parse_args()

    print('{dataset:>60s} |     N'.format(dataset=''))
    for dataset_file in args.dataset_files:
        print('{dataset:>60s} | {N:>5}'.format(
            dataset=dataset_file,
            N=sum(1 for _ in load_all(dataset_file)),
        ))

    interaction_type = args.interaction_type
    ## args
    if interaction_type == 'lf':
        args.pos_angle = 0
        args.pos_range = 15
        args.vel_angle = 0
        args.vel_range = 15

    elif interaction_type == 'ca':
        args.pos_angle = 0
        args.pos_range = 15
        args.vel_angle = 180
        args.vel_range = 15

    elif interaction_type == 'group':
        args.pos_range = 45
        args.choice = 'pos'

    else:
        pass

    for dataset_file in args.dataset_files:
        # pass

        ## Interaction
        interaction_plots(dataset_file, interaction_type, args)

        ## Position Global
        # distribution_plots(dataset_file, args)

if __name__ == '__main__':
    main()
