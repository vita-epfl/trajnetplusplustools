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
    print("Args: ", args)
    n_instances = 0
    reader = Reader(input_file, scene_type='paths')
    if interaction_type in [1, 2, 3, 4]:
        type_ids = [scene_id for scene_id in reader.scenes_by_id \
                    if interaction_type in reader.scenes_by_id[scene_id].tag[1]]
    else:
        type_ids = [scene_id for scene_id in reader.scenes_by_id \
                    if 4 in reader.scenes_by_id[scene_id].tag]

    scenes = [s for _, s in reader.scenes()]
    for type_id in type_ids:
        scene = scenes[type_id]
        rows = reader.paths_to_xy(scene)
        path = rows[:, 0]
        neigh_path = rows[:, 1:]

        if interaction_type == 1:
            interaction_index = leader_follower(rows, args)
        elif interaction_type == 2:
            interaction_index = collision_avoidance(rows, args)
        elif interaction_type == 3:
            interaction_index = group(rows, args)
        elif interaction_type == 4:
            interaction_matrix = get_interaction_matrix(rows, args, output='matrix')
            # "Shape": PredictionLength x Number of Neighbours
            interaction_index = interaction_length(interaction_matrix, length=1)
        else:
            interaction_index = [False]*neigh_path.shape[1]

        neigh = neigh_path[:, interaction_index]

        kf = None
        # nl_tag, kf = non_linear(scene)
        # kf = reader.paths_to_xy([kf])

        n_instances += 1
        ## n Examples of interactions ##
        if n_instances < args.n:
            output = '{}_{}_{}.pdf'.format(input_file, interaction_type, n_instances)
            with show.interaction_path(path, neigh, kalman=kf, output_file=output):
                pass
            output = '{}_{}_{}_full.pdf'.format(input_file, interaction_type, n_instances)
            with show.interaction_path(path, neigh_path, kalman=kf, output_file=output):
                pass

    print("Number of Instances: ", n_instances)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_files', nargs='+',
                        help='Trajnet dataset file(s).')
    parser.add_argument('--pos_angle', type=int, default=0,
                        help='axis angle of position cone (in deg)')
    parser.add_argument('--vel_angle', type=int, default=0,
                        help='relative velocity centre (in deg)')
    parser.add_argument('--pos_range', type=int, default=45,
                        help='range of position cone (in deg)')
    parser.add_argument('--vel_range', type=int, default=20,
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
    parser.add_argument('--interaction_type', type=int, default=2,
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
    if interaction_type == 1:
        args.pos_angle = 0
        args.pos_range = 15
        args.vel_angle = 0
        args.vel_range = 15

    elif interaction_type == 2:
        args.pos_angle = 0
        args.pos_range = 15
        args.vel_angle = 180
        args.vel_range = 15

    elif interaction_type == 3:
        args.pos_range = 45
        args.choice = 'pos'

    elif interaction_type == 4:
        args.pos_range = 15
        args.choice = 'pos'

    ## Type Non-Linear without interaction_sum
    else:
        pass

    for dataset_file in args.dataset_files:
        ## Interaction
        interaction_plots(dataset_file, interaction_type, args)

if __name__ == '__main__':
    main()
