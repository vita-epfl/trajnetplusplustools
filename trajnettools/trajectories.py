import argparse

from .readers import Reader
from . import show


def trajectories(primary_pedestrian, rows, output_file=None):
    with show.rows(primary_pedestrian, rows, output_file):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_file',
                        help='trajnet dataset file')
    parser.add_argument('--n', type=int, default=5,
                        help='sample n trajectories')
    parser.add_argument('--id', type=int, nargs='*',
                        help='plot a particular scene')
    parser.add_argument('-o', '--output', default=None,
                        help='specify output prefix')
    parser.add_argument('--random', default=False, action='store_true',
                        help='randomize scenes')
    args = parser.parse_args()

    if args.output is None:
        args.output = args.dataset_file

    reader = Reader(args.dataset_file)
    if args.id:
        scenes = reader.scenes(ids=args.id, randomize=args.random)
    elif args.n:
        scenes = reader.scenes(limit=args.n, randomize=args.random)
    else:
        scenes = reader.scenes(randomize=args.random)

    for scene_id, primary_ped, rows in scenes:
        output = '{}.scene{}.png'.format(args.output, scene_id)
        trajectories(primary_ped, rows, output)


if __name__ == '__main__':
    main()
