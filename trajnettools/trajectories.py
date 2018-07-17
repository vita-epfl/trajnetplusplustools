import argparse
from collections import defaultdict
from contextlib import contextmanager
import matplotlib.pyplot as plt
import numpy as np

from . import readers


@contextmanager
def show(fig_file=None, **kwargs):
    fig, ax = plt.subplots(**kwargs)

    yield ax

    fig.set_tight_layout(True)
    if fig_file:
        fig.savefig(fig_file, dpi=300)
    fig.show()
    plt.close(fig)


def trajectories(primary_pedestrian, rows, output_file):
    trajectories_by_id = defaultdict(list)
    for row in rows:
        trajectories_by_id[row.pedestrian].append(row)

    with show(output_file, figsize=(8, 8)) as ax:
        # primary
        xs = [r.x for r in trajectories_by_id[primary_pedestrian]]
        ys = [r.y for r in trajectories_by_id[primary_pedestrian]]
        # markers
        ax.plot(xs[0:1], ys[0:1], color='black', marker='x', label='start', linestyle='None')
        ax.plot(xs[-1:], ys[-1:], color='black', marker='o', label='end', linestyle='None')
        # track
        ax.plot(xs, ys, color='black', linestyle='solid', label='primary')

        # other tracks
        for ped_id, ped_rows in trajectories_by_id.items():
            if ped_id == primary_pedestrian:
                continue

            xs = [r.x for r in ped_rows]
            ys = [r.y for r in ped_rows]
            # markers
            ax.plot(xs[0:1], ys[0:1], color='black', marker='x', linestyle='None')
            ax.plot(xs[-1:], ys[-1:], color='black', marker='o', linestyle='None')
            # track
            ax.plot(xs, ys, color='black', linestyle='dotted')

        # frame
        ax.grid(linestyle='dotted')
        ax.legend()
        ax.set_aspect(1.0)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')


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

    reader = readers.TrajnetReader(args.dataset_file)
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
