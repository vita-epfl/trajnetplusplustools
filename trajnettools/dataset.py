import argparse
import glob
import logging

from . import readers

LOG = logging.getLogger(__name__)


def load(path, recursive=True, randomize=False):
    """Parsed scenes at the given path returned as a generator.

    Each scene contains a list of `Row`s where the first pedestrian is the
    pedestrian of interest.

    The path supports `**` when the `recursive` argument is True (default).
    """
    LOG.info('loading dataset from %s', path)
    filenames = glob.iglob(path, recursive=recursive)
    for filename in filenames:
        yield from readers.TrajnetReader(filename).scenes(randomize=randomize)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_files', nargs='+',
                        help='dataset file(s)')
    args = parser.parse_args()

    print('{dataset:>60s} |     N'.format(dataset=''))
    for dataset_file in args.dataset_files:
        print('{dataset:>60s} | {N:>5}'.format(
            dataset=dataset_file,
            N=sum(1 for _ in load(dataset_file)),
        ))


if __name__ == '__main__':
    main()
