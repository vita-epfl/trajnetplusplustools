import argparse
import glob
import logging

from . import Reader

LOG = logging.getLogger(__name__)


def load_all(path, recursive=True, as_paths=False, sample=None):
    """Parsed scenes at the given path returned as a generator.

    Each scene contains a list of `Row`s where the first pedestrian is the
    pedestrian of interest.

    The path supports `**` when the `recursive` argument is True (default).
    """
    LOG.info('loading dataset from %s', path)
    filenames = glob.iglob(path, recursive=recursive)
    for filename in filenames:
        sample_rate = None
        if sample is not None:
            for k, v in sample.items():
                if k in filename:
                    sample_rate = v
        yield from Reader(filename).scenes(as_paths=as_paths, sample=sample_rate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_files', nargs='+',
                        help='dataset file(s)')
    args = parser.parse_args()

    print('{dataset:>60s} |     N'.format(dataset=''))
    for dataset_file in args.dataset_files:
        print('{dataset:>60s} | {N:>5}'.format(
            dataset=dataset_file,
            N=sum(1 for _ in load_all(dataset_file)),
        ))


if __name__ == '__main__':
    main()
