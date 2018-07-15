import glob
import logging

from . import readers

LOG = logging.getLogger(__name__)


def load(path, recursive=True):
    """Parsed scenes at the given path returned as a generator.

    Each scene contains a list of `Row`s where the first pedestrian is the
    pedestrian of interest.

    The path supports `**` when the `recursive` argument is True (default).
    """
    LOG.info('loading dataset from %s', path)
    filenames = glob.iglob(path, recursive=recursive)
    for filename in filenames:
        with open(filename, 'r') as f:
            yield readers.trajnet(f.read())
