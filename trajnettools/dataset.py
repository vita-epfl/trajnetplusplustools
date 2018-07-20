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
