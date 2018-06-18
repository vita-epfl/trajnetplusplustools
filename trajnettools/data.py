from collections import namedtuple


Row = namedtuple('Row', ['frame', 'pedestrian', 'x', 'y'])


MarkedRow = namedtuple('MarkedRow', ['frame', 'pedestrian', 'x', 'y', 'mark'])
