from collections import namedtuple


TrackRow = namedtuple('Row', ['frame', 'pedestrian', 'x', 'y', 'prediction_number'])
TrackRow.__new__.__defaults__ = (None, None, None, None, None)
SceneRow = namedtuple('Row', ['scene', 'pedestrian', 'start', 'end', 'fps', 'tag'])
