import json
from .data import SceneRow, TrackRow


def trajnet_tracks(row):
    x = round(row.x, 2)
    y = round(row.y, 2)
    if row.prediction_number is None:
        return json.dumps({'track': {'f': row.frame, 'p': row.pedestrian, 'x': x, 'y': y}})
    else:
        return json.dumps({'track': {'f': row.frame, 'p': row.pedestrian, 'x': x, 'y': y,
                                     'prediction_number': row.prediction_number}})


def trajnet_scenes(row):
    return json.dumps(
        {'scene': {'id': row.scene, 'p': row.pedestrian, 's': row.start, 'e': row.end, 'fps': row.fps, 'tag': row.tag}})


def trajnet(row):
    if isinstance(row, TrackRow):
        return trajnet_tracks(row)
    elif isinstance(row, SceneRow):
        return trajnet_scenes(row)

    raise Exception('unknown row type')
