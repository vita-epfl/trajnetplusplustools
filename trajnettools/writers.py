import json
from .data import SceneRow, TrackRow


def trajnet_tracks_(row):
    return '{r.frame:d} {r.pedestrian:d} {r.x:.2f} {r.y:.2f}'.format(r=row)


def trajnet_scenes_(row):
    return '{r.scene:d} {r.pedestrian:d} {r.start:d} {r.end:d}'.format(r=row)


def trajnet_tracks(row):
    return json.dumps(
        {'track': {'f': row.frame, 'p': row.pedestrian, 'x': round(row.x, 2), 'y': round(row.y, 2)}})


def trajnet_scenes(row):
    return json.dumps(
        {'scene': {'id': row.scene, 'p': row.pedestrian, 's': row.start, 'e': row.end}})


def trajnet(row):
    if isinstance(row, TrackRow):
        return trajnet_tracks(row)
    return trajnet_scenes(row)
