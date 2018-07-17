from collections import defaultdict
import json

from .data import SceneRow, TrackRow


def trajnet_(whole_file):
    marked = defaultdict(list)
    others = defaultdict(list)
    for line in whole_file.split('\n'):
        if not line:
            continue
        line = [e for e in line.split(' ') if e != '']
        mark = bool(float(line[4]))
        ped_id = int(float(line[1]))
        row = TrackRow(int(float(line[0])), ped_id, float(line[2]), float(line[3]))
        if mark:
            marked[ped_id].append(row)
        else:
            others[ped_id].append(row)

    return list(marked.values()) + list(others.values())


class TrajnetReader(object):
    def __init__(self, input_file):
        self.tracks_by_frame = defaultdict(list)
        self.scenes_by_id = dict()

        self.read_file(input_file)

    def read_file(self, input_file):
        with open(input_file, 'r') as f:
            for line in f:
                line = json.loads(line)

                track = line.get('track')
                if track is not None:
                    row = TrackRow(track['f'], track['p'], track['x'], track['y'])
                    self.tracks_by_frame[row.frame].append(row)
                    continue

                scene = line.get('scene')
                if scene is not None:
                    row = SceneRow(scene['id'], scene['p'], scene['s'], scene['e'])
                    self.scenes_by_id[row.scene] = row

    def scenes(self):
        for scene_id in self.scenes_by_id:
            yield self.scene(scene_id)

    def scene(self, scene_id):
        scene = self.scenes_by_id.get(scene_id)
        if scene is None:
            raise Exception('scene with that id not found')

        frames = range(scene.start, scene.end + 1)
        track_rows = [r
                      for frame in frames
                      for r in self.tracks_by_frame.get(frame, [])]
        return scene_id, scene.pedestrian, track_rows


def trajnet_tracks(whole_file):
    for line in whole_file.splitlines():
        line = json.loads(line)
        yield TrackRow()
