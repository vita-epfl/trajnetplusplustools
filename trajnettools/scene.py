CHUNK_DURATION = 8  # seconds (at 25fps this is 200 frames)
CHUNK_STRIDE = 2  # seconds (at 25fps this is 50 frames)


def euclidean_distance_2(row1, row2):
    return (row1.x - row2.x)**2 + (row1.y - row2.y)**2


def to_scenes(rows, frame_rate=25.0):
    by_pedestrian = rows.groupBy(lambda r: r.pedestrian).cache()

    # scenes: pedestrian of interest, start frame
    min_frames = int(CHUNK_DURATION * frame_rate)
    scenes = (by_pedestrian
              .mapValues(lambda path: (min(r.frame for r in path),
                                       max(r.frame for r in path)))
              .filter(lambda p_frames: p_frames[1][1] - p_frames[1][0] >= min_frames)
              .flatMapValues(lambda frames: list(range(
                  frames[0],
                  frames[1] - min_frames + 1,
                  int(CHUNK_STRIDE * frame_rate)))))

    def d2_frames(frame1, frame2, path):
        row1 = next(iter(r for r in path if r.frame >= frame1))
        row2 = next(iter(r for r in path if r.frame >= frame2))
        return euclidean_distance_2(row1, row2)

    # filter scenes for moving pedestrian (>1m in first 4seconds)
    scenes = (scenes
              .leftOuterJoin(by_pedestrian)
              .mapValues(lambda sf_path: (sf_path[0], d2_frames(sf_path[0], sf_path[0] + 4.0 * frame_rate, sf_path[1])))
              .flatMapValues(lambda sf_ed2: [sf_ed2[0]] if sf_ed2[1] > 1.0 else [])
              .collect())

    # output
    for scene_id, (ped_id, start_frame) in enumerate(scenes):
        scene_rows = (rows
                      .filter(lambda r: start_frame <= r.frame < start_frame + 200)
                      .sortBy(lambda r: (r.pedestrian, r.frame)))
        yield scene_id, ped_id, scene_rows
