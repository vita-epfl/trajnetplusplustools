CHUNK_STRIDE = 5  # rows
CHUNK_SIZE = 20  # rows


def euclidean_distance_2(row1, row2):
    return (row1.x - row2.x)**2 + (row1.y - row2.y)**2


def to_scenes(rows):
    by_pedestrian = rows.groupBy(lambda r: r.pedestrian).cache()

    # scenes: pedestrian of interest, [(start frame, end frame)]
    scenes = (by_pedestrian
              .filter(lambda p_path: len(p_path[1]) >= CHUNK_SIZE)
              .flatMapValues(lambda path: [
                  (path[i].frame, path[i+CHUNK_SIZE].frame)
                  for i in range(0,
                                 len(path) - CHUNK_SIZE,
                                 CHUNK_STRIDE)
                  # filter for pedestrians moving by more than 1 meter
                  if euclidean_distance_2(path[i], path[i+CHUNK_SIZE]) > 1.0
              ]))

    # output
    for scene_id, (ped_id, (start, end)) in enumerate(scenes.collect()):
        scene_rows = (rows
                      .filter(lambda r: start <= r.frame <= end)
                      .sortBy(lambda r: (r.pedestrian, r.frame)))
        yield scene_id, ped_id, scene_rows
