from __future__ import division

import numpy as np


def final_l2(path1, path2):
    row1 = path1[-1]
    row2 = path2[-1]
    return np.linalg.norm((row2.x - row1.x, row2.y - row1.y))


def average_l2(path1, path2, n_predictions=12):
    assert len(path1) >= n_predictions
    assert len(path2) >= n_predictions
    path1 = path1[-n_predictions:]
    path2 = path2[-n_predictions:]

    return sum(np.linalg.norm((r1.x - r2.x, r1.y - r2.y))
               for r1, r2 in zip(path1, path2)) / n_predictions


def collision(path1, path2, n_predictions=12, person_radius=0.3):
    """Check if there is collision or not"""

    assert len(path1) >= n_predictions
    path1 = path1[-n_predictions:]
    path2 = path2[-n_predictions:]

    def getinsidepoints(p1, p2, parts=10):
        """return: equally distanced points between starting and ending "control" points"""

        return np.array((np.linspace(p1[0], p2[0], parts + 1),
                         np.linspace(p1[1], p2[1], parts + 1)))

    common_frames = np.where(np.array(list(r1.frame == r2.frame for r1 in path1 for r2 in path2)) > 0)[0]
    # If there is no interaction, there is no collision
    if len(common_frames) == 0:
        return 0

    index_path1 = np.floor(common_frames / n_predictions).astype(int).tolist()  # get index for
    index_path2 = (common_frames % n_predictions).tolist()                      # both paths

    path1 = path1[index_path1[0]:index_path1[-1] + 1]  # reduced path to have
    path2 = path2[index_path2[0]:index_path2[-1] + 1]  # only potential collisions
    for i in range(len(path1) - 1):
        p1, p2 = [path1[i].x, path1[i].y], [path1[i + 1].x, path1[i + 1].y]
        p3, p4 = [path2[i].x, path2[i].y], [path2[i + 1].x, path2[i + 1].y]
        if np.min(np.linalg.norm(getinsidepoints(p1, p2) - getinsidepoints(p3, p4), axis=0)) <= 2*person_radius:
            return 1

    return 0