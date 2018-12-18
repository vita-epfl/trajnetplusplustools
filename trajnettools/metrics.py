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


def collision(path1, path2, n_predictions=12, person_radius=0.1):
    """Check if there is collision or not"""

    assert len(path1) >= n_predictions
    path1 = path1[-n_predictions:]
    frames1 = set(f1.frame for f1 in path1)
    frames2 = set(f2.frame for f2 in path2)
    common_frames = frames1.intersection(frames2)

    # If there is no interaction, there is no collision
    if not common_frames:
        return False

    path1 = [path1[i] for i in range(len(path1)) if path1[i].frame in common_frames]
    path2 = [path2[i] for i in range(len(path2)) if path2[i].frame in common_frames]

    def getinsidepoints(p1, p2, parts=10):
        """return: equally distanced points between starting and ending "control" points"""

        return np.array((np.linspace(p1[0], p2[0], parts + 1),
                         np.linspace(p1[1], p2[1], parts + 1)))

    for i in range(len(path1) - 1):
        p1, p2 = [path1[i].x, path1[i].y], [path1[i + 1].x, path1[i + 1].y]
        p3, p4 = [path2[i].x, path2[i].y], [path2[i + 1].x, path2[i + 1].y]
        if np.min(np.linalg.norm(getinsidepoints(p1, p2) - getinsidepoints(p3, p4), axis=0)) <= 2 * person_radius:
            return True

    return False
