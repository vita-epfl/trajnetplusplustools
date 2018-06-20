from __future__ import division

import numpy as np


def final_l2(path1_path2):
    row1 = path1_path2[0][-1]
    row2 = path1_path2[1][-1]
    return np.linalg.norm((row2.x - row1.x, row2.y - row1.y))


def average_l2(path1_path2, n_predictions=12):
    path1, path2 = path1_path2
    assert len(path1) >= n_predictions
    assert len(path2) >= n_predictions
    path1 = path1[-n_predictions:]
    path2 = path2[-n_predictions:]

    return sum(np.linalg.norm((r1.x - r2.x, r1.y - r2.y))
               for r1, r2 in zip(path1, path2)) / n_predictions
