import numpy as np
import pykalman

from .data import Row


def predict(rows):
    initial_state_mean = [rows[0].x, 0, rows[0].y, 0]

    transition_matrix = [[1, 1, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 1],
                         [0, 0, 0, 1]]

    observation_matrix = [[1, 0, 0, 0],
                          [0, 0, 1, 0]]

    kf = pykalman.KalmanFilter(transition_matrices=transition_matrix,
                               observation_matrices=observation_matrix,
                               transition_covariance=1e-5 * np.eye(4),
                               observation_covariance=0.25**2 * np.eye(2),
                               initial_state_mean=initial_state_mean)
    # kf.em([(r.x, r.y) for r in rows[:9]], em_vars=['transition_matrices',
    #                                                'observation_matrices'])
    kf.em([(r.x, r.y) for r in rows[:9]])
    observed_states, _ = kf.smooth([(r.x, r.y) for r in rows[:9]])

    # prepare predictions
    frame_diff = rows[1].frame - rows[0].frame
    first_frame = rows[-1].frame + frame_diff
    ped_id = rows[-1].pedestrian

    # sample predictions (first sample corresponds to last state)
    # average 5 sampled predictions
    predictions = None
    for _ in range(5):
        _, pred = kf.sample(12 + 1, initial_state=observed_states[-1])
        if predictions is None:
            predictions = pred
        else:
            predictions += pred
    predictions /= 5.0

    return [Row(first_frame + i * frame_diff, ped_id, x, y)
            for i, (x, y) in enumerate(predictions[1:])]
