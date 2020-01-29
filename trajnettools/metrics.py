from __future__ import division
import numpy as np
from scipy.stats import gaussian_kde

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

    def getinsidepoints(p1, p2, parts=2):
        """return: equally distanced points between starting and ending "control" points"""

        return np.array((np.linspace(p1[0], p2[0], parts + 1),
                         np.linspace(p1[1], p2[1], parts + 1)))

    for i in range(len(path1) - 1):
        p1, p2 = [path1[i].x, path1[i].y], [path1[i + 1].x, path1[i + 1].y]
        p3, p4 = [path2[i].x, path2[i].y], [path2[i + 1].x, path2[i + 1].y]
        if np.min(np.linalg.norm(getinsidepoints(p1, p2) - getinsidepoints(p3, p4), axis=0)) \
           <= 2 * person_radius:
            return True

    return False

def topk(self, primary_tracks, ground_truth, n_predictions=12, topk=3):
    ## TopK multimodal 

    l2 = 1e10
    ## preds: Pred_len x Num_preds x 2
    for pred_num in range(topk):
        primary_prediction = [t for t in primary_tracks if t.prediction_number == pred_num]
        tmp_score = final_l2(ground_truth, primary_prediction)
        if tmp_score < l2:      
            l2 = tmp_score 
            topk_fde = tmp_score
            topk_ade = average_l2(ground_truth, primary_prediction, n_predictions=n_predictions)

    return topk_ade, topk_fde

def nll(self, primary_tracks, ground_truth, n_predictions=12, log_pdf_lower_bound=-20):
    ## Inspired from Boris.
    gt = np.array([[t.x, t.y] for t in ground_truth][-n_predictions:])
    frame_gt = [t.frame for t in ground_truth][-n_predictions:]
    preds = np.array([[[t.x, t.y] for t in primary_tracks if t.frame == frame] for frame in frame_gt])
    ## preds: Pred_len x Num_preds x 2

    ## To verify if 100 predictions
    if preds.shape[1] != 100:
        raise Exception('Need 100 predictions')

    pred_len = len(frame_gt)

    ll = 0.0
    same_pred = 0
    for timestep in range(pred_len):
        curr_gt = gt[timestep]
        try:
            scipy_kde = gaussian_kde(preds[timestep].T)
            # We need [0] because it's a (1,)-shaped numpy array.
            log_pdf = np.clip(scipy_kde.logpdf(curr_gt.T), a_min=log_pdf_lower_bound, a_max=None)[0]
            ll += log_pdf
        except:
            same_pred += 1

    if same_pred == pred_len:
        raise Exception('All 100 Predictions are Identical')

    ll = ll / (pred_len - same_pred)
    return ll