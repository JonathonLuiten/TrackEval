import os

import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_metric import _BaseMetric
from .. import _timing


class Det(_BaseMetric):
    """Implements detection metrics.

    The array-valued metrics use confidence-based matching and are parameterized by recall.
    """

    def __init__(self):
        super().__init__()
        self.plottable = False
        # Use array labels for recall.
        # TODO: Eliminate 0% recall? (noisy if not using max-prec)
        self.array_labels = np.arange(0, 100 + 1) / 100.
        self.integer_fields = ['Det_Frames', 'Det_Sequences', 'Det_TP', 'Det_FP', 'Det_FN']
        self.float_fields = ['Det_AP', 'Det_AP_sum', 'Det_AP_coarse',
                             'Det_MODA', 'Det_MODP', 'Det_MODP_sum', 'Det_FAF',
                             'Det_Re', 'Det_Pr', 'Det_F1']
        self.float_array_fields = ['Det_PrAtRe', 'Det_PrAtRe_sum']
        self.fields = self.integer_fields + self.float_fields + self.float_array_fields
        self.summary_fields = ['Det_AP', 'Det_PrAtRe', 'Det_AP_coarse',
                               'Det_MODA', 'Det_MODP', 'Det_FAF',
                               'Det_Re', 'Det_Pr', 'Det_F1']

        self.threshold = 0.5
        self._summed_fields = self.integer_fields + ['Det_AP_sum', 'Det_MODP_sum', 'Det_PrAtRe_sum']

    @_timing.time
    def eval_sequence(self, data):
        """Calculates CLEAR metrics for one sequence"""
        # Initialise results
        res = {}
        for field in self.fields:
            if field in self.float_array_fields:
                res[field] = np.zeros((len(self.array_labels),), dtype=np.float)
            else:
                res[field] = 0
        res['Det_Frames'] = data['num_timesteps']
        res['Det_Sequences'] = 1

        # Find per-frame correspondence by priority of score.
        correct = [None for _ in range(data['num_timesteps'])]
        for t in range(data['num_timesteps']):
            correct[t] = _match_by_score(data['tracker_confidences'][t],
                                         data['similarity_scores'][t],
                                         self.threshold)
        # Concatenate results from all frames to compute AUC.
        scores = np.concatenate(data['tracker_confidences'])
        correct = np.concatenate(correct)
        # TODO: Compute AUC without taking max precision with at least given recall?
        # TODO: Compute precision-recall curve over all sequences, not per sequence?
        # MOT Challenge seems to do it per-sequence.
        res['Det_AP_sum'] = _compute_average_precision(data['num_gt_dets'], scores, correct)
        # TODO: Take max precision with at least given recall?
        res['Det_PrAtRe_sum'] = _find_prec_at_recall(data['num_gt_dets'], scores, correct,
                                                     self.array_labels)

        # Find per-frame correspondence (without accounting for switches).
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            # Deal with the case that there are no gt_det/tracker_det in a timestep.
            if len(gt_ids_t) == 0:
                res['Det_FP'] += len(tracker_ids_t)
                continue
            if len(tracker_ids_t) == 0:
                res['Det_FN'] += len(gt_ids_t)
                continue

            # Construct score matrix to optimize number of matches and then localization.
            similarity = data['similarity_scores'][t]
            assert np.all(~(similarity < 0))
            assert np.all(~(similarity > 1))
            eps = 1. / (max(similarity.shape) + 1.)
            overlap_mask = (similarity >= self.threshold)
            score_mat = overlap_mask.astype(np.float) + eps * (similarity * overlap_mask)
            # Hungarian algorithm to find best matches
            match_rows, match_cols = linear_sum_assignment(-score_mat)
            num_matches = np.sum(overlap_mask[match_rows, match_cols])
            # Ensure that similarity could not have overwhelmed a match.
            delta = np.sum(score_mat[match_rows, match_cols]) - num_matches
            assert 0 <= delta
            assert delta < 1

            # Calculate and accumulate basic statistics
            res['Det_TP'] += num_matches
            res['Det_FN'] += len(gt_ids_t) - num_matches
            res['Det_FP'] += len(tracker_ids_t) - num_matches
            res['Det_MODP_sum'] += np.sum(similarity[match_rows, match_cols])

        res = self._compute_final_fields(res)
        return res

    @staticmethod
    def _compute_final_fields(res):
        """Calculate sub-metric ('field') values which only depend on other sub-metric values.
        This function is used both for both per-sequence calculation, and in combining values across sequences.
        """
        res = dict(res)
        res['Det_AP'] = res['Det_AP_sum'] / res['Det_Sequences']
        res['Det_PrAtRe'] = res['Det_PrAtRe_sum'] / res['Det_Sequences']
        # Coarse integral matches implementation in matlab toolkit.
        # TODO: Eliminate 0% recall? (see above)
        res['Det_AP_coarse'] = np.mean(res['Det_PrAtRe'][::10])
        res['Det_MODA'] = (res['Det_TP'] - res['Det_FP']) / np.maximum(1.0, res['Det_TP'] + res['Det_FN'])
        res['Det_MODP'] = res['Det_MODP_sum'] / np.maximum(1.0, res['Det_TP'])
        res['Det_FAF'] = res['Det_FP'] / res['Det_Frames']
        res['Det_Re'] = res['Det_TP'] / np.maximum(1.0, res['Det_TP'] + res['Det_FN'])
        res['Det_Pr'] = res['Det_TP'] / np.maximum(1.0, res['Det_TP'] + res['Det_FP'])
        res['Det_F1'] = res['Det_TP'] / (
                np.maximum(1.0, res['Det_TP'] + 0.5*res['Det_FN'] + 0.5*res['Det_FP']))
        return res

    def combine_sequences(self, all_res):
        res = {}
        for field in self._summed_fields:
            res[field] = self._combine_sum(all_res, field)
        res = self._compute_final_fields(res)
        return res

    def combine_classes_det_averaged(self, all_res):
        # TODO: Implement.
        raise NotImplementedError

    def combine_classes_class_averaged(self, all_res):
        # TODO: Implement.
        raise NotImplementedError


class DetLoc(_BaseMetric):
    """Implements detection metrics.

    Metrics are parameterized by IOU threshold.
    """

    def __init__(self):
        super().__init__()
        self.plottable = False  # TODO
        self.array_labels = np.arange(5, 95 + 1, 5) / 100.
        self.integer_fields = ['DetLoc_Sequences']
        self.float_fields = ['DetLoc_AP_50', 'DetLoc_AP_75', 'DetLoc_AP_50_95']
        self.float_array_fields = ['DetLoc_AP', 'DetLoc_AP_sum']
        self.fields = self.integer_fields + self.float_fields + self.float_array_fields
        self.summary_fields = ['DetLoc_AP', 'DetLoc_AP_50', 'DetLoc_AP_75', 'DetLoc_AP_50_95']

        self._summed_fields = self.integer_fields + ['DetLoc_AP_sum']

    @_timing.time
    def eval_sequence(self, data):
        """Calculates CLEAR metrics for one sequence"""
        # Initialise results
        res = {}
        for field in self.fields:
            if field in self.float_array_fields:
                res[field] = np.zeros((len(self.array_labels),), dtype=np.float)
            else:
                res[field] = 0
        res['DetLoc_Sequences'] = 1

        # Find per-frame correspondence (without accounting for switches).
        for i, sim_threshold in enumerate(self.array_labels):
            # Find per-frame correspondence by priority of score.
            correct = [None for _ in range(data['num_timesteps'])]
            for t in range(data['num_timesteps']):
                correct[t] = _match_by_score(data['tracker_confidences'][t],
                                             data['similarity_scores'][t],
                                             sim_threshold)
            # Concatenate results from all frames to compute AUC.
            scores = np.concatenate(data['tracker_confidences'])
            correct = np.concatenate(correct)
            # TODO: Compute precision-recall curve over all sequences, not per sequence?
            # MOT Challenge seems to do it per-sequence.
            res['DetLoc_AP_sum'][i] = _compute_average_precision(data['num_gt_dets'], scores, correct)

        res = self._compute_final_fields(res)
        return res

    @staticmethod
    def _compute_final_fields(res):
        """Calculate sub-metric ('field') values which only depend on other sub-metric values.
        This function is used both for both per-sequence calculation, and in combining values across sequences.
        """
        res = dict(res)
        res['DetLoc_AP'] = res['DetLoc_AP_sum'] / res['DetLoc_Sequences']
        res['DetLoc_AP_50'] = res['DetLoc_AP'][(50 - 5) // 5]
        res['DetLoc_AP_75'] = res['DetLoc_AP'][(75 - 5) // 5]
        res['DetLoc_AP_50_95'] = np.mean(res['DetLoc_AP'][(np.arange(50, 95 + 1, 5) - 5) // 5])
        return res

    def combine_sequences(self, all_res):
        res = {}
        for field in self._summed_fields:
            res[field] = self._combine_sum(all_res, field)
        res = self._compute_final_fields(res)
        return res

    def combine_classes_det_averaged(self, all_res):
        # TODO: Implement.
        raise NotImplementedError

    def combine_classes_class_averaged(self, all_res):
        # TODO: Implement.
        raise NotImplementedError


def _match_by_score(confidence, similarity, similarity_threshold):
    """Matches by priority of confidence.

    Args:
        confidence: Array of shape [num_pred].
        similarity: Array of shape [num_gt, num_pred].
        similarity_threshold: Scalar constant.

    Returns:
        Array of bools indicating whether prediction was matched.

    Assumes confidence scores are unique.
    """
    num_gt, num_pr = similarity.shape
    # Sort descending by confidence, preserve order of repeated elements.
    order = np.argsort(-confidence, kind='stable')

    # Sort gt decreasing by similarity for each pred.
    # Note: Could save some time by re-using for different thresholds.
    gt_order = np.argsort(-similarity, kind='stable', axis=0)
    # Create a sorted list of matches for each prediction.
    feasible = (similarity >= similarity_threshold)
    candidates = {}
    for pr_id, conf in enumerate(confidence):
        # Take feasible subset.
        candidates[pr_id] = gt_order[feasible[gt_order[:, pr_id], pr_id], pr_id]
    gt_matched = [False for _ in range(num_gt)]
    pr_matched = [False for _ in range(num_pr)]
    for pr_id in order:
        # Find first gt_id which is still available.
        for gt_id in candidates[pr_id]:
            if gt_matched[gt_id]:
                continue
            gt_matched[gt_id] = True
            pr_matched[pr_id] = True
            break

    return pr_matched


def _find_prec_at_recall(num_gt, scores, correct, thresholds):
    """Computes precision at a given minimum recall threshold.

    Args:
        num_gt: Number of ground-truth elements.
        scores: Score of each prediction.
        correct: Whether or not each prediction is correct.
        thresholds: Recall thresholds at which to evaluate.

    Follows implementation from Piotr Dollar toolbox.
    """
    # Sort descending by score.
    order = np.argsort(-scores, kind='stable')
    correct = correct[order]
    # Note: cumsum() does not include operating point with zero predictions.
    # However, this matches the original implementation.
    tp = np.cumsum(correct)
    num_pred = 1 + np.arange(len(scores))
    recall = np.true_divide(tp, num_gt)
    prec = np.true_divide(tp, num_pred)
    # Extend curve to infinity with zeros.
    recall = np.concatenate([recall, [np.inf]])
    prec = np.concatenate([prec, [0.]])
    assert np.all(np.isfinite(thresholds)), 'assume finite thresholds'
    # Find first element with minimum recall.
    # Use argmax() to take first element that satisfies criterion.
    # TODO: Use maximum precision at equal or better recall?
    # https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
    return np.asarray([prec[np.argmax(recall >= threshold)] for threshold in thresholds])


def _compute_average_precision(num_gt, scores, correct):
    # TODO: Could use sklearn.average_precision_score?
    # However, it doesn't seem to have nice support for:
    # (1) detections that were completely missed
    # (2) max precision at equal or greater recall

    # Sort descending by score.
    order = np.argsort(-scores, kind='stable')
    correct = correct[order]
    tp = np.cumsum(correct)
    num_pred = 1 + np.arange(len(scores))
    recall = np.true_divide(tp, num_gt)
    prec = np.true_divide(tp, num_pred)
    # Extend recall to [0, 1].
    recall = np.concatenate([[0.], recall, [1.]])
    prec = np.concatenate([[np.nan], prec, [0.]])  # The nan will not be used.
    # Take max precision available at equal or greater recall.
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    # Take integral.
    assert np.all(np.diff(recall) >= 0)
    return np.dot(prec[1:], recall[1:] - recall[:-1])
