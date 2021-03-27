import heapq
import os

import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_metric import _BaseMetric
from .. import _timing


class Detection(_BaseMetric):
    """Class which implements detection metrics.

    Metrics are parameterized by IOU threshold.
    """

    def __init__(self):
        super().__init__()
        self.plottable = False  # TODO
        self.array_labels = np.arange(5, 95 + 1, 5) / 100.
        self.integer_fields = ['Det_Frames', 'Det_Sequences']
        self.integer_array_fields = ['Det_TP', 'Det_FP', 'Det_FN']
        self.float_fields = ['Det_AP_50_95']
        # TODO: The name Det_MODP_sum with mixed case and underscores is not pretty?
        self.float_array_fields = ['Det_AP', 'Det_AP_sum',
                                   'Det_MODA', 'Det_MODP', 'Det_MODP_sum', 'Det_FAF',
                                   'Det_Re', 'Det_Pr', 'Det_F1']
        self.fields = self.integer_fields + self.integer_array_fields + self.float_array_fields
        self.summary_fields = ['Det_AP', 'Det_AP_50_95',
                               'Det_MODA', 'Det_MODP', 'Det_FAF',
                               'Det_Re', 'Det_Pr', 'Det_F1']

        self._summed_fields = self.integer_fields + self.integer_array_fields + ['Det_AP_sum', 'Det_MODP_sum']

    @_timing.time
    def eval_sequence(self, data):
        """Calculates CLEAR metrics for one sequence"""
        # Initialise results
        res = {}
        for field in self.fields:
            if field in self.integer_array_fields:
                res[field] = np.zeros((len(self.array_labels),), dtype=np.int)
            elif field in self.float_array_fields:
                res[field] = np.zeros((len(self.array_labels),), dtype=np.float)
            else:
                res[field] = 0
        res['Det_Frames'] = data['num_timesteps']
        res['Det_Sequences'] = 1

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
            res['Det_AP_sum'][i] = _compute_average_precision(data['num_gt_dets'], scores, correct)

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
                overlap_mask = (similarity >= sim_threshold)
                score_mat = overlap_mask.astype(np.float) + eps * (similarity * overlap_mask)
                # Hungarian algorithm to find best matches
                match_rows, match_cols = linear_sum_assignment(-score_mat)
                num_matches = np.sum(overlap_mask[match_rows, match_cols])
                # Ensure that similarity could not have overwhelmed a match.
                assert np.sum(score_mat[match_rows, match_cols]) - num_matches < 1

                # Calculate and accumulate basic statistics
                res['Det_TP'][i] += num_matches
                res['Det_FN'][i] += len(gt_ids_t) - num_matches
                res['Det_FP'][i] += len(tracker_ids_t) - num_matches
                res['Det_MODP_sum'][i] += np.sum(similarity[match_rows, match_cols])

        res = self._compute_final_fields(res)
        return res

    @staticmethod
    def _compute_final_fields(res):
        """Calculate sub-metric ('field') values which only depend on other sub-metric values.
        This function is used both for both per-sequence calculation, and in combining values across sequences.
        """
        res = dict(res)
        res['Det_AP'] = res['Det_AP_sum'] / res['Det_Sequences']
        res['Det_AP_50_95'] = np.mean(res['Det_AP'][(np.arange(50, 95 + 1, 5) - 5) // 5])
        res['Det_MODA'] = (res['Det_TP'] - res['Det_FP']) / np.maximum(1.0, res['Det_TP'] + res['Det_FN'])
        res['Det_MODP'] = res['Det_MODP_sum'] / np.maximum(1.0, res['Det_TP'])
        res['Det_Re'] = res['Det_TP'] / np.maximum(1.0, res['Det_TP'] + res['Det_FN'])
        res['Det_Pr'] = res['Det_TP'] / np.maximum(1.0, res['Det_TP'] + res['Det_FP'])
        res['Det_F1'] = res['Det_TP'] / (
                np.maximum(1.0, res['Det_TP'] + 0.5*res['Det_FN'] + 0.5*res['Det_FP']))
        res['Det_FAF'] = res['Det_FP'] / res['Det_Frames']
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


class DetectionConfidence(_BaseMetric):
    """Class which implements detection metrics using confidence scores.

    Metrics are parameterized by recall.
    """

    def __init__(self):
        super().__init__()
        self.plottable = False  # TODO
        # TODO: Use sample frequency of 0.01 instead of 0.1? (virtually free)
        # TODO: Eliminate 0% recall? (noisy)
        self.array_labels = np.arange(0, 100 + 1, 10) / 100.  # Recall thresholds.
        self.integer_fields = ['Det_Sequences_PrAtRe']
        self.float_array_fields = ['Det_PrAtRe', 'Det_PrAtRe_sum']
        self.fields = self.integer_fields + self.float_array_fields
        self.summary_fields = ['Det_PrAtRe']

        self._summed_fields = self.integer_fields + ['Det_PrAtRe_sum']

        self.threshold = 0.5

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
        res['Det_Sequences_PrAtRe'] = 1

        # Find per-frame correspondence by priority of score.
        correct = [None for _ in range(data['num_timesteps'])]
        for t in range(data['num_timesteps']):
            correct[t] = _match_by_score(data['tracker_confidences'][t],
                                         data['similarity_scores'][t],
                                         self.threshold)
        # Concatenate results from all frames to compute AUC.
        scores = np.concatenate(data['tracker_confidences'])
        correct = np.concatenate(correct)
        res['Det_PrAtRe_sum'] = _find_prec_at_recall(data['num_gt_dets'], scores, correct, self.array_labels)

        res = self._compute_final_fields(res)
        return res

    @staticmethod
    def _compute_final_fields(res):
        """Calculate sub-metric ('field') values which only depend on other sub-metric values.
        This function is used both for both per-sequence calculation, and in combining values across sequences.
        """
        res = dict(res)
        res['Det_PrAtRe'] = res['Det_PrAtRe_sum'] / res['Det_Sequences_PrAtRe']
        return res

    def combine_sequences(self, all_res):
        # TODO: Compute precision-recall curve for detections in all sequences together?
        # TODO: Plot "fine" precision-recall curve?
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

    # pr_matched = np.full(confidence.shape, False)
    # feasible = (similarity >= similarity_threshold)
    # for pr_id in order:
    #     if not np.any(feasible[:, pr_id]):
    #         continue
    #     subset, = np.nonzero(feasible[:, pr_id])
    #     gt_id = subset[np.argmax(similarity[subset, pr_id])]
    #     feasible[gt_id, h] = False
    #     pr_matched[pr_id] = True

    # similarity = similarity[:, order].T
    # feasible = (similarity >= similarity_threshold)
    # pr_matched = np.full(confidence.shape, False)
    # gt_subset = np.arange(num_gt)
    # for pr_id in range(num_pr):
    #     feasible_inds, = np.nonzero(feasible[0, :])
    #     if len(feasible_inds) > 0:
    #         ind = feasible_inds[np.argmax(similarity[0, feasible_inds])]
    #         gt_id = gt_subset[ind]
    #         # Eliminate gt_id from matrix.
    #         keep_mask = (gt_subset != gt_id)
    #         gt_subset = gt_subset[keep_mask]
    #         feasible = feasible[1:, keep_mask]
    #     else:
    #         feasible = feasible[1:, :]
    #     pr_matched[pr_id] = True

    # Construct priority queue for each prediction.
    feasible = (similarity >= similarity_threshold)
    candidates = {}
    for pr_id in range(num_pr):
        gt_subset, = np.nonzero(feasible[:, pr_id])
        candidates[pr_id] = [(-similarity[gt_id, pr_id], gt_id) for gt_id in gt_subset]
        heapq.heapify(candidates[pr_id])
    # Proceed by popping elements from priority queue.
    gt_matched = [False for _ in range(num_gt)]
    pr_matched = [False for _ in range(num_pr)]
    for pr_id in order:
        while not pr_matched[pr_id] and candidates[pr_id]:
            _, gt_id = candidates[pr_id][0]  # Take element at top of heap.
            # Skip elements that are already matched.
            if gt_matched[gt_id]:
                heapq.heappop(candidates[pr_id])
                continue
            pr_matched[pr_id] = True
            gt_matched[gt_id] = True

    # # Make multiple matches at once to reduce iterations in python.
    # pr_matched = np.full(confidence.shape, False)
    # gt_subset = np.arange(num_gt)
    # pr_subset = np.arange(num_pr)
    # feasible = (similarity >= similarity_threshold)
    # feasible_confidence = np.where(feasible, confidence[np.newaxis, :], -np.inf)
    # similarity = np.where(feasible, similarity, -np.inf)
    # # Take feasible subset.
    # gt_mask = np.any(feasible, axis=1)
    # pr_mask = np.any(feasible, axis=0)
    # gt_subset, pr_subset = gt_subset[gt_mask], pr_subset[pr_mask]
    # feasible = feasible[gt_mask, :][:, pr_mask]
    # feasible_confidence = feasible_confidence[gt_mask, :][:, pr_mask]
    # similarity = similarity[gt_mask, :][:, pr_mask]
    # while np.size(feasible):
    #     # Find max score prediction for each ground-truth.
    #     # Find max similarity ground-truth for each prediction.
    #     # All exclusive matches can be made immediately.
    #     num_rows, num_cols = feasible.shape
    #     is_best_for_gt = np.full(feasible.shape, False)
    #     is_best_for_pr = np.full(feasible.shape, False)
    #     best_for_gt = np.argmax(feasible_confidence, axis=1)
    #     best_for_pr = np.argmax(similarity, axis=0)
    #     is_best_for_gt[np.arange(num_rows), best_for_gt] = True
    #     is_best_for_pr[best_for_pr, np.arange(num_cols)] = True
    #     is_match = (is_best_for_gt & is_best_for_pr)
    #     assert np.sum(is_match) > 0
    #     gt_is_match = np.any(is_match, axis=1)
    #     pr_is_match = np.any(is_match, axis=0)
    #     # Update list of matched elements.
    #     pr_matched[pr_subset[pr_is_match]] = True
    #     # Eliminate matched elements.
    #     gt_mask = ~gt_is_match
    #     pr_mask = ~pr_is_match
    #     gt_subset, pr_subset = gt_subset[gt_mask], pr_subset[pr_mask]
    #     feasible = feasible[gt_mask, :][:, pr_mask]
    #     feasible_confidence = feasible_confidence[gt_mask, :][:, pr_mask]
    #     similarity = similarity[gt_mask, :][:, pr_mask]
    #     # Eliminate elements without a feasible match.
    #     gt_mask = np.any(feasible, axis=1)
    #     pr_mask = np.any(feasible, axis=0)
    #     gt_subset, pr_subset = gt_subset[gt_mask], pr_subset[pr_mask]
    #     feasible = feasible[gt_mask, :][:, pr_mask]
    #     feasible_confidence = feasible_confidence[gt_mask, :][:, pr_mask]
    #     similarity = similarity[gt_mask, :][:, pr_mask]

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
    return np.dot(prec[1:], (recall[1:] - recall[:-1]))
