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
        self.plottable = True
        self.array_labels = self._get_array_labels()
        self.integer_fields = ['Det_Frames', 'Det_Sequences', 'Det_TP', 'Det_FP', 'Det_FN']
        self.float_fields = ['Det_AP', 'Det_AP_sum', 'Det_AP_union', 'Det_AP_coarse', 'Det_AP_coarse_naive',
                             'Det_MODA', 'Det_MODP', 'Det_MODP_sum', 'Det_FAF',
                             'Det_Re', 'Det_Pr', 'Det_F1',
                             'Det_num_gt_dets']
        self.float_array_fields = ['Det_PrAtRe', 'Det_PrAtRe_sum',
                                   'Det_PrAtRe_naive', 'Det_PrAtRe_naive_sum',
                                   'Det_PrAtRe_union']
        self.fields = self.integer_fields + self.float_fields + self.float_array_fields
        self.summary_fields = ['Det_AP', 'Det_PrAtRe', 'Det_AP_coarse',
                               'Det_PrAtRe_naive', 'Det_AP_coarse_naive',
                               'Det_AP_union', 'Det_PrAtRe_union',
                               'Det_MODA', 'Det_MODP', 'Det_FAF',
                               'Det_Re', 'Det_Pr', 'Det_F1']

        self.threshold = 0.5
        self.summed_fields = (
                self.integer_fields +
                ['Det_AP_sum', 'Det_MODP_sum', 'Det_PrAtRe_sum', 'Det_PrAtRe_naive_sum',
                 'Det_num_gt_dets'])
        self.concat_fields = ['Det_scores', 'Det_correct']

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
        # MOT Challenge devkit computes AP per sequence, not over all sequences.
        res['Det_AP_sum'] = _compute_average_precision(data['num_gt_dets'], scores, correct)
        res['Det_PrAtRe_sum'] = _find_max_prec_at_recall(
                data['num_gt_dets'], scores, correct, self.array_labels)
        # TODO: Remove naive precision after comparison.
        res['Det_PrAtRe_naive_sum'] = _find_prec_at_recall(
                data['num_gt_dets'], scores, correct, self.array_labels)
        # Take union of all sequences for precision-recall curve.
        res['Det_num_gt_dets'] = data['num_gt_dets']
        res['Det_scores'] = scores
        res['Det_correct'] = correct

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
    def _get_array_labels():
        return np.arange(0, 100 + 1) / 100.

    @classmethod
    def _compute_final_fields(cls, res):
        """Calculate sub-metric ('field') values which only depend on other sub-metric values.
        This function is used both for both per-sequence calculation, and in combining values across sequences.
        """
        res = dict(res)
        # TODO: Keep only one AP metric?
        res['Det_AP'] = res['Det_AP_sum'] / res['Det_Sequences']
        res['Det_AP_union'] = _compute_average_precision(
                res['Det_num_gt_dets'], res['Det_scores'], res['Det_correct'])
        res['Det_PrAtRe_union'] = _find_max_prec_at_recall(
                res['Det_num_gt_dets'], res['Det_scores'], res['Det_correct'],
                cls._get_array_labels())
        res['Det_PrAtRe'] = res['Det_PrAtRe_sum'] / res['Det_Sequences']
        res['Det_PrAtRe_naive'] = res['Det_PrAtRe_naive_sum'] / res['Det_Sequences']
        # TODO: This average may assign too much importance to 0% and 100% recall?
        # res['Det_AP_coarse'] = np.mean(res['Det_PrAtRe'][::10], )
        # res['Det_AP_coarse_naive'] = np.mean(res['Det_PrAtRe_naive'][::10])
        res['Det_AP_coarse'] = np.trapz(res['Det_PrAtRe'][::10], dx=0.1)
        res['Det_AP_coarse_naive'] = np.trapz(res['Det_PrAtRe_naive'][::10], dx=0.1)
        res['Det_MODA'] = (res['Det_TP'] - res['Det_FP']) / np.maximum(1.0, res['Det_TP'] + res['Det_FN'])
        res['Det_MODP'] = res['Det_MODP_sum'] / np.maximum(1.0, res['Det_TP'])
        res['Det_FAF'] = res['Det_FP'] / res['Det_Frames']
        res['Det_Re'] = res['Det_TP'] / np.maximum(1.0, res['Det_TP'] + res['Det_FN'])
        res['Det_Pr'] = res['Det_TP'] / np.maximum(1.0, res['Det_TP'] + res['Det_FP'])
        res['Det_F1'] = res['Det_TP'] / (
                np.maximum(1.0, res['Det_TP'] + 0.5*res['Det_FN'] + 0.5*res['Det_FP']))
        assert np.all(res['Det_PrAtRe_naive'] <= res['Det_PrAtRe'])
        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        for field in self.summed_fields:
            res[field] = self._combine_sum(all_res, field)
        for field in self.concat_fields:
            res[field] = np.concatenate([all_res[k][field] for k in all_res.keys()])
        res = self._compute_final_fields(res)
        return res

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""
        return self.combine_sequences(all_res)

    def combine_classes_class_averaged(self, all_res):
        raise NotImplementedError

    def plot_single_tracker_results(self, table_res, tracker, cls, output_folder):
        """Create plot of results"""

        # Only loaded when run to reduce minimum requirements
        from matplotlib import pyplot as plt

        res = table_res['COMBINED_SEQ']
        styles_to_plot = ['r', 'b', 'g', 'b--', 'b:', 'g--', 'g:', 'm']
        plot_fields = [x for x in self.summary_fields if x in self.float_array_fields]
        for name, style in zip(plot_fields, styles_to_plot):
            plt.plot(self.array_labels, res[name], style)
        plt.xlabel('recall')
        plt.ylabel('score')
        plt.title(tracker + ' - ' + cls)
        plt.axis([0, 1, 0, 1])
        legend = []
        for name in plot_fields:
            legend += [name + ' (' + str(np.round(np.mean(res[name]), 2)) + ')']
        plt.legend(legend, loc='lower left')
        out_file = os.path.join(output_folder, cls + '_plot.pdf')
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        plt.savefig(out_file)
        plt.savefig(out_file.replace('.pdf', '.png'))
        plt.clf()


class DetLoc(_BaseMetric):
    """Implements detection metrics.

    Metrics are parameterized by IOU threshold.
    """

    def __init__(self):
        super().__init__()
        self.plottable = True
        self.array_labels = np.arange(5, 95 + 1, 5) / 100.
        self.integer_fields = ['DetLoc_Sequences']
        self.float_fields = ['DetLoc_AP_50', 'DetLoc_AP_75', 'DetLoc_AP_50_95']
        self.float_array_fields = ['DetLoc_AP', 'DetLoc_AP_sum']
        self.fields = self.integer_fields + self.float_fields + self.float_array_fields
        self.summary_fields = ['DetLoc_AP', 'DetLoc_AP_50', 'DetLoc_AP_75', 'DetLoc_AP_50_95']

        self.summed_fields = self.integer_fields + ['DetLoc_AP_sum']

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
        """Combines metrics across all sequences"""
        res = {}
        for field in self.summed_fields:
            res[field] = self._combine_sum(all_res, field)
        res = self._compute_final_fields(res)
        return res

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""
        return self.combine_sequences(all_res)

    def combine_classes_class_averaged(self, all_res):
        raise NotImplementedError

    def plot_single_tracker_results(self, table_res, tracker, cls, output_folder):
        """Create plot of results"""

        # Only loaded when run to reduce minimum requirements
        from matplotlib import pyplot as plt

        res = table_res['COMBINED_SEQ']
        styles_to_plot = ['r', 'b', 'g', 'b--', 'b:', 'g--', 'g:', 'm']
        plot_fields = [x for x in self.summary_fields if x in self.float_array_fields]
        for name, style in zip(plot_fields, styles_to_plot):
            plt.plot(self.array_labels, res[name], style)
        plt.xlabel('similarity')
        plt.ylabel('score')
        plt.title(tracker + ' - ' + cls)
        plt.axis([0, 1, 0, 1])
        legend = []
        for name in plot_fields:
            legend += [name + ' (' + str(np.round(np.mean(res[name]), 2)) + ')']
        plt.legend(legend, loc='lower left')
        out_file = os.path.join(output_folder, cls + '_plot.pdf')
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        plt.savefig(out_file)
        plt.savefig(out_file.replace('.pdf', '.png'))
        plt.clf()


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


# TODO: Remove after testing.
def _find_prec_at_recall(num_gt, scores, correct, thresholds):
    """Computes precision at a given minimum recall threshold.

    Args:
        num_gt: Number of ground-truth elements.
        scores: Score of each prediction.
        correct: Whether or not each prediction is correct.
        thresholds: Recall thresholds at which to evaluate.

    Follows implementation from Piotr Dollar toolbox (used by Matlab devkit).
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
    return np.asarray([prec[np.argmax(recall >= threshold)] for threshold in thresholds])


def _find_max_prec_at_recall(num_gt, scores, correct, thresholds):
    """Computes precision at a given minimum recall threshold.

    Args:
        num_gt: Number of ground-truth elements.
        scores: Score of each prediction.
        correct: Whether or not each prediction is correct.
        thresholds: Recall thresholds at which to evaluate.

    Follows implementation from Piotr Dollar toolbox (used by Matlab devkit).
    """
    recall, prec = _prec_recall_curve(num_gt, scores, correct)
    assert np.all(thresholds >= 0), thresholds
    assert np.all(thresholds <= 1), thresholds
    # Find first element with minimum recall.
    # Use argmax() to take first element that satisfies criterion.
    return np.asarray([prec[np.argmax(recall >= threshold)] for threshold in thresholds])


def _compute_average_precision(num_gt, scores, correct):
    recall, prec = _prec_recall_curve(num_gt, scores, correct)
    # Take integral.
    assert np.all(np.diff(recall) >= 0)
    return np.dot(prec[1:], recall[1:] - recall[:-1])


def _prec_recall_curve(num_gt, scores, correct):
    # Sort descending by score.
    order = np.argsort(-scores, kind='stable')
    correct = correct[order]
    tp = np.empty(len(scores) + 1)
    tp[0] = 0
    tp[1:] = np.cumsum(correct)
    num_pred = np.arange(len(scores) + 1)
    # Add an extra element for 100% recall, zero precision.
    recall = np.empty(len(scores) + 2)
    recall[:-1] = np.true_divide(tp, num_gt)
    recall[-1] = 1.
    prec = np.empty(len(scores) + 2)
    prec[:-1] = np.true_divide(tp, num_pred)
    prec[-1] = 0.
    # Avoid nan in prec[0]. This will be replaced with max prec.
    prec[0] = 0.
    # Take max precision available at equal or greater recall.
    # https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    return recall, prec
