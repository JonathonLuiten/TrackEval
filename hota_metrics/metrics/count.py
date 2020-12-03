
import os
import numpy as np
from matplotlib import pyplot as plt
from ._base_metric import _BaseMetric
from .. import _timing


class Count(_BaseMetric):
    """Class which simply counts the number of tracker and gt detections and ids."""
    def __init__(self):
        super().__init__()
        self.integer_headers = ['Dets', 'GT_Dets', 'IDs', 'GT_IDs']
        self.headers = self.integer_headers
        self.summary_headers = self.headers
        self.register_headers_globally()

    @_timing.time
    def eval_sequence(self, data):
        """Returns counts for one sequence"""
        # Get results
        res = {'Dets': data['num_tracker_dets'],
               'GT_Dets': data['num_gt_dets'],
               'IDs': data['num_tracker_ids'],
               'GT_IDs': data['num_gt_ids']}
        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        for header in self.integer_headers:
            res[header] = self._combine_sum(all_res, header)
        return res