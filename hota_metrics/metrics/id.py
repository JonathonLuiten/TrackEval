
import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_metric import _BaseMetric
from .. import _timing


class ID(_BaseMetric):
    """Class which implements the ID metrics"""
    def __init__(self):
        super().__init__()
        self.integer_headers = ['IDTP', 'IDFN', 'IDFP']
        self.float_headers = ['IDF1', 'IDR', 'IDP']
        self.headers = self.float_headers + self.integer_headers
        self.summary_headers = self.headers
        self.register_headers_globally()

        self.threshold = 0.5

    @_timing.time
    def eval_sequence(self, data):
        """Calculates ID metrics for one sequence"""
        # Initialise results
        res = {}
        for header in self.headers:
            res[header] = 0

        # Return result quickly if tracker or gt sequence is empty
        if data['num_tracker_dets'] == 0:
            res['IDFN'] = data['num_gt_dets']
            return res
        if data['num_gt_dets'] == 0:
            res['IDFP'] = data['num_tracker_dets']
            return res

        # Variables counting global association
        potential_matches_count = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
        gt_id_count = np.zeros(data['num_gt_ids'])
        tracker_id_count = np.zeros(data['num_tracker_ids'])

        # First loop through each timestep and accumulate global track information.
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            # Count the potential matches between ids in each timestep
            matches_mask = np.greater_equal(data['similarity_scores'][t], self.threshold)
            match_idx_gt, match_idx_tracker = np.nonzero(matches_mask)
            potential_matches_count[gt_ids_t[match_idx_gt], tracker_ids_t[match_idx_tracker]] += 1

            # Calculate the total number of dets for each gt_id and tracker_id.
            gt_id_count[gt_ids_t] += 1
            tracker_id_count[tracker_ids_t] += 1

        # Calculate optimal assignment cost matrix for ID metrics
        num_gt_ids = data['num_gt_ids']
        num_tracker_ids = data['num_tracker_ids']
        fp_mat = np.zeros((num_gt_ids + num_tracker_ids, num_gt_ids + num_tracker_ids))
        fn_mat = np.zeros((num_gt_ids + num_tracker_ids, num_gt_ids + num_tracker_ids))
        fp_mat[num_gt_ids:, :num_tracker_ids] = 1e10
        fn_mat[:num_gt_ids, num_tracker_ids:] = 1e10
        for gt_id in range(num_gt_ids):
            fn_mat[gt_id, :num_tracker_ids] = gt_id_count[gt_id]
            fn_mat[gt_id, num_tracker_ids + gt_id] = gt_id_count[gt_id]
        for tracker_id in range(num_tracker_ids):
            fp_mat[:num_gt_ids, tracker_id] = tracker_id_count[tracker_id]
            fp_mat[tracker_id + num_gt_ids, tracker_id] = tracker_id_count[tracker_id]
        fn_mat[:num_gt_ids, :num_tracker_ids] -= potential_matches_count
        fp_mat[:num_gt_ids, :num_tracker_ids] -= potential_matches_count

        # Hungarian algorithm
        match_rows, match_cols = linear_sum_assignment(fn_mat + fp_mat)

        # Accumulate basic statistics
        res['IDFN'] = fn_mat[match_rows, match_cols].sum().astype(np.int)
        res['IDFP'] = fp_mat[match_rows, match_cols].sum().astype(np.int)
        res['IDTP'] = (gt_id_count.sum() - res['IDFN']).astype(np.int)

        # Calculate final ID scores
        res['IDP'] = res['IDTP'] / np.maximum(1.0, res['IDTP'] + res['IDFP'])
        res['IDR'] = res['IDTP'] / np.maximum(1.0, res['IDTP'] + res['IDFN'])
        res['IDF1'] = res['IDTP'] / np.maximum(1.0, res['IDTP'] + 0.5 * res['IDFN'] + 0.5 * res['IDFP'])
        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        for header in self.integer_headers:
            res[header] = self._combine_sum(all_res, header)
        res['IDR'] = res['IDTP'] / np.maximum(1.0, res['IDTP'] + res['IDFN'])
        res['IDP'] = res['IDTP'] / np.maximum(1.0, res['IDTP'] + res['IDFP'])
        res['IDF1'] = res['IDTP'] / np.maximum(1.0, res['IDTP'] + 0.5*res['IDFP'] + 0.5*res['IDFN'])
        return res
