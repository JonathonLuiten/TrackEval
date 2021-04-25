
import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_metric import _BaseMetric
from .. import _timing
from collections import defaultdict

class Identity(_BaseMetric):
    """Class which implements the ID metrics"""
    def __init__(self):
        super().__init__()
        self.integer_fields = ['IDTP', 'IDFN', 'IDFP']
        self.float_fields = ['IDF1', 'IDR', 'IDP']
        self.fields = self.float_fields + self.integer_fields
        self.summary_fields = self.fields

        self.threshold = 0.4

    @_timing.time
    def eval_sequence(self, data):
        """Calculates ID metrics for one sequence"""
        # Initialise results
        res = {}
        for field in self.fields:
            res[field] = 0

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

        # ID Eucl metric
        data['centroid'] = []
        for t, gt_det in enumerate(data['gt_dets']):
            data['centroid'].append(self._compute_centroid(gt_det))

        oid_hid_cent = defaultdict(list)
        oid_cent = defaultdict(list)
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            matches_mask = np.greater_equal(data['similarity_scores'][t], self.threshold)

            # I hope the orders of ids and boxes are maintained in `data`
            for ind, gid in enumerate(gt_ids_t):
                oid_cent[gid].append(data['centroid'][t][ind])

            match_idx_gt, match_idx_tracker = np.nonzero(matches_mask)
            for m_gid, m_tid in zip(match_idx_gt, match_idx_tracker):
                oid_hid_cent[gt_ids_t[m_gid], tracker_ids_t[m_tid]].append(data['centroid'][t][m_gid])

        oid_hid_dist = {k : np.sum(np.linalg.norm(np.diff(np.array(v), axis=0), axis=1)) for k, v in oid_hid_cent.items()}
        oid_dist = {int(k) : np.sum(np.linalg.norm(np.diff(np.array(v), axis=0), axis=1)) for k, v in oid_cent.items()}

        unique_oid = np.unique([i[0] for i in oid_hid_dist.keys()]).tolist()
        unique_hid = np.unique([i[1] for i in oid_hid_dist.keys()]).tolist()
        o_len = len(unique_oid)
        h_len = len(unique_hid)
        dist_matrix = np.zeros((o_len, h_len))
        for ((oid, hid), dist) in oid_hid_dist.items():
            oid_ind = unique_oid.index(oid)
            hid_ind = unique_hid.index(hid)
            dist_matrix[oid_ind, hid_ind] = dist


        
        # opt_hyp_dist contains GT ID : max dist covered by track
        opt_hyp_dist = dict.fromkeys(oid_dist.keys(), 0.)
        cost_matrix = np.max(dist_matrix) - dist_matrix
        rows, cols = linear_sum_assignment(cost_matrix)
        for (row, col) in zip(rows, cols):
            value = dist_matrix[row, col]
            opt_hyp_dist[int(unique_oid[row])] = value

        assert len(opt_hyp_dist.keys()) == len(oid_dist.keys())
        hyp_length = np.sum(list(opt_hyp_dist.values()))
        gt_length = np.sum(list(oid_dist.values()))
        res['IDEucl'] = hyp_length / gt_length
        ## End of IDEucl metric

        #for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
        # Hungarian algorithm
        match_rows, match_cols = linear_sum_assignment(fn_mat + fp_mat)

        # Accumulate basic statistics
        res['IDFN'] = fn_mat[match_rows, match_cols].sum().astype(np.int)
        res['IDFP'] = fp_mat[match_rows, match_cols].sum().astype(np.int)
        res['IDTP'] = (gt_id_count.sum() - res['IDFN']).astype(np.int)

        # Calculate final ID scores
        res = self._compute_final_fields(res)
        return res

    def combine_classes_class_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the class values"""
        res = {}
        for field in self.integer_fields:
            res[field] = self._combine_sum({k: v for k, v in all_res.items()
                                            if v['IDTP'] + v['IDFN'] + v['IDFP'] > 0 + np.finfo('float').eps}, field)
        for field in self.float_fields:
            res[field] = np.mean([v[field] for v in all_res.values()
                                  if v['IDTP'] + v['IDFN'] + v['IDFP'] > 0 + np.finfo('float').eps], axis=0)
        return res

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""
        res = {}
        for field in self.integer_fields:
            res[field] = self._combine_sum(all_res, field)
        res = self._compute_final_fields(res)
        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        for field in self.integer_fields:
            res[field] = self._combine_sum(all_res, field)
        res = self._compute_final_fields(res)
        return res


    @staticmethod
    def _compute_centroid(box):
        box = np.array(box)
        if len(box.shape) == 1:
            centroid = (box[0:2] + box[2:4])/2
        else:
            centroid = (box[:, 0:2] + box[:, 2:4])/2
        return centroid


    @staticmethod
    def _compute_final_fields(res):
        """Calculate sub-metric ('field') values which only depend on other sub-metric values.
        This function is used both for both per-sequence calculation, and in combining values across sequences.
        """
        res['IDR'] = res['IDTP'] / np.maximum(1.0, res['IDTP'] + res['IDFN'])
        res['IDP'] = res['IDTP'] / np.maximum(1.0, res['IDTP'] + res['IDFP'])
        res['IDF1'] = res['IDTP'] / np.maximum(1.0, res['IDTP'] + 0.5*res['IDFP'] + 0.5*res['IDFN'])
        return res
