
import numpy as np
import math
import cv2
from scipy.optimize import linear_sum_assignment
from pycocotools import mask as mask_utils
from skimage.morphology import disk
from ._base_metric import _BaseMetric
from .. import _timing


class JAndF(_BaseMetric):
    """Class which simply counts the number of tracker and gt detections and ids."""
    def __init__(self):
        super().__init__()
        self.integer_fields = ['num_gt_tracks']
        self.float_fields = ['J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay', 'J&F']
        self.fields = self.float_fields + self.integer_fields
        self.summary_fields = self.float_fields

    @_timing.time
    def eval_sequence(self, data):
        """Returns counts for one sequence"""
        # Get results

        tracker_masks = np.stack([det for det in data['tracker_dets']], axis=0)
        gt_masks = np.stack([det for det in data['gt_dets']], axis=0).astype(np.uint8)
        if data['num_tracker_ids'] < data['num_gt_ids']:
            diff = data['num_gt_ids'] - data['num_tracker_ids']
            zero_padding = np.zeros((tracker_masks.shape[0], diff, *tracker_masks.shape[2:]))
            tracker_masks = np.concatenate((tracker_masks, zero_padding), axis=1).astype(np.uint8)

        j = JAndF._compute_j(gt_masks, tracker_masks)
        bound_th = 0.008

        # perform matching on J measure
        j_metrics = np.mean(j, axis=2)
        row_ind, col_ind = linear_sum_assignment(-j_metrics)
        j_m = j[row_ind, col_ind, :]
        f_m = np.zeros_like(j_m)
        for i, (tr_ind, gt_ind) in enumerate(zip(row_ind, col_ind)):
            f_m[i] = self._compute_f(gt_masks, tracker_masks, tr_ind, gt_ind, bound_th)

        # append zeros for false negatives
        if j_m.shape[0] < data['num_gt_ids']:
            diff = data['num_gt_ids'] - j_m.shape[0]
            j_m = np.concatenate((j_m, np.zeros((diff, j_m.shape[1]))), axis=0)
            f_m = np.concatenate((f_m, np.zeros((diff, f_m.shape[1]))), axis=0)

        # compute the metrics for each ground truth track
        res = {
            'J-Mean': [np.nanmean(j_m[i, :]) for i in range(j_m.shape[0])],
            'J-Recall': [np.nanmean(j_m[i, :] > 0.5) for i in range(j_m.shape[0])],
            'F-Mean': [np.nanmean(f_m[i, :]) for i in range(f_m.shape[0])],
            'F-Recall': [np.nanmean(f_m[i, :] > 0.5) for i in range(f_m.shape[0])],
            'J-Decay': [],
            'F-Decay': []
        }
        n_bins = 4
        ids = np.round(np.linspace(1, data['num_timesteps'], n_bins + 1) + 1e-10) - 1
        ids = ids.astype(np.uint8)

        for k in range(j_m.shape[0]):
            d_bins_j = [j_m[k][ids[i]:ids[i + 1] + 1] for i in range(0, n_bins)]
            res['J-Decay'].append(np.nanmean(d_bins_j[0]) - np.nanmean(d_bins_j[3]))
        for k in range(f_m.shape[0]):
            d_bins_f = [f_m[k][ids[i]:ids[i + 1] + 1] for i in range(0, n_bins)]
            res['F-Decay'].append(np.nanmean(d_bins_f[0]) - np.nanmean(d_bins_f[3]))

        res['num_gt_tracks'] = len(res['J-Mean'])
        for field in ['J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']:
            res[field] = np.mean(res[field])
        res['J&F'] = (res['J-Mean'] + res['F-Mean']) / 2
        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        res['num_gt_tracks'] = self._combine_sum(all_res, 'num_gt_tracks')
        for field in self.summary_fields:
            res[field] = self._combine_weighted_av(all_res, field, res, weight_field='num_gt_tracks')
        return res

    @staticmethod
    def _seg2bmap(seg, width=None, height=None):
        """
        From a segmentation, compute a binary boundary map with 1 pixel wide
        boundaries.  The boundary pixels are offset by 1/2 pixel towards the
        origin from the actual segment boundary.
        Arguments:
            seg     : Segments labeled from 1..k.
            width	  :	Width of desired bmap  <= seg.shape[1]
            height  :	Height of desired bmap <= seg.shape[0]
        Returns:
            bmap (ndarray):	Binary boundary map.
         David Martin <dmartin@eecs.berkeley.edu>
         January 2003
        """

        seg = seg.astype(np.bool)
        seg[seg > 0] = 1

        assert np.atleast_3d(seg).shape[2] == 1

        width = seg.shape[1] if width is None else width
        height = seg.shape[0] if height is None else height

        h, w = seg.shape[:2]

        ar1 = float(width) / float(height)
        ar2 = float(w) / float(h)

        assert not (
                width > w | height > h | abs(ar1 - ar2) > 0.01
        ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

        e = np.zeros_like(seg)
        s = np.zeros_like(seg)
        se = np.zeros_like(seg)

        e[:, :-1] = seg[:, 1:]
        s[:-1, :] = seg[1:, :]
        se[:-1, :-1] = seg[1:, 1:]

        b = seg ^ e | seg ^ s | seg ^ se
        b[-1, :] = seg[-1, :] ^ e[-1, :]
        b[:, -1] = seg[:, -1] ^ s[:, -1]
        b[-1, -1] = 0

        if w == width and h == height:
            bmap = b
        else:
            bmap = np.zeros((height, width))
            for x in range(w):
                for y in range(h):
                    if b[y, x]:
                        j = 1 + math.floor((y - 1) + height / h)
                        i = 1 + math.floor((x - 1) + width / h)
                        bmap[j, i] = 1

        return bmap

    @staticmethod
    def _compute_f(gt_data, tracker_data, tracker_data_id, gt_id, bound_th):
        f = np.zeros(gt_data.shape[0])

        for t, (gt_masks, tracker_masks) in enumerate(zip(gt_data, tracker_data)):
            bound_pix = bound_th if bound_th >= 1 else \
                np.ceil(bound_th * np.linalg.norm(tracker_masks[tracker_data_id].shape))

            # Get the pixel boundaries of both masks
            fg_boundary = JAndF._seg2bmap(tracker_masks[tracker_data_id])
            gt_boundary = JAndF._seg2bmap(gt_masks[gt_id])

            # fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
            fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
            # gt_dil = binary_dilation(gt_boundary, disk(bound_pix))
            gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

            # Get the intersection
            gt_match = gt_boundary * fg_dil
            fg_match = fg_boundary * gt_dil

            # Area of the intersection
            n_fg = np.sum(fg_boundary)
            n_gt = np.sum(gt_boundary)

            # % Compute precision and recall
            if n_fg == 0 and n_gt > 0:
                precision = 1
                recall = 0
            elif n_fg > 0 and n_gt == 0:
                precision = 0
                recall = 1
            elif n_fg == 0 and n_gt == 0:
                precision = 1
                recall = 1
            else:
                precision = np.sum(fg_match) / float(n_fg)
                recall = np.sum(gt_match) / float(n_gt)

            # Compute F measure
            if precision + recall == 0:
                f_val = 0
            else:
                f_val = 2 * precision * recall / (precision + recall)

            f[t] = f_val

        return f

    @staticmethod
    def _compute_j(gt_data, tracker_data):
        j = np.zeros((tracker_data.shape[1], gt_data.shape[1], gt_data.shape[0]))

        # J computation
        for t, (time_gt, time_data) in enumerate(zip(gt_data, tracker_data)):
            # run length encoded masks with pycocotools
            gt_encoded = mask_utils.encode(np.array(np.transpose(time_gt, (1, 2, 0)), order='F'))
            tr_encoded = mask_utils.encode(np.array(np.transpose(time_data, (1, 2, 0)), order='F'))
            area_gt = mask_utils.area(gt_encoded)
            area_tr = mask_utils.area(tr_encoded)

            area_tr = np.repeat(area_tr[:, np.newaxis], len(area_gt), axis=1)
            area_gt = np.repeat(area_gt[np.newaxis, :], len(area_tr), axis=0)

            # mask iou computation with pycocotools
            ious = mask_utils.iou(tr_encoded, gt_encoded, np.zeros([len(tr_encoded)]))
            # set iou to 1 if both masks are close to 0 (no ground truth and no predicted mask in timestep)
            ious[np.isclose(area_tr, 0) & np.isclose(area_gt, 0)] = 1
            assert (ious >= 0).all()
            assert (ious <= 1).all()

            j[..., t] = ious

        return j
