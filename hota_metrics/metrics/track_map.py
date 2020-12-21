
import numpy as np
from ._base_metric import _BaseMetric
from .. import _timing
from functools import partial


class TrackMAP(_BaseMetric):
    """Class which implements the CLEAR metrics"""
    def __init__(self):
        super().__init__()
        # main_integer_fields = ['CLR_TP', 'CLR_FN', 'CLR_FP', 'IDSW', 'MT', 'PT', 'ML', 'Frag']
        # extra_integer_fields = ['CLR_Frames']
        # self.integer_fields = main_integer_fields + extra_integer_fields
        # main_float_fields = ['MOTA', 'MOTP', 'MODA', 'CLR_Re', 'CLR_Pr', 'MTR', 'PTR', 'MLR']
        # extra_float_fields = ['CLR_F1', 'FP_per_frame', 'Frag_per_Re', 'IDSW_per_Re', 'MOTAL']
        # self.float_fields = main_float_fields + extra_float_fields
        # self.fields = self.float_fields + self.integer_fields
        # self.summary_fields = main_float_fields + main_integer_fields

        self.area_rngs = [[0 ** 2, 32 ** 2],
                          [32 ** 2, 96 ** 2],
                          [96 ** 2, 1e5 ** 2]]
        self.area_rng_lbls = ["area_s", "area_m", "area_l"]
        self.time_rngs = [[0, 3], [3, 10], [10, 1e5]]
        self.time_rng_lbls = ["time_s", "time_m", "time_l"]
        self.num_ig_masks = 1 + len(self.area_rng_lbls) + len(self.time_rng_lbls)
        self.lbls = ['all'] + self.area_rng_lbls + self.time_rng_lbls

        self.array_labels = np.arange(0.0, 1.01, 0.05)
        self.rec_thrs = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01) + 1), endpoint=True)

        self.float_array_fields = ['AP_' + lbl for lbl in self.lbls] + ['AR_' + lbl for lbl in self.lbls]
        self.fields = self.float_array_fields
        self.summary_fields = self.float_array_fields

    @_timing.time
    def eval_sequence(self, data):
        """Calculates CLEAR metrics for one sequence"""
        # Initialise results
        res = {}
        for field in self.fields:
            res[field] = 0

        gt_ids, dt_ids = data['gt_track_ids'], data['dt_track_ids']

        if len(gt_ids) == 0 and len(dt_ids) == 0:
            for idx in range(self.num_ig_masks):
                res[idx] = None
            return res

        gt_ig_masks = self._compute_track_ig_masks(track_lengths=data['gt_track_lengths'],
                                                   track_areas=data['gt_track_areas'])
        dt_ig_masks = self._compute_track_ig_masks(track_lengths=data['dt_track_lengths'],
                                                   track_areas=data['dt_track_areas'],
                                                   is_not_exhaustively_labeled=data['not_exhaustively_labeled'],
                                                   is_gt=False)

        assert len(gt_ig_masks) == self.num_ig_masks, \
            'gt data does not have the correct number of ignore masks to consider'
        assert len(dt_ig_masks) == self.num_ig_masks, \
            'tracker data does not have the correct number of ignore masks to consider'

        ious = self._compute_track_ious(data['dt_tracks'], data['gt_tracks'], iou_function='bbox')

        for mask_idx in range(self.num_ig_masks):
            gt_ig_mask = gt_ig_masks[mask_idx]

            # Sort gt ignore last
            gt_idx = np.argsort([g for g in gt_ig_mask], kind="mergesort")
            gt_ids = [gt_ids[i] for i in gt_idx]

            ious_sorted = ious[:, gt_idx] if len(ious) > 0 else ious

            num_thrs = len(self.array_labels)
            num_gt = len(gt_ids)
            num_dt = len(dt_ids)

            # Array to store the "id" of the matched dt/gt
            gt_m = np.zeros((num_thrs, num_gt)) - 1
            dt_m = np.zeros((num_thrs, num_dt)) - 1

            gt_ig = np.array([gt_ig_mask[idx] for idx in gt_idx])
            dt_ig = np.zeros((num_thrs, num_dt))

            for iou_thr_idx, iou_thr in enumerate(self.array_labels):
                if len(ious_sorted) == 0:
                    break

                for dt_idx, _dt in enumerate(dt_ids):
                    iou = min([iou_thr, 1 - 1e-10])
                    # information about best match so far (m=-1 -> unmatched)
                    # store the gt_idx which matched for _dt
                    m = -1
                    for gt_idx, _ in enumerate(gt_ids):
                        # if this gt already matched continue
                        if gt_m[iou_thr_idx, gt_idx] > 0:
                            continue
                        # if _dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gt_ig[m] == 0 and gt_ig[gt_idx] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious_sorted[dt_idx, gt_idx] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou = ious_sorted[dt_idx, gt_idx]
                        m = gt_idx

                    # No match found for _dt, go to next _dt
                    if m == -1:
                        continue

                    # if gt to ignore for some reason update dt_ig.
                    # Should not be used in evaluation.
                    dt_ig[iou_thr_idx, dt_idx] = gt_ig[m]
                    # _dt match found, update gt_m, and dt_m with "id"
                    dt_m[iou_thr_idx, dt_idx] = gt_ids[m]
                    gt_m[iou_thr_idx, m] = _dt

            dt_ig_mask = dt_ig_masks[mask_idx]

            dt_ig_mask = np.array(dt_ig_mask).reshape((1, num_dt))  # 1 X num_dt
            dt_ig_mask = np.repeat(dt_ig_mask, num_thrs, 0)  # num_thrs X num_dt

            # Based on dt_ig_mask ignore any unmatched detection by updating dt_ig
            dt_ig = np.logical_or(dt_ig, np.logical_and(dt_m == -1, dt_ig_mask))
            # store results for given video and category
            res[mask_idx] = {
                "dt_ids": dt_ids,
                "gt_ids": gt_ids,
                "dt_matches": dt_m,
                "gt_matches": gt_m,
                "dt_scores": data['dt_track_scores'],
                "gt_ignore": gt_ig,
                "dt_ignore": dt_ig,
            }

        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences
        Adapted from https://github.com/TAO-Dataset/tao/blob/master/tao/toolkit/tao/eval.py
        """
        num_thrs = len(self.array_labels)
        num_recalls = len(self.rec_thrs)

        # -1 for absent categories
        precision = -np.ones(
            (num_thrs, num_recalls, self.num_ig_masks)
        )
        recall = -np.ones((num_thrs, self.num_ig_masks))

        for ig_idx in range(self.num_ig_masks):
            ig_idx_results = [res[ig_idx] for res in all_res.values() if res[ig_idx] is not None]

            # Remove elements which are None
            if len(ig_idx_results) == 0:
                continue

            # Append all scores: shape (N,)
            dt_scores = np.concatenate([res["dt_scores"] for res in ig_idx_results], axis=0)

            dt_idx = np.argsort(-dt_scores, kind="mergesort")

            dt_m = np.concatenate([e["dt_matches"] for e in ig_idx_results],
                                  axis=1)[:, dt_idx]
            dt_ig = np.concatenate([e["dt_ignore"] for e in ig_idx_results],
                                   axis=1)[:, dt_idx]

            gt_ig = np.concatenate([res["gt_ignore"] for res in ig_idx_results])
            # num gt anns to consider
            num_gt = np.count_nonzero(gt_ig == 0)

            if num_gt == 0:
                continue

            tps = np.logical_and(dt_m != -1, np.logical_not(dt_ig))
            fps = np.logical_and(dt_m == -1, np.logical_not(dt_ig))

            tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
            fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)

            for iou_thr_idx, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                tp = np.array(tp)
                fp = np.array(fp)
                num_tp = len(tp)
                rc = tp / num_gt
                if num_tp:
                    recall[iou_thr_idx, ig_idx] = rc[-1]
                else:
                    recall[iou_thr_idx, ig_idx] = 0

                # np.spacing(1) ~= eps
                pr = tp / (fp + tp + np.spacing(1))
                pr = pr.tolist()

                # Ensure precision values are monotonically decreasing
                for i in range(num_tp - 1, 0, -1):
                    if pr[i] > pr[i - 1]:
                        pr[i - 1] = pr[i]

                # find indices at the predefined recall values
                rec_thrs_insert_idx = np.searchsorted(rc, self.rec_thrs, side="left")

                pr_at_recall = [0.0] * num_recalls

                try:
                    for _idx, pr_idx in enumerate(rec_thrs_insert_idx):
                        pr_at_recall[_idx] = pr[pr_idx]
                except IndexError:
                    pass

                precision[iou_thr_idx, :, ig_idx] = (np.array(pr_at_recall))

        res = {'precision': precision, 'recall': recall}

        # compute the precision and recall averages for the respective alpha thresholds and ignore masks
        for lbl in self.lbls:
            res['AP_' + lbl] = np.zeros((len(self.array_labels)), dtype=np.float)
            res['AR_' + lbl] = np.zeros((len(self.array_labels)), dtype=np.float)

        for a_id, alpha in enumerate(self.array_labels):
            for lbl_idx, lbl in enumerate(self.lbls):
                p = precision[a_id, :, lbl_idx]
                if len(p[p > -1]) == 0:
                    mean_p = -1
                else:
                    mean_p = np.mean(p[p > -1])
                res['AP_' + lbl][a_id] = mean_p
                res['AR_' + lbl][a_id] = recall[a_id, lbl_idx]

        return res

    def combine_classes(self, all_res):
        all_prec = np.array([res['precision'] for res in all_res.values()])
        all_rec = np.array([res['recall'] for res in all_res.values()])

        res = {}

        # compute the precision and recall averages for the respective alpha thresholds and ignore masks
        for lbl in self.lbls:
            res['AP_' + lbl] = np.zeros((len(self.array_labels)), dtype=np.float)
            res['AR_' + lbl] = np.zeros((len(self.array_labels)), dtype=np.float)

        for a_id, alpha in enumerate(self.array_labels):
            for lbl_idx, lbl in enumerate(self.lbls):
                p = all_prec[:, a_id, :, lbl_idx]
                if len(p[p > -1]) == 0:
                    mean_p = -1
                else:
                    mean_p = np.mean(p[p > -1])
                res['AP_' + lbl][a_id] = mean_p

                r = all_rec[:, a_id, lbl_idx]
                if len(r[r > -1]) == 0:
                    mean_r = -1
                else:
                    mean_r = np.mean(r[r > -1])
                res['AR_' + lbl][a_id] = mean_r

        return res

    def _compute_track_ig_masks(self, track_lengths=None, track_areas=None,
                                is_not_exhaustively_labeled=False, is_gt=True):
        if not is_gt and is_not_exhaustively_labeled:
            track_ig_masks = [[1 for _ in track_lengths] for i in range(self.num_ig_masks)]
        else:
            # consider all tracks
            track_ig_masks = [[0 for _ in track_lengths]]

            # consider tracks with certain area
            if self.area_rngs:
                for rng in self.area_rngs:
                    track_ig_masks.append([0 if rng[0] <= area <= rng[1] else 1 for area in track_areas])

            # consider tracks with certain duration
            if self.time_rngs:
                for rng in self.time_rngs:
                    track_ig_masks.append([0 if rng[0] <= length <= rng[1] else 1 for length in track_lengths])

        return track_ig_masks

    @staticmethod
    def _compute_bb_track_iou(dt_track, gt_track, boxformat='xywh'):
        intersect = 0
        union = 0
        image_ids = set(gt_track.keys()) | set(dt_track.keys())
        for image in image_ids:
            g = gt_track.get(image, None)
            d = dt_track.get(image, None)
            if boxformat == 'xywh':
                if d and g:
                    dx, dy, dw, dh = d
                    gx, gy, gw, gh = g
                    w = max(min(dx + dw, gx + gw) - max(dx, gx), 0)
                    h = max(min(dy + dh, gy + gh) - max(dy, gy), 0)
                    i = w * h
                    u = dw * dh + gw * gh - i
                    intersect += i
                    union += u
                elif not d and g:
                    union += g[2] * g[3]
                elif d and not g:
                    union += d[2] * d[3]
            else:
                raise (Exception('BoxFormat not implemented'))
        assert intersect <= union
        return intersect / union if union > 0 else 0

    @staticmethod
    def _compute_mask_track_iou(dt_track, gt_track):
        #TODO implement mask track iou computation
        pass

    @staticmethod
    def _compute_track_ious(dt, gt, iou_function='bbox', boxformat='xywh'):
        """
        Adapted from https://github.com/TAO-Dataset/tao/blob/master/tao/toolkit/tao/eval.py
        Calculate track IoUs for a set of ground truth tracks gt and a set of detected tracks dt
        """

        if len(gt) == 0 and len(dt) == 0:
            return []

        if iou_function == 'bbox':
            track_iou_function = partial(TrackMAP._compute_bb_track_iou, boxformat=boxformat)
        elif iou_function == 'mask':
            track_iou_function = partial(TrackMAP._compute_mask_track_iou)
        else:
            raise Exception('IoU function not implemented')

        ious = np.zeros([len(dt), len(gt)])
        for i, j in np.ndindex(ious.shape):
            ious[i, j] = track_iou_function(dt[i], gt[j])
        return ious

    @staticmethod
    def _row_print(*argv):
        """Prints results in an evenly spaced rows, with more space in first row"""
        if len(argv) == 1:
            argv = argv[0]
        to_print = '%-40s' % argv[0]
        for v in argv[1:]:
            to_print += '%-12s' % str(v)
        print(to_print)
