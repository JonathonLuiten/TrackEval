
from typing import Tuple

import numpy as np

from trackeval.types import DT, FMT


def get_global_info(data: DT) -> Tuple[FMT, FMT, FMT]:
    # Variables counting global association
    potential_matches_count = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
    gt_id_count = np.zeros((data['num_gt_ids'], 1))
    tracker_id_count = np.zeros((1, data['num_tracker_ids']))

    # First loop through each timestep and accumulate global track information.
    for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
        # Count the potential matches between ids in each timestep
        # These are normalized, weighted by the match similarity.
        similarity = data['similarity_scores'][t]
        sim_iou_den = similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
        sim_iou = np.zeros_like(similarity)
        sim_iou_mask = sim_iou_den > 0 + np.finfo('float').eps
        sim_iou[sim_iou_mask] = similarity[sim_iou_mask] / sim_iou_den[sim_iou_mask]
        potential_matches_count[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] += sim_iou

        # Calculate the total number of dets for each gt_id and tracker_id.
        gt_id_count[gt_ids_t] += 1
        tracker_id_count[0, tracker_ids_t] += 1
    return potential_matches_count, gt_id_count, tracker_id_count
