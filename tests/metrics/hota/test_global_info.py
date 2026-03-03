import numpy as np
import pytest

from trackeval.metrics.hota.global_info import get_global_info


def test_get_global_info():
    data = {
        'num_gt_ids': 4,
        'num_tracker_ids': 3,
        'gt_ids': [np.zeros(0, int), 1 * np.ones(1, int)],
        'tracker_ids': [np.zeros(0, int), 2 * np.ones(1, int), None],
        'similarity_scores': [np.zeros((0, 0)), np.full((1, 1), 0.567), None]
    }

    potential_matches_count, gt_id_count, tracker_id_count = get_global_info(data)
    ref_pm = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    ref_gt_id_count = [[0.0], [1.0], [0.0], [0.0]]
    ref_trk_id_count = [[0.0, 0.0, 1.0]]
    assert potential_matches_count == pytest.approx(np.array(ref_pm))
    assert gt_id_count == pytest.approx(np.array(ref_gt_id_count))
    assert tracker_id_count == pytest.approx(np.array(ref_trk_id_count))
    assert potential_matches_count.shape == (4, 3)
    assert gt_id_count.shape == (4, 1)
    assert tracker_id_count.shape == (1, 3)
