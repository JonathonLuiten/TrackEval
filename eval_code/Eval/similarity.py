
import numpy as np
from copy import deepcopy as copy

def calc_similarities(raw_gt_data, raw_tracker_data):
  similarity_scores = []
  assert len(raw_gt_data) == len(raw_tracker_data), 'gt and prediction have different number of timesteps'
  for t, (time_data, time_gt) in enumerate(zip(raw_tracker_data, raw_gt_data)):

    # Check the requirement that PrID and gtIDs are unique per timestep
    gt_ids = time_gt[1].astype(int)
    curr_ids = time_data[1].astype(int)
    if len(curr_ids) > 0:
      _, counts = np.unique(curr_ids, return_counts=True)
      assert np.max(counts) == 1, 'Tracker predicts the same ID more than once in a single timestep'
    if len(gt_ids) > 0:
      _, counts = np.unique(gt_ids, return_counts=True)
      assert np.max(counts) == 1, 'Ground-truth has the same ID more than once in a single timestep'

    # Calculate similarity score between predicted and gt boxes (needed for all metrics)
    ious_all = calculate_box_ious(time_gt[0], time_data[0])
    similarity_scores.append(ious_all)
  return similarity_scores

def calculate_box_ious(bboxes1, bboxes2):
  # assume layout (x0, y0, w, h)
  bboxes1 = copy(bboxes1)
  bboxes2 = copy(bboxes2)

  bboxes1[:, 2] = bboxes1[:, 0] + bboxes1[:, 2]
  bboxes1[:, 3] = bboxes1[:, 1] + bboxes1[:, 3]
  bboxes2[:, 2] = bboxes2[:, 0] + bboxes2[:, 2]
  bboxes2[:, 3] = bboxes2[:, 1] + bboxes2[:, 3]

  # assume layout (x0, y0, x1, y1)
  min_ = np.minimum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
  max_ = np.maximum(bboxes1[:, np.newaxis, :], bboxes2[np.newaxis, :, :])
  I = np.maximum(min_[..., 2] - max_[..., 0], 0) * np.maximum(min_[..., 3] - max_[..., 1], 0)
  area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
  area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
  U = area1[:, np.newaxis] + area2[np.newaxis, :] - I

  if not (U > 0).all():
    raise ValueError("Area not > 0 for all boxes!")

  IOUs = I / U
  assert (IOUs >= 0).all()
  assert (IOUs <= 1).all()

  return IOUs