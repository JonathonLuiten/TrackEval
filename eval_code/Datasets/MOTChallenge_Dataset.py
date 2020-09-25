import numpy as np
import os
import configparser
from eval_code.Eval.eval_utils import squarify, linear_assignment_problem
import csv

def load_raw_MOTCha_seq(seq,fol,meta_data=None, isGT=True):
    if meta_data == None:
        meta_data = dict()
        info_file = os.path.join(fol, seq, "seqinfo.ini")
        config = configparser.ConfigParser()
        config.read(info_file)
        meta_data['num_timesteps'] = int(config["Sequence"]["seqLength"])
        meta_data['preproc_type'] = config["Sequence"]["name"].split('-')[0]
        meta_data['seq'] = seq
    if isGT:
        file = os.path.join(fol, seq, 'gt', 'gt.txt')
    else:
        file = os.path.join(fol, seq + '.txt')
    if not os.path.exists(file):
        raise Exception('File does not exist: %s' % file)

    read_data = dict()
    with open(file) as fp:
        dialect = csv.Sniffer().sniff(fp.read(10240))
        fp.seek(0)
        reader = csv.reader(fp, dialect)
        for row in reader:
            timestep = row[0]
            if timestep in read_data.keys():
                read_data[timestep].append(row)
            else:
                read_data[timestep] = [row]

    raw_data = [None] * meta_data['num_timesteps']
    len_extras = None
    for key in read_data.keys():
        t = int(key) - 1
        if t < 0 or t >= meta_data['num_timesteps']: continue
        time_data = np.asarray(read_data[key], dtype=np.float)
        boxes = time_data[:, 2:6]
        IDs = time_data[:, 1].astype(int) - 1
        extras = time_data[:, 6:]
        raw_data[t] = [boxes,IDs,extras]
        if len_extras is None:
            len_extras = len(extras[0])
    for i, d in enumerate(raw_data):
        if d is None:
            raw_data[i] = [np.empty((0, 4)), np.empty((0)), np.empty((0,len_extras))]
    return raw_data, meta_data

def preproc(raw_gt_data, raw_tracker_data, raw_similarity_scores , meta_data):
    tracker_data = []
    gt_data = []
    similarity_scores = []
    meta_data['num_trackIDs'] = -1
    meta_data['num_gtIDs'] = -1
    unique_gt_ids = []
    unique_pr_ids = []
    assert len(raw_gt_data) == len(raw_tracker_data), 'gt and prediction have different number of timesteps'
    for t, (time_data, time_gt) in enumerate(zip(raw_tracker_data, raw_gt_data)):

        boxes = time_data[0]
        IDs = time_data[1]
        gt_boxes = time_gt[0]
        gt_IDs = time_gt[1]
        gt_classes = time_gt[2][:, 1].astype(int)
        gt_zero_marked = time_gt[2][:, 0]

        if 'MOT20' in meta_data['preproc_type']:
            classes_to_remove = [2, 6, 7, 8, 12]
        else:
            classes_to_remove = [2, 7, 8, 12]

        # Preprocesses preds (remove predictions matching with occluded gt boxes and gt boxes of distractor classes)
        if meta_data['preproc_type'] is not 'MOT15' and boxes.shape[0] > 0 and boxes.shape[0] > 0:
            ious_all = squarify(raw_similarity_scores[t],0)
            ious_all[ious_all < 0.5] = -1
            match_rows, match_cols = linear_assignment_problem(ious_all)
            to_remove = np.isin(gt_classes[match_rows], classes_to_remove)
            to_remove = match_cols[to_remove]
            boxes = np.delete(boxes, to_remove, axis=0)
            IDs = np.delete(IDs, to_remove, axis=0)
            raw_similarity_scores[t] = np.delete(raw_similarity_scores[t], to_remove, axis=1)

        # Preprocesses GT (Only keep gt boxes that are: a. not zero marked and b. pedestrian class)
        clean_data_mask = (np.not_equal(gt_zero_marked, 0)) & (np.equal(gt_classes, 1))
        gt_boxes = gt_boxes[clean_data_mask, :]
        gt_IDs = gt_IDs[clean_data_mask]
        similarity_scores.append(raw_similarity_scores[t][clean_data_mask, :])

        tracker_data.append([boxes, IDs.astype(int)])
        gt_data.append([gt_boxes, gt_IDs])

        unique_gt_ids += list(np.unique(gt_IDs))
        unique_pr_ids += list(np.unique(IDs))

    # Re-label IDs such that there are no empty IDs
    unique_gt_ids = np.unique(unique_gt_ids)
    gt_id_map = np.nan*np.ones((np.max(unique_gt_ids)+1))
    gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))

    unique_pr_ids = np.unique(unique_pr_ids)
    pr_id_map = np.nan * np.ones((np.max(unique_pr_ids)+1),np.int)
    pr_id_map[unique_pr_ids] = np.arange(len(unique_pr_ids))

    meta_data['num_gtIDs'] = len(unique_gt_ids)
    meta_data['num_trackIDs'] = len(unique_pr_ids)
    for t, (time_data, time_gt) in enumerate(zip(tracker_data, gt_data)):
        gt_data[t][1] = gt_id_map[time_gt[1]]
        tracker_data[t][1] = pr_id_map[time_data[1]]

    return gt_data, tracker_data, similarity_scores, meta_data