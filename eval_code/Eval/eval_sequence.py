
import numpy as np
from eval_code.Eval.eval_utils import squarify, linear_assignment_problem
from lapjv import lapjv as hungarian

def eval_sequence(gt_data, tracker_data, similarity_scores, alpha_range, meta_data, metrics):

    # Init accumulators for sequence
    acc = {}

    if 'CLEAR' in metrics:
        acc = eval_CLEAR(acc, gt_data, tracker_data, similarity_scores, alpha_range, meta_data)

    if 'HOTA' in metrics or 'ID' in metrics:
        # Calculate gtID/prID alignement over all timesteps
        global_alignment = calc_global_alignment(gt_data, tracker_data, similarity_scores, alpha_range, meta_data)
        if 'HOTA' in metrics:
            acc = eval_HOTA(acc, gt_data, tracker_data, similarity_scores, global_alignment, alpha_range)
        if 'ID' in metrics:
            acc = eval_ID(acc, global_alignment, alpha_range, meta_data)

    return acc

def calc_global_alignment(gt_data, tracker_data, similarity_scores, alpha_range, meta_data):

    # Output dict with variables to save
    var_names = ['potential_matches_count','gt_id_count','pr_id_count']
    global_alignment = {k: [] for k in var_names}

    # Loop over all alpha thresholds
    for a_id, alpha in enumerate(alpha_range):

        potential_matches_count = np.zeros((meta_data['num_gtIDs'], meta_data['num_trackIDs']))
        gt_id_count = np.zeros((meta_data['num_gtIDs']))
        pr_id_count = np.zeros((meta_data['num_trackIDs']))

        # Loop over all timesteps,
        # count the number of potential matches for each gtID/prID combo,
        # and the total number of dets for each gtID and prID
        for t, (time_gt, time_data) in enumerate(zip(gt_data, tracker_data)):
            gt_ids = time_gt[1].astype(int)
            curr_ids = time_data[1].astype(int)

            gt_ids_mat = np.repeat(gt_ids[:, np.newaxis], len(curr_ids), axis=1)
            curr_ids_mat = np.repeat(curr_ids[np.newaxis, :], len(gt_ids), axis=0)
            matches_mask = np.greater(similarity_scores[t], alpha)
            potential_matches_count[gt_ids_mat[matches_mask],curr_ids_mat[matches_mask]] += 1

            gt_id_count[gt_ids] += 1
            pr_id_count[curr_ids] += 1

        global_alignment['potential_matches_count'].append(potential_matches_count)
        global_alignment['gt_id_count'].append(gt_id_count)
        global_alignment['pr_id_count'].append(pr_id_count)
    return global_alignment

def eval_HOTA(acc, gt_data, tracker_data, similarity_scores, global_alignment, alpha_range):

    # Add HOTA variables to accumulator
    acc_names = ['HOTA_TP','HOTA_FN','HOTA_FP','AssA','AssRe','AssPr','LocA']
    for name in acc_names:
        acc[name] = np.zeros((len(alpha_range)),dtype=np.float)

    # Loop over all alpha thresholds
    for a_id, alpha in enumerate(alpha_range):

        # Calculate the global alignment score (Jaccard Index) between each gtID/prID.
        potential_matches_count = global_alignment['potential_matches_count'][a_id]
        gt_id_count = global_alignment['gt_id_count'][a_id][:,np.newaxis]
        pr_id_count = global_alignment['pr_id_count'][a_id][np.newaxis,:]
        global_alignment_score = potential_matches_count / (gt_id_count +  pr_id_count - potential_matches_count)

        matches_count = np.zeros_like(potential_matches_count)

        # Loop over each timestep
        for t, (time_gt, time_data) in enumerate(zip(gt_data, tracker_data)):
            similarity = squarify(similarity_scores[t], 0)
            gt_ids = time_gt[1].astype(int)
            pred_ids = time_data[1].astype(int)

            # Deal with the case that there are no gtDet/prDet in a timestep.
            if len(gt_ids)==0:
                acc['HOTA_FP'][a_id] += len(pred_ids)
                matches_count[-1, pred_ids] += 1
                continue
            if len(pred_ids)==0:
                acc['HOTA_FN'][a_id] += len(gt_ids)
                matches_count[gt_ids, -1] += 1
                continue

            # Get matching score pair of Dets for optimizing HOTA
            score_mat = global_alignment_score[gt_ids[:, np.newaxis],pred_ids[np.newaxis, :]]
            score_mat = squarify(score_mat, 0)
            score_mat = score_mat.astype(np.float) + 1.0e4 + 1.0e-4 * similarity
            score_mat[np.less(similarity, alpha)] = -1.0e4

            # Hungarian algorithm to find best matches
            match_rows, match_cols =  linear_assignment_problem(score_mat)

            # Calculate and accumulate basic statistics
            num_matches = len(match_rows)
            acc['HOTA_TP'][a_id] += num_matches
            acc['HOTA_FN'][a_id] += len(gt_ids) - num_matches
            acc['HOTA_FP'][a_id] += len(pred_ids) - num_matches
            acc['LocA'][a_id] += sum(similarity[match_rows, match_cols])
            matches_count[gt_ids[match_rows], pred_ids[match_cols]] += 1

        # Calculate association scores for sequence.
        acc['AssA'][a_id] = np.sum(matches_count * matches_count / (gt_id_count +  pr_id_count - matches_count)) / np.maximum(1.0, acc['HOTA_TP'][a_id])
        acc['AssRe'][a_id] = np.sum(matches_count * matches_count / gt_id_count) / np.maximum(1.0, acc['HOTA_TP'][a_id])
        acc['AssPr'][a_id] = np.sum(matches_count * matches_count / pr_id_count) / np.maximum(1.0, acc['HOTA_TP'][a_id])

    # Calculate final scores
    acc['LocA'] = acc['LocA'] / np.maximum(1.0, acc['HOTA_TP'])
    acc['DetRe'] = acc['HOTA_TP'] / (acc['HOTA_TP'] + acc['HOTA_FN'])
    acc['DetPr'] = acc['HOTA_TP'] / (acc['HOTA_TP'] + acc['HOTA_FP'])
    acc['DetA'] = acc['HOTA_TP'] / (acc['HOTA_TP'] + acc['HOTA_FN'] + acc['HOTA_FP'])
    acc['HOTA'] = np.sqrt(acc['DetA'] * acc['AssA'])
    return acc

def eval_CLEAR(acc, gt_data, tracker_data, similarity_scores, alpha_range, meta_data):
    num_gt_ids = meta_data['num_gtIDs']

    # Add CLEAR variables to accumulator
    acc_names = ['CLR_TP','CLR_FN','CLR_FP','IDSW','MOTP','MT','PT','ML','Frag']
    for name in acc_names:
        acc[name] = np.zeros((len(alpha_range)), dtype=np.float)

    # Loop over all alpha thresholds
    for a_id, alpha in enumerate(alpha_range):
        gt_id_count = np.zeros((num_gt_ids)) # For MT/ML/PT
        gt_matched_count = np.zeros((num_gt_ids)) # For MT/ML/PT
        gt_frag_count = np.zeros((num_gt_ids)) # For counting Frag
        PrevPrID = np.NaN*np.zeros((num_gt_ids)) # For scoring IDS (previous prID for each gt over all prev timesteps)
        PrevPrID_only_prev = np.NaN * np.zeros((num_gt_ids))  # For matching IDS (previous prID for each gt over only last timestep)

        # Loop over timesteps
        for t, (time_gt, time_data) in enumerate(zip(gt_data, tracker_data)):
            similarity = squarify(similarity_scores[t], 0)
            gt_ids = time_gt[1].astype(int)
            pred_ids = time_data[1].astype(int)

            # Deal with the case that there are no gtDet/prDet in a timestep.
            if len(gt_ids)==0:
                acc['CLR_FP'][a_id] += len(pred_ids)
                continue
            if len(pred_ids)==0:
                acc['CLR_FN'][a_id] += len(gt_ids)
                continue

            # Calc score matrix to first minimise IDSWs from previous frame, and then maximise MOTP secondarily
            score_mat = (pred_ids[np.newaxis, :] == PrevPrID_only_prev[gt_ids[:, np.newaxis]])
            score_mat = squarify(score_mat, 0)
            score_mat = 1000*score_mat + similarity
            score_mat[similarity < alpha] = -1

            # Hungarian algorithm
            match_rows, match_cols = linear_assignment_problem(score_mat)
            matched_gt_ids = gt_ids[match_rows]
            matched_pred_ids = pred_ids[match_cols]

            # Calc IDSW for MOTA
            prev_matched_pred_ids = PrevPrID[matched_gt_ids]
            is_IDSW = (np.logical_not(np.isnan(prev_matched_pred_ids))) & (np.not_equal(matched_pred_ids, prev_matched_pred_ids))
            acc['IDSW'][a_id] += np.sum(is_IDSW)

            # Update counters for MT/ML/PT/Frag and record for IDS/Frag for next timestep
            gt_id_count[gt_ids] += 1
            gt_matched_count[matched_gt_ids] += 1
            not_previously_tracked = np.isnan(PrevPrID_only_prev)
            PrevPrID[matched_gt_ids] = matched_pred_ids
            PrevPrID_only_prev[:] = np.nan
            PrevPrID_only_prev[matched_gt_ids] = matched_pred_ids
            currently_tracked = np.logical_not(np.isnan(PrevPrID_only_prev))
            gt_frag_count += np.logical_and(not_previously_tracked, currently_tracked)

            # Calculate and accumulate basic statistics
            num_matches = len(matched_gt_ids)
            acc['CLR_TP'][a_id] += num_matches
            acc['CLR_FN'][a_id] += len(time_gt[0]) - num_matches
            acc['CLR_FP'][a_id] += len(time_data[0]) - num_matches
            acc['MOTP'][a_id] += sum(similarity[match_rows, match_cols])

        # Calculate MT/ML/PT/Frag
        tracked_ratio = gt_matched_count[gt_id_count>0]/gt_id_count[gt_id_count>0]
        acc['MT'][a_id] = np.sum(np.greater(tracked_ratio,0.8))
        acc['ML'][a_id] = np.sum(np.less(tracked_ratio, 0.2))
        acc['PT'][a_id] = sum(gt_id_count>0) - acc['MT'][a_id] - acc['ML'][a_id]
        acc['Frag'][a_id] = np.sum(np.subtract(gt_frag_count[gt_frag_count>0],1))

    # Calc final metrics
    acc['MODA'] = (acc['CLR_TP'] - acc['CLR_FP']) / (acc['CLR_TP'] + acc['CLR_FN'])
    acc['MOTA'] = (acc['CLR_TP'] - acc['CLR_FP'] - acc['IDSW']) / (acc['CLR_TP'] + acc['CLR_FN'])
    acc['Recall'] = (acc['CLR_TP']) / (acc['CLR_TP'] + acc['CLR_FN'])
    acc['Precision'] = (acc['CLR_TP']) / (acc['CLR_TP'] + acc['CLR_FP'])
    acc['MOTP'] = acc['MOTP'] / np.maximum(1.0, acc['CLR_TP'])
    return acc

def eval_ID(acc, global_alignment, alpha_range, meta_data):
    num_gt_ids = meta_data['num_gtIDs']
    num_tracker_ids = meta_data['num_trackIDs']

    # Add ID variables to accumulator
    acc_names = ['IDTP', 'IDFP', 'IDFN']
    for name in acc_names:
        acc[name] = np.zeros((len(alpha_range)), dtype=np.float)

    # Loop over alpha thresholds
    for a_id, alpha in enumerate(alpha_range):
        potential_matches_count = global_alignment['potential_matches_count'][a_id]
        gt_id_count = global_alignment['gt_id_count'][a_id]
        pr_id_count = global_alignment['pr_id_count'][a_id]

        # Calculate optimal assignment cost matrix for ID
        FPmat = np.zeros((num_gt_ids + num_tracker_ids, num_gt_ids + num_tracker_ids))
        FNmat = np.zeros((num_gt_ids + num_tracker_ids, num_gt_ids + num_tracker_ids))
        FPmat[num_gt_ids:, :num_tracker_ids] = 1e10
        FNmat[:num_gt_ids, num_tracker_ids:] = 1e10
        for gt_id in range(num_gt_ids):
            FNmat[gt_id, :num_tracker_ids] = gt_id_count[gt_id]
            FNmat[gt_id, num_tracker_ids + gt_id] = gt_id_count[gt_id]
        for pr_id in range(num_tracker_ids):
            FPmat[:num_gt_ids, pr_id] = pr_id_count[pr_id]
            FPmat[pr_id + num_gt_ids, pr_id] = pr_id_count[pr_id]
        FPmat[:num_gt_ids, :num_tracker_ids] -= potential_matches_count
        FNmat[:num_gt_ids, :num_tracker_ids] -= potential_matches_count

        # Hungarian algorithm
        match_cols, _, _ = hungarian(FPmat + FNmat)
        match_rows = np.arange(len(match_cols))

        # Accumulate basic statistics
        acc['IDFP'][a_id] = FPmat[match_rows, match_cols].sum().astype(np.int)
        acc['IDFN'][a_id] = FNmat[match_rows, match_cols].sum().astype(np.int)
        acc['IDTP'][a_id] = (gt_id_count.sum() - acc['IDFN'][a_id]).astype(np.int)

    # Calculate final ID scores
    acc['IDP'] = acc['IDTP'] / (acc['IDTP'] + acc['IDFP'])
    acc['IDR'] = acc['IDTP'] / (acc['IDTP'] + acc['IDFN'])
    acc['IDF1'] = acc['IDTP'] / (acc['IDTP'] + 0.5 * acc['IDFN'] + 0.5 * acc['IDFP'])
    return acc

