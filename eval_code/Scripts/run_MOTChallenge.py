
# Run HOTA + other metrics evaluation on MOTChallenge (https://motchallenge.net/)
# Code by Jonathon Luiten (https://github.com/JonathonLuiten/HOTA-metrics)

#######################################################################################################################

# CONFIG
# Default values below. Can either be changed here in the script, or as command line arguments.
# Example command line arguments below (default values again, can obviously just set one or two parameters and keep the rest of the defaults)
# --TRACKERS_FOLDER tracker_output/ --GT_FOLDER gt_data/ --OUTPUT_FOLDER output/ --GT_TO_USE MOT17_train --TRACKERS_TO_EVAL MOT17_train/Lif_T MOT17_train/SSAT --METRICS HOTA CLEAR ID --ALPHA_BEHAVIOUR default --PREPROC_TYPE default --PRINT_RESULTS True --PRINT_ONLY_COMBINED False --OUTPUT_TO_CSV True --CSV_OUT_FILE results.csv --CSV_OUT_MODE new --PLOT_CURVES True --USE_PARALLEL True --NUM_PARALLEL_CORES 8 --BREAK_ON_ERROR True --PRINT_PROGRESS True --PRINT_CONFIG True

class dotdict(dict):
    """dot.notation access to dictionary attributes - for config file simplicity and readability"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
c = dotdict({})

## Default data directories:
c.TRACKERS_FOLDER = 'tracker_output/'
c.GT_FOLDER = 'gt_data/'
c.OUTPUT_FOLDER = 'output/'

## What data to use for eval:
c.GT_TO_USE = 'MOT17_train'
c.TRACKERS_TO_EVAL = ['MOT17_train/Lif_T', 'MOT17_train/SSAT']
# c.GT_TO_USE = 'MOT20_train'
# c.TRACKERS_TO_EVAL = ['MOT20_train/LPC_MOT']

## What metrics to eval:
c.METRICS = ['HOTA','CLEAR','ID'] # Implemented: ['HOTA','CLEAR','ID']
c.ALPHA_BEHAVIOUR = 'default' # Implemented: 'default','fifty_only'
c.PREPROC_TYPE = 'default' # Implemented: 'default','none'

## How to output results:
c.PRINT_RESULTS = True
c.PRINT_ONLY_COMBINED = False
c.OUTPUT_TO_CSV = True
c.CSV_OUT_FILE = 'results.csv'
c.CSV_OUT_MODE = 'new' # Implemented: 'new','append'
c.PLOT_CURVES = True

## Program running options:
c.USE_PARALLEL = True
c.NUM_PARALLEL_CORES = 8
c.BREAK_ON_ERROR = True
c.PRINT_PROGRESS = True
c.PRINT_CONFIG = True

#######################################################################################################################

import time
from multiprocessing.pool import Pool
from functools import partial
import numpy as np
import os
import glob
import argparse
from eval_code.Datasets.MOTChallenge_Dataset import load_raw_MOTCha_seq, preproc
from eval_code.Eval.eval_sequence import eval_sequence
from eval_code.Eval.similarity import calc_similarities
from eval_code.Eval.eval_utils import combine_sequences, print_results, output_csv, plot_results

parser = argparse.ArgumentParser()
for setting in c.keys():
    if type(c[setting]) == list: parser.add_argument("--" + setting, nargs='+')
    else: parser.add_argument("--"+setting)
args = parser.parse_args().__dict__
for setting in args.keys():
    if args[setting] is not None:
        if type(c[setting]) == type(True):
            if args[setting] == 'True': x = True
            elif args[setting] == 'False': x = False
            else: raise Exception('Command line parameter ' + setting + 'must be True or False')
        elif type(c[setting]) == type(1):
            x = int(args[setting])
        else: x = args[setting]
        c[setting] = x

if c.PRINT_CONFIG:
    print('\nConfig:')
    for setting in c.keys(): print('%-20s : %-30s'%(setting, c[setting]))

if c.ALPHA_BEHAVIOUR == 'default':
    alpha_thresholds = np.arange(0.05, 1, 0.05)
elif c.ALPHA_BEHAVIOUR == 'fifty_only':
    alpha_thresholds = np.array([0.5])
else:
    raise Exception('Alpha behaviour not implemented')

def eval_single_sequence(seq, gt_fol, tracker_fol):
    try:
        tt = time.time()
        raw_gt_data, meta_data = load_raw_MOTCha_seq(seq, gt_fol,isGT=True)
        raw_tracker_data, meta_data = load_raw_MOTCha_seq(seq, tracker_fol, meta_data,isGT=False)
        raw_similarity_scores = calc_similarities(raw_gt_data, raw_tracker_data)
        if c.PREPROC_TYPE == 'none': meta_data['preproc_type'] = 'MOT15'
        gt_data, tracker_data, similarity_scores, meta_data = preproc(raw_gt_data, raw_tracker_data, raw_similarity_scores, meta_data)
        res = eval_sequence(gt_data, tracker_data, similarity_scores, alpha_thresholds, meta_data, c.METRICS)
        if c.PRINT_PROGRESS: print('%s finished in %.2f seconds'%(seq,time.time()-tt))
        return res
    except Exception as err:
        if c.BREAK_ON_ERROR: raise(err)
        else: print(err)


for tr_iteration,tracker in enumerate(c.TRACKERS_TO_EVAL):
    if c.PRINT_PROGRESS: print('\nEvaluating %s against %s\n' % (tracker, c.GT_TO_USE))
    tracker_fol = os.path.join(c.TRACKERS_FOLDER, tracker)
    gt_fol = os.path.join(c.GT_FOLDER, c.GT_TO_USE)
    seq_list = sorted([x.split('/')[-2] for x in glob.glob(gt_fol+'/*/')])

    time_start_for_all = time.time()
    if c.USE_PARALLEL:
        with Pool(c.NUM_PARALLEL_CORES) as pool:
          results = pool.map(partial(eval_single_sequence,gt_fol=gt_fol, tracker_fol=tracker_fol),seq_list)
          all_res = dict(zip(seq_list, results))
    else:
        all_res = {}
        for seq in seq_list:
            res = eval_single_sequence(seq,gt_fol,tracker_fol)
            all_res[seq] = res

    all_res['COMBINED'] = combine_sequences(all_res, c.METRICS)
    if c.PRINT_PROGRESS: print('\nAll sequences for %s finished in %.2f seconds' % (tracker, time.time() - time_start_for_all))
    if c.PRINT_RESULTS: print_results(all_res, tracker, c.METRICS, c.ALPHA_BEHAVIOUR, c.PRINT_ONLY_COMBINED)
    if c.OUTPUT_TO_CSV: output_csv(all_res,tracker,c.CSV_OUT_FILE,c.OUTPUT_FOLDER,c.ALPHA_BEHAVIOUR,c.CSV_OUT_MODE,tr_iteration)
    if c.PLOT_CURVES and c.ALPHA_BEHAVIOUR=='default': plot_results(all_res,alpha_thresholds,tracker,c.METRICS,c.OUTPUT_FOLDER)

