""" run_unified.py

Run example:
run_unfied.py --USE_PARALLEL False --BENCHMARK davis_unsupervised --SPLIT_TO_EVAL val

Command Line Arguments: Defaults, # Comments
    Eval arguments:
            'USE_PARALLEL': False,
            'NUM_PARALLEL_CORES': 8,
            'BREAK_ON_ERROR': True,  # Raises exception and exits with error
            'RETURN_ON_ERROR': False,  # if not BREAK_ON_ERROR, then returns from function on error
            'LOG_ON_ERROR': os.path.join(code_path, 'error_log.txt'),  # if not None, save any errors into a log file.
            'PRINT_RESULTS': True,
            'PRINT_ONLY_COMBINED': False,
            'PRINT_CONFIG': True,
            'TIME_PROGRESS': True,
            'DISPLAY_LESS_PROGRESS': True,
            'OUTPUT_SUMMARY': True,
            'OUTPUT_EMPTY_CLASSES': True,  # If False, summary files are not output for classes with no detections
            'OUTPUT_DETAILED': True,
            'PLOT_CURVES': True,
    Dataset arguments:
            'GT_FOLDER': os.path.join(code_path, 'data/converted_gt'),  # Location of GT data
            'TRACKERS_FOLDER': os.path.join(code_path, 'data/converted_trackers'),  # Trackers location
            'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            'TRACKERS_TO_EVAL': None,  # Filenames of trackers to eval (if None, all in folder)
            'BENCHMARK': None,  # valid: 'MOT17', 'MOT16', 'MOT20', 'MOT15', 'MOTS', 'kitti_2d_box', 'kitti_mots',
                                # 'bdd100k_2d_box', 'davis_unsupervised', 'tao', 'youtube_vis'
            'CLASSES_TO_EVAL': None,  # if None, all valid classes
            'SPLIT_TO_EVAL': None,
            'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
            'PRINT_CONFIG': True,  # Whether to print current config
            'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/DATA_LOC_FORMAT/OUTPUT_SUB_FOLDER
            'TRACKER_SUB_FOLDER': 'data',  # Tracker files are in TRACKER_FOLDER/DATA_LOC_FORMAT/TRACKER_SUB_FOLDER
            'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
            'SEQMAP_FOLDER': None,  # Where seqmaps are found (if None, GT_FOLDER/dataset_subfolder/seqmaps)
            'SEQMAP_FILE': None,  # Directly specify seqmap file (if none use SEQMAP_FOLDER/BENCHMARK_SPLIT_TO_EVAL)
            'CLSMAP_FOLDER': None,  # Where seqmaps are found (if None, GT_FOLDER/dataset_subfolder/clsmaps)
            'CLSMAP_FILE': None,  # Directly specify seqmap file (if none use CLSMAP_FOLDER/BENCHMARK_SPLIT_TO_EVAL)
            'DATA_LOC_FORMAT': '{dataset}/{benchmark}_{split}/',    # data localization format for GT, Tracker
                                                                    # and output subfolder structure
    Script arguments:
            'BENCHMARKS':   ['MOT20', 'MOTS', 'kitti_2d_box', 'kitti_mots', 'bdd100k_2d_box', 'davis_unsupervised',
                            'tao', 'youtube_vis'],
            'SPLITS_TO_EVAL': ['train', 'train', 'training', 'val', 'val', 'val', 'training', 'train_sub_split']
"""

import sys
import os
from multiprocessing import freeze_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval  # noqa: E402
from trackeval import utils

if __name__ == '__main__':
    freeze_support()

    script_config = {'BENCHMARKS': ['MOT20', 'MOTS', 'kitti_2d_box', 'kitti_mots', 'bdd100k_2d_box',
                                    'davis_unsupervised', 'tao', 'youtube_vis'],
                     'SPLITS_TO_EVAL': ['train', 'train', 'training', 'val', 'val', 'val', 'training',
                                        'train_sub_split']}

    # Command line interface:
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['PRINT_ONLY_COMBINED'] = True
    default_eval_config['DISPLAY_LESS_PROGRESS'] = True
    default_dataset_config = trackeval.datasets.Unified.get_default_dataset_config()
    config = {**default_eval_config, **default_dataset_config, **script_config}
    config = utils.update_config(config)
    benchmarks = config['BENCHMARKS']
    splits_to_eval = config['SPLITS_TO_EVAL']
    print(benchmarks)

    for i in range(len(benchmarks)):
        benchmark = benchmarks[i]
        if benchmark in ['tao', 'youtube_vis']:
            metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity', 'TrackMAP']}
        elif benchmark =='davis_unsupervised':
            metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity', 'JAndF']}
        else:
            metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity']}

        eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
        dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
        dataset_config['BENCHMARK'] = benchmark
        dataset_config['SPLIT_TO_EVAL'] = splits_to_eval[i]

        # Run code
        evaluator = trackeval.Evaluator(eval_config)
        dataset_list = [trackeval.datasets.Unified(dataset_config)]
        metrics_list = []
        for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity,
                       trackeval.metrics.TrackMAP, trackeval.metrics.JAndF]:
            if benchmark == 'youtube_vis' and metric == trackeval.metrics.TrackMAP:
                default_track_map_config = metric.get_default_metric_config()
                default_track_map_config['USE_TIME_RANGES'] = False
                default_track_map_config['AREA_RANGES'] = [[0 ** 2, 128 ** 2],
                                                           [ 128 ** 2, 256 ** 2],
                                                           [256 ** 2, 1e5 ** 2]]
                metrics_list.append(metric(default_track_map_config))
            elif metric.get_name() in metrics_config['METRICS']:
                metrics_list.append(metric())
        if len(metrics_list) == 0:
            raise Exception('No metrics selected for evaluation')
        evaluator.evaluate(dataset_list, metrics_list)