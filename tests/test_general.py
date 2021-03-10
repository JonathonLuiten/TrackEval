import sys
import os
import numpy as np
from multiprocessing import freeze_support
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval  # noqa: E402

# Fixes multiprocessing on windows, does nothing otherwise
if __name__ == '__main__':
    freeze_support()

eval_config = {'USE_PARALLEL': False,
               'NUM_PARALLEL_CORES': 8,
               'PRINT_RESULTS': False,
               'PRINT_CONFIG': True,
               'TIME_PROGRESS': True,
               'DISPLAY_LESS_PROGRESS': True,
               'OUTPUT_SUMMARY': False,
               'OUTPUT_EMPTY_CLASSES': False,
               'OUTPUT_DETAILED': False,
               'PLOT_CURVES': False,
               }
evaluator = trackeval.Evaluator(eval_config)

tests = [
    #{'BENCHMARK': 'BDD100K', 'SPLIT_TO_EVAL': 'val', 'TRACKERS_TO_EVAL': ['qdtrack']},
    #{'BENCHMARK': 'DAVIS', 'SPLIT_TO_EVAL': 'val', 'TRACKERS_TO_EVAL': ['ags']},
    {'BENCHMARK': 'YouTubeVIS', 'SPLIT_TO_EVAL': 'train_sub_split', 'TRACKERS_TO_EVAL': ['STEm_Seg'] },
    #{'BENCHMARK': 'TAO', 'SPLIT_TO_EVAL': 'training', 'TRACKERS_TO_EVAL': ['Tracktor++']},
    #{'BENCHMARK': 'KittiMOTS', 'SPLIT_TO_EVAL': 'val', 'TRACKERS_TO_EVAL': ['trackrcnn']},
    #{'BENCHMARK': 'MOTS', 'SPLIT_TO_EVAL': 'train', 'TRACKERS_TO_EVAL': ['TrackRCNN']},
    #{'BENCHMARK': 'Kitti2DBox', 'SPLIT_TO_EVAL': 'training', 'TRACKERS_TO_EVAL': ['CIWT']},
    #{'BENCHMARK': 'MOT15', 'SPLIT_TO_EVAL': 'train', 'TRACKERS_TO_EVAL': ['MPNTrack']},
    #{'BENCHMARK': 'MOT16', 'SPLIT_TO_EVAL': 'train', 'TRACKERS_TO_EVAL': ['MPNTrack']},
    #{'BENCHMARK': 'MOT17', 'SPLIT_TO_EVAL': 'train', 'TRACKERS_TO_EVAL': ['MPNTrack']},
    #{'BENCHMARK': 'MOT20', 'SPLIT_TO_EVAL': 'train', 'TRACKERS_TO_EVAL': ['MPNTrack']},
]

for dataset_config in tests:

    dataset_list = [trackeval.datasets.General(dataset_config)]
    if dataset_config['BENCHMARK'] == 'BDD100K':
        file_loc = os.path.join('bdd100k', 'bdd100k_' + dataset_config['SPLIT_TO_EVAL'])
        metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    elif dataset_config['BENCHMARK'] == 'DAVIS':
        file_loc = os.path.join('davis', 'davis_unsupervised_' + dataset_config['SPLIT_TO_EVAL'])
        metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity(),
                        trackeval.metrics.JAndF()]
    elif dataset_config['BENCHMARK'] == 'YouTubeVIS':
        file_loc = os.path.join('youtube_vis', 'youtube_vis_' + dataset_config['SPLIT_TO_EVAL'])
        metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
        default_track_map_config = trackeval.metrics.TrackMAP.get_default_metric_config()
        default_track_map_config['USE_TIME_RANGES'] = False
        default_track_map_config['AREA_RANGES'] = [[0 ** 2, 128 ** 2],
                                                   [ 128 ** 2, 256 ** 2],
                                                   [256 ** 2, 1e5 ** 2]]
        metrics_list.append(trackeval.metrics.TrackMAP(default_track_map_config))
    elif dataset_config['BENCHMARK'] == 'TAO':
        file_loc = os.path.join('tao', 'tao_' + dataset_config['SPLIT_TO_EVAL'])
        metrics_list = [trackeval.metrics.Identity(), trackeval.metrics.CLEAR(), trackeval.metrics.HOTA(),
                        trackeval.metrics.TrackMAP()]
    elif dataset_config['BENCHMARK'] == 'KittiMOTS':
        file_loc = os.path.join('kitti', 'kitti_mots_' + dataset_config['SPLIT_TO_EVAL'])
        metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    elif dataset_config['BENCHMARK'] in ['MOTS', 'MOT15', 'MOT16', 'MOT17', 'MOT20']:
        file_loc = os.path.join('mot_challenge', dataset_config['BENCHMARK'] + '-' + dataset_config['SPLIT_TO_EVAL'])
        metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    elif dataset_config['BENCHMARK'] == 'Kitti2DBox':
        file_loc = os.path.join('kitti', 'kitti_2d_box_' + dataset_config['SPLIT_TO_EVAL'])
        metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    else:
        raise Exception("Benchmark %s not implemented." % dataset_config['BENCHMARK'])

    raw_results, messages = evaluator.evaluate(dataset_list, metrics_list)

    trackers = dataset_config['TRACKERS_TO_EVAL']
    test_data_loc = os.path.join(os.path.dirname(__file__), '..', 'data', 'tests', file_loc)

    for tracker in trackers:
        tracker_data_dir = os.path.join(test_data_loc, tracker)
        classes = [cls.split("_detailed.csv")[0] for cls in os.listdir(tracker_data_dir)]
        for cls in classes:
            results = {seq: raw_results['General'][tracker][seq][cls] for seq in raw_results['General'][tracker].keys()}
            current_metrics_list = metrics_list + [trackeval.metrics.Count()]
            metric_names = trackeval.utils.validate_metrics_list(current_metrics_list)

            # Load expected results:
            test_data = trackeval.utils.load_detail(os.path.join(test_data_loc, tracker, cls + '_detailed.csv'))

            # Do checks
            for seq in tqdm(test_data.keys(), desc='Testing sequences for class %s' % cls):
                assert len(test_data[seq].keys()) > 250, len(test_data[seq].keys())

                details = []
                for metric, metric_name in zip(current_metrics_list, metric_names):
                    table_res = {seq_key: seq_value[metric_name] for seq_key, seq_value in results.items()}
                    details.append(metric.detailed_results(table_res))
                res_fields = sum([list(s['COMBINED_SEQ'].keys()) for s in details], [])
                res_values = sum([list(s[seq].values()) for s in details], [])
                res_dict = dict(zip(res_fields, res_values))

                for field in test_data[seq].keys():
                    assert np.isclose(res_dict[field], test_data[seq][field]), \
                        seq + ': ' + cls + ': ' + field + " (" + str(res_dict[field]) + "," \
                        + str(test_data[seq][field]) + ")"

        print('Tracker %s tests passed' % tracker)
    print('Benchmark %s tests passed' % dataset_config['BENCHMARK'])
print('All tests passed')
