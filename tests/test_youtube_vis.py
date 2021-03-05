import sys
import os
import numpy as np
from multiprocessing import freeze_support

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
metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
default_track_map_config = trackeval.metrics.TrackMAP.get_default_metric_config()
default_track_map_config['USE_TIME_RANGES'] = False
default_track_map_config['AREA_RANGES'] = [[0 ** 2, 128 ** 2],
                                           [ 128 ** 2, 256 ** 2],
                                           [256 ** 2, 1e5 ** 2]]
metrics_list.append(trackeval.metrics.TrackMAP(default_track_map_config))

tests = [
    {'SPLIT_TO_EVAL': 'train_sub_split', 'TRACKERS_TO_EVAL': ['STEm_Seg']},
]

for dataset_config in tests:

    dataset_list = [trackeval.datasets.YouTubeVIS(dataset_config)]
    file_loc = os.path.join('youtube_vis', 'youtube_vis_' + dataset_config['SPLIT_TO_EVAL'])

    raw_results, messages = evaluator.evaluate(dataset_list, metrics_list)

    tracker = dataset_config['TRACKERS_TO_EVAL'][0]
    test_data_loc = os.path.join(os.path.dirname(__file__), '..', 'data', 'tests', file_loc)
    tracker_data_dir = os.path.join(test_data_loc, tracker)
    classes = [cls.split("_detailed.csv")[0] for cls in os.listdir(tracker_data_dir)]

    for cls in classes:
        results = {seq: raw_results['YouTubeVIS'][tracker][seq][cls] for seq in raw_results['YouTubeVIS'][tracker].keys()}
        current_metrics_list = metrics_list + [trackeval.metrics.Count()]
        metric_names = trackeval.utils.validate_metrics_list(current_metrics_list)

        # Load expected results:
        test_data = trackeval.utils.load_detail(os.path.join(test_data_loc, tracker, cls + '_detailed.csv'))

        # Do checks
        for seq in ['COMBINED_SEQ']:
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

        print('Class %s tests passed' % cls)

    print('Tracker %s tests passed' % tracker)
print('All tests passed')