import sys
import os
from multiprocessing import freeze_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval  # noqa: E402
from trackeval import utils

if __name__ == '__main__':
    freeze_support()

    # Command line interface:
    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['PRINT_ONLY_COMBINED'] = True
    default_eval_config['DISPLAY_LESS_PROGRESS'] = True
    default_dataset_config = trackeval.datasets.General.get_default_dataset_config()
    config = {**default_eval_config, **default_dataset_config}
    config = utils.update_config(config)
    benchmarks = ['TAO']
    splits_to_eval = ['training']

    for i in range(len(benchmarks)):
        benchmark = benchmarks[i]
        if benchmark in ['TAO', 'YouTubeVIS']:
            metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity', 'TrackMAP']}
        elif benchmark =='DAVIS':
            metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity', 'JAndF']}
        else:
            metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity']}

        eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
        dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
        dataset_config['BENCHMARK'] = benchmark
        dataset_config['SPLIT_TO_EVAL'] = splits_to_eval[i]

        # Run code
        evaluator = trackeval.Evaluator(eval_config)
        dataset_list = [trackeval.datasets.General(dataset_config)]
        metrics_list = []
        for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity,
                       trackeval.metrics.TrackMAP, trackeval.metrics.JAndF]:
            if benchmark == 'YouTubeVIS' and metric == trackeval.metrics.TrackMAP:
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