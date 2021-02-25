import sys
import os
from multiprocessing import freeze_support

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hota_metrics as hm  # noqa: E402
from hota_metrics import utils

if __name__ == '__main__':
    freeze_support()

    # Command line interface:
    default_eval_config = hm.Evaluator.get_default_eval_config()
    default_dataset_config = hm.datasets.General.get_default_dataset_config()
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
        evaluator = hm.Evaluator(eval_config)
        dataset_list = [hm.datasets.General(dataset_config)]
        metrics_list = []
        for metric in [hm.metrics.HOTA, hm.metrics.CLEAR, hm.metrics.Identity]:
            if metric.get_name() in metrics_config['METRICS']:
                metrics_list.append(metric())
        if len(metrics_list) == 0:
            raise Exception('No metrics selected for evaluation')
        evaluator.evaluate(dataset_list, metrics_list)