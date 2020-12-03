
import os
import csv


def init_config(config, default_config, name):
    """Initialise non-given config values with defaults"""
    if config is None:
        config = default_config
    else:
        for k in default_config.keys():
            if k not in config.keys():
                config[k] = default_config[k]
    if config['PRINT_CONFIG']:
        print('\n%s Config:' % name)
        for c in config.keys():
            print('%-20s : %-30s' % (c, config[c]))
    return config


def get_code_path():
    """Get base path where code is"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def write_summary_results(summaries, cls, output_folder):
    """Write summary results to file"""
    headers = sum([list(s.keys()) for s in summaries], [])
    values = sum([list(s.values()) for s in summaries], [])
    out_file = os.path.join(output_folder, cls + '_summary.txt')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow(headers)
        writer.writerow(values)


def write_detailed_results(details, cls, output_folder):
    """Write detailed results to file"""
    sequences = details[0].keys()
    headers = ['seq'] + sum([list(s['COMBINED_SEQ'].keys()) for s in details], [])
    out_file = os.path.join(output_folder, cls + '_detailed.csv')
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for seq in sorted(sequences):
            if seq == 'COMBINED_SEQ':
                continue
            writer.writerow([seq] + sum([list(s[seq].values()) for s in details], []))
        writer.writerow(['COMBINED'] + sum([list(s['COMBINED_SEQ'].values()) for s in details], []))
