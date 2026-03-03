"""."""

import argparse

from typing import Dict, Sequence, Union


def modify_with_cmd_line(config: Dict[str, Union[str, int, float, bool]],
                         args: Union[None, Sequence[str]]) -> None:
    parser = argparse.ArgumentParser()
    for setting in config.keys():
        if isinstance(config[setting], list) or config[setting] is None:
            parser.add_argument("--" + setting, nargs='+')
        else:
            parser.add_argument("--" + setting)
    namespace = parser.parse_args(args).__dict__

    for setting in namespace.keys():
        if namespace[setting] is not None:
            if type(config[setting]) == type(True):
                if namespace[setting] == 'True':
                    x = True
                elif namespace[setting] == 'False':
                    x = False
                else:
                    raise Exception('Command line parameter ' + setting + 'must be True or False')
            elif type(config[setting]) == type(1):
                x = int(namespace[setting])
            elif type(namespace[setting]) == type(None):
                x = None
            else:
                x = namespace[setting]
            config[setting] = x