
from typing import Dict, List, Union, Tuple

import numpy as np

DT = Dict[str, Union[str, int, List[np.ndarray]]]
FMT = np.ndarray[Tuple[int, int], np.dtype[np.float64]]
