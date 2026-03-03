from typing import Union

from .clear import CLEAR
from .count import Count
from trackeval.metrics.hota.hota import HOTA
from .identity import Identity
from .ideucl import IDEucl
from .j_and_f import JAndF
from .track_map import TrackMAP
from .vace import VACE

MET = Union[HOTA, CLEAR, Identity]