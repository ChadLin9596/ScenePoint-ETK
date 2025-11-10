"""
Argoverse2 dataset management module aiming to access sensor data
through log ids.


city:

    "ATX": "Austin, Texas"
    "DTW": "Detroit, Michigan"
    "MIA": "Miami, Florida"
    "PAO": "Palo Alto, California"
    "PIT": "Pittsburgh, PA"
    "WDC": "Washington, DC"

city origin:

    "ATX": (30.27464237939507, -97.7404457407424)
    "DTW": (42.29993066912924, -83.17555750783717)
    "MIA": (25.77452579915163, -80.19656914449405)
    "PAO": (37.416065, -122.13571963362166)
    "PIT": (40.44177902989321, -80.01294377242584)
    "WDC": (38.889377, -77.0355047439081)
"""

from .av2_basic_tools import *
from .av2_annotations import Annotations
from .av2_image_sequence import ImageSequence, CameraSequence
from .av2_sweep import Sweep, SweepSequence, CLOUD_COMPARE_DTYPE
