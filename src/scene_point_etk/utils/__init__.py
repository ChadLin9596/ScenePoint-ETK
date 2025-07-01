import numpy as np

from . import s5_cmd
from .ground_process import (
    infer_ground_points_by_av2_log_id,
    infer_anchor_points_by_av2_log_id,
)
from .clustering import (
    cluster_voxel_grid_by_DBSCAN,
    filter_voxel_grid_by_DBSCAN,
)


def encode_rgba(r, g, b, a=255):
    """Encode RGB and alpha values into a single 32bytes."""

    r = np.asarray(r, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)

    if np.all(r <= 1.0) and np.all(g <= 1.0) and np.all(b <= 1.0):
        r = r * 255
        g = g * 255
        b = b * 255

    if a <= 1.0:
        a = a * 255

    r = r.astype(np.uint8)
    g = g.astype(np.uint8)
    b = b.astype(np.uint8)
    a = a.astype(np.uint8)

    assert len(r) == len(g) == len(b), "r, g, b must have the same length"

    if len(a) != len(r) and len(a) == 1:
        a = np.full_like(r, a[0], dtype=np.uint8)

    assert len(a) == len(r), "a must have the same length as r, g, b"

    # encode for CloudCompare
    bgra = np.vstack([b, g, r, a]).T
    bgra = bgra.astype(np.uint8).tobytes()
    bgra = np.frombuffer(bgra, dtype=np.float32)

    return bgra


def decode_rgba(bgra):
    """Decode a 32bytes RGBA value into r, g, b, a."""

    bgra = np.asarray(bgra, dtype=np.float32)
    bgra = bgra.view(np.uint8).reshape(-1, 4)

    b = bgra[:, 0]
    g = bgra[:, 1]
    r = bgra[:, 2]
    a = bgra[:, 3]

    if np.all(r <= 255) and np.all(g <= 255) and np.all(b <= 255):
        r = r / 255.0
        g = g / 255.0
        b = b / 255.0

    if np.all(a <= 255):
        a = a / 255.0

    return r, g, b, a
