import numpy as np

from . import s5_cmd
from .ground_process import (
    infer_ground_points_by_av2_log_id,
    infer_anchor_points_by_av2_log_id,
)
from .clustering import (
    cluster_voxel_grid_by_DBSCAN,
    filter_voxel_grid_by_DBSCAN,
    cluster_overlapping_lists,
)


def encode_rgba(r, g, b, a=255):
    """Encode RGB and alpha values into a single 32bytes."""

    if np.isscalar(r):
        r = [r]
    if np.isscalar(g):
        g = [g]
    if np.isscalar(b):
        b = [b]
    if np.isscalar(a):
        a = [a]

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

    bgra = np.asarray(bgra.copy(), dtype=np.float32)
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


def filter_visible(
    bg_depth_image,  # (H, W)
    fg_depth_image,  # (H, W)
    FOV_mask,  # (H, W) boolean mask of dense points
    neighborhood_size=7,  # Size of neighborhood for depth comparison
    depth_threshold=0.1,  # Depth threshold for visibility
    num_valid_points=3,
):

    assert bg_depth_image.shape == fg_depth_image.shape
    assert FOV_mask.shape == bg_depth_image.shape

    pad = neighborhood_size // 2

    H, W = FOV_mask.shape
    FOV_mask = FOV_mask & (bg_depth_image > 0)

    us, vs = np.argwhere(FOV_mask).T
    for u, v in zip(us, vs):
        u_min = max(0, u - pad)
        u_max = min(H, u + pad + 1)
        v_min = max(0, v - pad)
        v_max = min(W, v + pad + 1)

        depth_patch = fg_depth_image[u_min:u_max, v_min:v_max]
        valid_patch = depth_patch[depth_patch > 0]

        if len(valid_patch) < num_valid_points:
            continue

        bg_depth = bg_depth_image[u, v]
        fg_depth = np.min(valid_patch)

        if (bg_depth - fg_depth) <= depth_threshold:
            FOV_mask[u, v] = False

    return FOV_mask


def filter_visible_old(
    static_depth_map,  # (H, W)
    lidar_depth_map,  # (H, W)
    overall_sparse_mask,  # (H, W) boolean mask of dense points
    neighborhood_size=3,  # Size of neighborhood for depth comparison
    depth_threshold=0.1,  # Depth threshold for visibility
    num_valid_points=1,
):

    pad = neighborhood_size // 2

    valid_static = static_depth_map > 0
    valid_lidar = lidar_depth_map > 0
    visible_mask = overall_sparse_mask & valid_static

    H, W = static_depth_map.shape

    us, vs = np.argwhere(visible_mask).T
    for u, v in zip(us, vs):
        u_min = max(0, u - pad)
        u_max = min(H, u + pad + 1)
        v_min = max(0, v - pad)
        v_max = min(W, v + pad + 1)

        lidar_patch = lidar_depth_map[u_min:u_max, v_min:v_max]
        valid_lidar_patch = lidar_patch[lidar_patch > 0]

        if len(valid_lidar_patch) < num_valid_points:
            continue

        lidar_depth = np.min(valid_lidar_patch)
        static_depth = static_depth_map[u, v]

        if (static_depth - lidar_depth) > depth_threshold:
            visible_mask[u, v] = False

    return visible_mask
