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
from .. import patch_db

import py_utils.utils as utils
import py_utils.voxel_grid as voxel_grid


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


def modified_pcd_projection(xyz):

    # move points to the origin
    mean = np.mean(xyz, axis=0)
    xyz = xyz - mean

    # rotate the points
    cov = np.dot(xyz.T, xyz)
    eigen_values, eigen_vectors = np.linalg.eig(cov)
    eigen_values, eigen_vectors

    theta = np.arctan2(eigen_vectors[1, 0], eigen_vectors[0, 0])
    R = utils.euler_to_R(np.array([0, 0, theta]))

    return R, mean


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


def infer_merge_indices(scene_pcd, add_info):
    """
    Infer merge indices for the added patches based on the existing scene_pcd.
    This is used to avoid adding points that are already in the scene_pcd.
    """
    patches = add_info["patches"]
    anchor_xyzs = add_info["anchor_xyzs"]
    anchor_euls = add_info["anchor_eulers"]
    z_offset = add_info.get("z_offset", 0.0)
    voxel_size = add_info.get("voxel_size", 0.2)

    assert len(patches) == len(anchor_xyzs) == len(anchor_euls)

    anchor_Rs = utils.euler_to_R(anchor_euls)

    merge_indices = []
    for key, anchor_xyz, anchor_R in zip(patches, anchor_xyzs, anchor_Rs):

        patch = patch_db.get_patch_from_key(key).copy()
        patch_xyz = np.vstack([patch["x"], patch["y"], patch["z"]]).T

        # align the lowest point of the patch to the anchor_xyz
        i = np.argmin(patch_xyz[:, 2])
        pos_diff = anchor_xyz - patch_xyz[i]
        patch_xyz_world = patch_xyz @ anchor_R + pos_diff

        # updated the patch coordinates to world coordinates
        x, y, z = patch_xyz_world.T
        patch["x"] = x
        patch["y"] = y
        patch["z"] = z + z_offset

        # update the patch's center
        patch["center"] = patch_xyz_world // voxel_size + 0.5 * voxel_size

        _, indices = voxel_grid.unique_pcd(
            patch,
            voxel_size=voxel_size,
            return_indices=True,
        )

        # get the patch points that are not overlapping with the scene_pcd
        # no need to unique the scene_pcd, as it is already unique
        _, mask = voxel_grid.subtract_pcds(
            patch[indices],
            scene_pcd,
            voxel_size=voxel_size,
            return_mask=True,
        )

        merge_indices.append(indices[mask])

    return merge_indices
