"""
FORMAT:

delete_info:{
    "annotations": <Annotation object> or None,
    "margin": <float>,
    "indices": <list of int>
}

add_info:{
    "patches": <list of patch keys> (n),
    "anchor_xyzs": (n, 3) array,
    "anchor_eulers": (n, 3) array,
    "z_offset": <float> (optional, default 0.0)
    "voxel_size": <float> (optional, default 0.2)
}
"""

import numpy as np

from .. import patch_db
from ..argoverse2 import CLOUD_COMPARE_DTYPE
import py_utils.utils as utils
import py_utils.voxel_grid as voxel_grid


def unique_pcd(pcd_arr, voxel_size=0.2):

    pcd_indices = pcd_arr["center"] // voxel_size
    pcd_indices = pcd_indices.astype(np.int64)

    # unique indices
    _, unique_indices = np.unique(pcd_indices, axis=0, return_index=True)
    unique_indices = np.sort(unique_indices)
    return pcd_arr[unique_indices]


def subtract_pcds(pcd_a, pcd_b, voxel_size=0.2):

    pcd_a_indices = (pcd_a["center"] // voxel_size).astype(np.int64)
    pcd_b_indices = (pcd_b["center"] // voxel_size).astype(np.int64)

    # Use view-based conversion for efficient comparison
    def to_unique_1d(array):
        return array.view([("", array.dtype)] * array.shape[1]).ravel()

    a_voxel_keys = to_unique_1d(pcd_a_indices)
    b_voxel_keys = to_unique_1d(pcd_b_indices)

    # keys in 'a' but not in 'b'
    keep_mask = ~np.isin(a_voxel_keys, b_voxel_keys)
    return pcd_a[keep_mask]


def get_deleted_pcd(scene_pcd, delete_info, return_mask=False):

    # case: no delete point
    if len(delete_info) == 0:
        out = np.empty(0, dtype=np.dtype(CLOUD_COMPARE_DTYPE))

        if return_mask:
            return out, np.zeros(len(scene_pcd), dtype=bool)
        return out

    mask = np.zeros(len(scene_pcd), dtype=bool)

    # delete by annotations:
    annot = delete_info.get("annotations", None)
    if annot is not None:
        margin = delete_info.get("margin", 0.0)
        xyz = np.vstack([scene_pcd["x"], scene_pcd["y"], scene_pcd["z"]]).T
        indices = annot.is_points_in_bounding_boxes(xyz, margin=margin)
        mask[indices] = True

    # delete by indices:
    indices = np.r_[delete_info["indices"]]
    indices = np.unique(indices).astype(np.int64)
    mask[indices] = True

    if return_mask:
        return scene_pcd[mask], mask
    return scene_pcd[mask]


def get_added_pcd(scene_pcd, add_info, return_splits=False):

    if len(add_info) == 0:
        return np.empty(0, dtype=np.dtype(CLOUD_COMPARE_DTYPE))

    # extract information from add_info
    patches = add_info["patches"]
    anchor_xyzs = add_info["anchor_xyzs"]
    anchor_euls = add_info["anchor_eulers"]
    z_offset = add_info.get("z_offset", 0.0)
    voxel_size = add_info.get("voxel_size", 0.2)
    assert len(patches) == len(anchor_xyzs) == len(anchor_euls)

    anchor_Rs = utils.euler_to_R(anchor_euls)

    added_pcd = []
    for patch, anchor_xyz, anchor_R in zip(patches, anchor_xyzs, anchor_Rs):

        patch = patch_db.get_patch_from_key(patch).copy()
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
        patch = unique_pcd(patch, voxel_size=voxel_size)

        # get the patch points that are not overlapping with the scene_pcd
        # no need to unique the scene_pcd, as it is already unique
        patch = subtract_pcds(patch, scene_pcd, voxel_size=voxel_size)

        added_pcd.append(patch)

    splits = np.cumsum([len(i) for i in added_pcd])[:-1]
    added_pcd = np.concatenate(added_pcd)

    if return_splits:
        return added_pcd, splits
    return added_pcd


def apply_change_info_to_target_pcd(target_pcd, change_info):

    voxel_size = change_info.get("voxel_size", 0.2)
    detete_info = change_info.get("delete", {})
    add_info = change_info.get("add", {})

    # removed points from detete_info
    _, mask = get_deleted_pcd(target_pcd, detete_info, return_mask=True)
    target_pcd = target_pcd[~mask]

    # added points from add_info
    additional_pcd = get_added_pcd(target_pcd, add_info)
    target_pcd = np.concatenate([target_pcd, additional_pcd])

    # re-sort again, but does not need to re-voxelize
    xyz = np.vstack([target_pcd["x"], target_pcd["y"], target_pcd["z"]]).T
    _, _, I = voxel_grid._sort_for_voxel_grid(xyz, voxel_size)
    target_pcd = target_pcd[I]

    return target_pcd
