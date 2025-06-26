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
        patch = voxel_grid.unique_pcd(patch, voxel_size=voxel_size)

        # get the patch points that are not overlapping with the scene_pcd
        # no need to unique the scene_pcd, as it is already unique
        patch = voxel_grid.subtract_pcds(patch, scene_pcd, voxel_size=voxel_size)

        added_pcd.append(patch)

    splits = np.cumsum([len(i) for i in added_pcd])[:-1]
    added_pcd = np.concatenate(added_pcd)

    if return_splits:
        return added_pcd, splits
    return added_pcd


def apply_change_info_to_target_pcd(
    target_pcd,
    change_info,
    return_details=False,
):

    voxel_size = change_info.get("voxel_size", 0.2)
    delete_info = change_info.get("delete", {})
    add_info = change_info.get("add", {})

    # removed points from delete_info
    del_pcd, mask = get_deleted_pcd(target_pcd, delete_info, return_mask=True)
    target_pcd = target_pcd[~mask]

    # added points from add_info
    additional_pcd, split = get_added_pcd(
        target_pcd,
        add_info,
        return_splits=True,
    )
    updated_pcd = np.concatenate([target_pcd, additional_pcd])

    s = np.r_[0, split, len(additional_pcd)]
    segment_info = np.repeat(np.arange(len(s) - 1), np.diff(s))
    segment_info = np.r_[np.ones(len(target_pcd)) * -1, segment_info]

    # re-sort again, but does not need to re-voxelize
    xyz = np.vstack([updated_pcd["x"], updated_pcd["y"], updated_pcd["z"]]).T
    _, _, I = voxel_grid._sort_for_voxel_grid(xyz, voxel_size)
    updated_pcd = updated_pcd[I]
    segment_info = segment_info[I]

    # TODO:
    # get the segment_info from added_indices_of_source and split
    # added_indices_of_source = np.argsort(I)
    # added_indices_of_source = added_indices_of_source[len(target_pcd) :]

    # create indices of source for each segment
    segment_infos = []
    for i in range(len(split) + 1):
        segment_infos.append(np.where(segment_info == i)[0])

    details = {
        "deleted_points": del_pcd,
        "added_points": additional_pcd,
        "added_splits": split,
        "deleted_indices_of_target": np.where(mask)[0],
        "added_segment_indices_of_source": segment_infos,
    }

    if return_details:
        return updated_pcd, details

    return updated_pcd
