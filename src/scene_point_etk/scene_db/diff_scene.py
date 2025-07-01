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

import networkx as nx
import numpy as np

from .. import patch_db
from ..argoverse2 import CLOUD_COMPARE_DTYPE
import py_utils.utils as utils
import py_utils.voxel_grid as voxel_grid
import py_utils.visualization_pptk as visualization_pptk


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


def get_pca_obb_vertices(points):
    """
    Compute the oriented bounding box (OBB) vertices in world coordinates.

    Args:
        points: (N, 3) array of 3D points
        make_vertices_func: function(lx, ly, lz) -> (8, 3) local box corners

    Returns:
        vertices_world: (8, 3) array of box corners in world coordinates
    """
    # 1. Center the points
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    # 2. PCA: get rotation axes
    cov = np.cov(centered_points, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]  # (3,3) rotation matrix

    # 3. Rotate points to local frame
    points_local = centered_points @ axes

    # 4. Get extents in local frame
    min_local = np.min(points_local, axis=0)
    max_local = np.max(points_local, axis=0)
    size = max_local - min_local  # (lx, ly, lz)

    # 5. Compute OBB center in local frame
    obb_center_local = (min_local + max_local) / 2.0

    # 6. Transform center to world frame
    obb_center_world = centroid + axes @ obb_center_local

    # 7. Get local vertices relative to local origin
    vertices_local = visualization_pptk.make_bounding_box_vertices(
        *size
    )  # (8, 3)

    # 8. Transform local vertices to world frame
    vertices_world = obb_center_world + vertices_local @ axes.T

    return vertices_world


def get_deleted_pcd_bounding_boxes(scene_pcd, delete_info):

    if len(delete_info) == 0:
        return {}

    annots = delete_info.get("annotations", None)
    margin = delete_info.get("margin", 0.0)

    xyz = np.vstack([scene_pcd["x"], scene_pcd["y"], scene_pcd["z"]]).T

    results = {}
    categories = np.unique(annots.category)
    for category in categories:

        annot = annots.filter_by_category(category)
        indices = annot.is_points_in_bounding_boxes(
            xyz,
            margin=margin,
            separate=True,
        )
        indices = [set(i.tolist()) for i in indices]

        graph = nx.Graph()
        graph.add_nodes_from(range(len(indices)))

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if len(indices[i].intersection(indices[j])) > 0:
                    graph.add_edge(i, j)

        results[category] = []
        for cluster in nx.connected_components(graph):

            cluster_indices = np.hstack([list(indices[i]) for i in cluster])
            cluster_indices = np.unique(cluster_indices).astype(np.int32)
            cluster_xyz = xyz[cluster_indices]
            results[category].append(get_pca_obb_vertices(cluster_xyz))

    return results


def get_added_pcd(scene_pcd, add_info, return_splits=False):

    if len(add_info) == 0:
        if return_splits:
            return np.empty(0, dtype=np.dtype(CLOUD_COMPARE_DTYPE)), []
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
        patch = voxel_grid.subtract_pcds(
            patch,
            scene_pcd,
            voxel_size=voxel_size,
        )

        added_pcd.append(patch)

    splits = np.cumsum([len(i) for i in added_pcd])[:-1]
    added_pcd = np.concatenate(added_pcd)

    if return_splits:
        return added_pcd, splits
    return added_pcd


def get_added_pcd_bounding_boxes(add_info):

    if len(add_info) == 0:
        return {}

    # extract information from add_info
    patches = add_info["patches"]
    anchor_xyzs = add_info["anchor_xyzs"]
    anchor_euls = add_info["anchor_eulers"]
    z_offset = add_info.get("z_offset", 0.0)

    assert len(patches) == len(anchor_xyzs) == len(anchor_euls)

    anchor_Rs = utils.euler_to_R(anchor_euls)

    results = {}
    for key, anchor_xyz, anchor_R in zip(patches, anchor_xyzs, anchor_Rs):

        patch = patch_db.get_patch_from_key(key).copy()

        x_min = np.min(patch["x"])
        x_max = np.max(patch["x"])
        y_min = np.min(patch["y"])
        y_max = np.max(patch["y"])
        z_min = np.min(patch["z"])
        z_max = np.max(patch["z"])

        lx = x_max - x_min
        ly = y_max - y_min
        lz = z_max - z_min

        bbox_pts = visualization_pptk.make_bounding_box_vertices(lx, ly, lz)

        # align the lowest point of the patch to the anchor_xyz
        i = np.argmin(patch["z"])

        bottom_xyz = np.r_[patch["x"][i], patch["y"][i], patch["z"][i]]
        pos_diff = anchor_xyz - bottom_xyz
        bbox_pts_world = bbox_pts @ anchor_R + pos_diff

        # updated the patch coordinates to world coordinates
        x, y, z = bbox_pts_world.T
        z = z + z_offset

        results[key] = np.vstack([x, y, z]).T

    return results


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
    s = np.diff(s).astype(np.int64)
    segment_info = np.repeat(np.arange(len(s) - 1), s)
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
