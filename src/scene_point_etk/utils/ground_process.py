import numpy as np
import scipy.stats.qmc

import py_utils.voxel_grid as voxel_grid
from .. import argoverse2


def infer_ground_points_by_av2_log_id(
    log_id, points, threshold=0.1, return_indices=False
):

    assert points.ndim == 2
    assert points.shape[1] == 3

    ground_height = argoverse2.get_ground_height_by_points(
        log_id,
        points,
        filled_value=np.nan,
    )

    is_ground = np.logical_and(
        ~np.isnan(ground_height),
        np.abs(ground_height - points[:, 2]) <= threshold,
    )

    ground_points = points[is_ground]

    if return_indices:
        return ground_points, np.nonzero(is_ground)[0]
    return ground_points


def _infer_anchor_points_by_av2_log_id(
    log_id,
    area_per_point=50,
    voxel_size=0.2,
):
    ground_points = argoverse2.get_ground_height_points_by_log_id(log_id)

    # create a 2D grid
    v_sizes = [voxel_size, voxel_size, np.inf]
    voxel_grid_obj = voxel_grid.VoxelGrid(ground_points, voxel_size=v_sizes)

    # estimate the number of points should be sampled
    min_vg_ind = np.min(voxel_grid_obj.voxel_indices, axis=0)[:2]
    max_vg_ind = np.max(voxel_grid_obj.voxel_indices, axis=0)[:2]
    longest_side = np.max(max_vg_ind - min_vg_ind)
    N_selected_points = longest_side**2 * voxel_size**2 // area_per_point

    # poisson disk sampling
    radius = np.sqrt(1.0 / N_selected_points / np.pi)
    engine = scipy.stats.qmc.PoissonDisk(2, radius=radius * 2)
    anchors = engine.random(N_selected_points)

    # recover the sample_points from [0, 1] to the real world
    anchors = anchors * longest_side + min_vg_ind
    anchors = anchors * voxel_size

    # infer z value by the ground height map
    z = argoverse2.get_ground_height_by_points(log_id, anchors)
    anchors = np.hstack([anchors, z[:, None]])
    anchors = anchors[~np.isnan(z)]

    return anchors


def infer_anchor_points_by_av2_log_id(
    log_id,
    area_per_point=50,
    voxel_size=0.2,
    FOV_cameras=["ring_front_left"],
    FOV_max_distance=10,
    FOV_min_distance=0.0,
):

    anchors = _infer_anchor_points_by_av2_log_id(
        log_id,
        area_per_point=area_per_point,
        voxel_size=voxel_size,
    )

    # filter out anchor points outside FOV
    cameras = argoverse2.CameraSequence(log_id, cameras=FOV_cameras)
    FOV_mask = cameras.FOV_mask(
        map_points=anchors,
        max_distance=FOV_max_distance,
        min_distance=FOV_min_distance,
        verbose=False,
    )
    anchors = anchors[FOV_mask]

    return anchors
