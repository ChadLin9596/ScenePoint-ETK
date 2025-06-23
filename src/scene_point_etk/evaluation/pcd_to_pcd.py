import numpy as np
import scipy.spatial

import py_utils.voxel_grid as voxel_grid


def chamfer_distance(point_cloud_1, point_cloud_2):

    tree_1 = scipy.spatial.cKDTree(point_cloud_1)
    tree_2 = scipy.spatial.cKDTree(point_cloud_2)

    distances_pc1_to_pc2, _ = tree_2.query(point_cloud_1, k=1)
    distances_pc2_to_pc1, _ = tree_1.query(point_cloud_2, k=1)

    cd = np.mean(distances_pc1_to_pc2**2) + np.mean(distances_pc2_to_pc1**2)
    return cd


def intersection_over_union_by_distance(
    point_cloud_1,
    point_cloud_2,
    threshold=0.1,
):

    tree_1 = scipy.spatial.cKDTree(point_cloud_1)
    tree_2 = scipy.spatial.cKDTree(point_cloud_2)

    distances_pc1_to_pc2, _ = tree_1.query(point_cloud_2, k=1)
    distances_pc2_to_pc1, _ = tree_2.query(point_cloud_1, k=1)

    intersection = np.sum(distances_pc1_to_pc2 < threshold)
    intersection += np.sum(distances_pc2_to_pc1 < threshold)

    union = len(point_cloud_1) + len(point_cloud_2)
    iou = intersection / union
    return iou


def intersection_over_union_by_occupancy_grid(
    point_cloud_1,
    point_cloud_2,
    voxel_size=0.2,
):

    vg_1 = voxel_grid.VoxelGrid(point_cloud_1, voxel_size)
    vg_2 = voxel_grid.VoxelGrid(point_cloud_2, voxel_size)

    # (n, 3) -> (m, 3)
    voxel_indices_1 = vg_1.voxel_indices
    voxel_indices_2 = vg_2.voxel_indices

    # make voxel indices semi-positive
    min_x = min(np.min(voxel_indices_1[:, 0]), np.min(voxel_indices_2[:, 0]))
    min_y = min(np.min(voxel_indices_1[:, 1]), np.min(voxel_indices_2[:, 1]))
    min_z = min(np.min(voxel_indices_1[:, 2]), np.min(voxel_indices_2[:, 2]))
    offset = np.array([min_x, min_y, min_z])
    voxel_indices_1 = voxel_indices_1 - offset
    voxel_indices_2 = voxel_indices_2 - offset

    # reduce voxel indices to 1D indices
    max_y = max(np.max(voxel_indices_1[:, 1]), np.max(voxel_indices_2[:, 1]))
    max_z = max(np.max(voxel_indices_1[:, 2]), np.max(voxel_indices_2[:, 2]))

    voxel_indices_1 = (
        voxel_indices_1[:, 0] * (max_y + 1) * (max_z + 1)
        + voxel_indices_1[:, 1] * (max_z + 1)
        + voxel_indices_1[:, 2]
    )
    voxel_indices_2 = (
        voxel_indices_2[:, 0] * (max_y + 1) * (max_z + 1)
        + voxel_indices_2[:, 1] * (max_z + 1)
        + voxel_indices_2[:, 2]
    )

    # compute intersection and union
    intersection = np.intersect1d(
        voxel_indices_1,
        voxel_indices_2,
        assume_unique=True,
    )
    union = np.union1d(voxel_indices_1, voxel_indices_2)

    iou = len(intersection) / len(union)

    return iou


def precision(predict_point_cloud, target_point_cloud, threshold=0.1):
    tree = scipy.spatial.cKDTree(target_point_cloud)
    distances, _ = tree.query(predict_point_cloud, k=1)
    precision = np.sum(distances < threshold) / len(distances)
    return precision


def recall(predict_point_cloud, target_point_cloud, threshold=0.1):
    tree = scipy.spatial.cKDTree(predict_point_cloud)
    distances, _ = tree.query(target_point_cloud, k=1)
    recall = np.sum(distances < threshold) / len(distances)
    return recall


def F_score(predict_point_cloud, target_point_cloud, threshold=0.1):

    precision_score = precision(
        predict_point_cloud,
        target_point_cloud,
        threshold=threshold,
    )

    recall_score = recall(
        predict_point_cloud,
        target_point_cloud,
        threshold=threshold,
    )

    score = 2 * precision_score * recall_score
    score = score / (precision_score + recall_score + 1e-6)

    return score
