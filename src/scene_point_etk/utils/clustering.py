import sklearn.cluster
import numpy as np

import py_utils.utils as utils


def cluster_voxel_grid_by_DBSCAN(
    pcd_points,
    eps=0.3,
    min_samples=10,
):

    dbscan = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples)
    xyz = np.vstack([pcd_points["x"], pcd_points["y"], pcd_points["z"]]).T
    labels = dbscan.fit_predict(xyz)

    return labels


def filter_voxel_grid_by_DBSCAN(
    pcd_points,
    eps=0.3,
    min_samples=10,
    min_cluster_size=80,
    return_details=False,
):

    labels = cluster_voxel_grid_by_DBSCAN(
        pcd_points, eps=eps, min_samples=min_samples
    )

    counts = utils.group_sizes_by_label(labels)

    mask = (labels != -1) & (counts >= min_cluster_size)

    if not np.any(mask):
        return []

    pcd_points = pcd_points[mask]
    labels = labels[mask]

    # resorted by labels
    I = np.argsort(labels)
    labels = labels[I]
    pcd_points = pcd_points[I]

    splits = np.where(np.diff(labels))[0] + 1

    cluster_pcd = np.split(pcd_points, splits)

    # TODO:
    details = {}

    if return_details:
        return cluster_pcd, details
    return cluster_pcd
