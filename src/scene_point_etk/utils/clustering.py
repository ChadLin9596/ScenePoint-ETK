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


def cluster_overlapping_lists(lists_of_indices):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    # Step 1: Union all overlapping values
    for group in lists_of_indices:
        for i in range(1, len(group)):
            union(group[i - 1], group[i])

    # Step 2: Group by root
    clusters = {}
    for group in lists_of_indices:
        for item in group:
            root = find(item)
            if root not in clusters:
                clusters[root] = set()
            clusters[root].add(item)

    return list(clusters.values())
