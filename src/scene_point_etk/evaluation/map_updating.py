import numpy as np
import scipy.spatial


def nearest_distance(point_cloud_1, point_cloud_2):

    tree = scipy.spatial.cKDTree(point_cloud_2)
    distances, _ = tree.query(point_cloud_1, k=1)
    return distances


def _chamfer_dist_by_nearest_dist(dist_pc1_to_pc2, dist_pc2_to_pc1):
    cd = np.mean(dist_pc1_to_pc2**2) + np.mean(dist_pc2_to_pc1**2)
    return cd


def _hausdorff_dist_by_nearest_dist(dist_pc1_to_pc2, dist_pc2_to_pc1):
    return max(np.max(dist_pc1_to_pc2), np.max(dist_pc2_to_pc1))


def _modified_hausdorff_dist_by_nearest_dist(dist_pc1_to_pc2, dist_pc2_to_pc1):
    return max(np.mean(dist_pc1_to_pc2), np.mean(dist_pc2_to_pc1))


def _median_point_dist_by_nearest_dist(dist_pc1_to_pc2, dist_pc2_to_pc1):
    return max(np.median(dist_pc1_to_pc2), np.median(dist_pc2_to_pc1))


def chamfer_distance(point_cloud_1, point_cloud_2):

    if len(point_cloud_1) == 0 and len(point_cloud_2) == 0:
        return 0.0

    if len(point_cloud_1) == 0 or len(point_cloud_2) == 0:
        return np.inf

    func = _chamfer_dist_by_nearest_dist
    dist_pc1_to_pc2 = nearest_distance(point_cloud_1, point_cloud_2)
    dist_pc2_to_pc1 = nearest_distance(point_cloud_2, point_cloud_1)
    return func(dist_pc1_to_pc2, dist_pc2_to_pc1)


def hausdorff_distance(point_cloud_1, point_cloud_2):

    if len(point_cloud_1) == 0 and len(point_cloud_2) == 0:
        return 0.0

    if len(point_cloud_1) == 0 or len(point_cloud_2) == 0:
        return np.inf

    func = _hausdorff_dist_by_nearest_dist
    dist_pc1_to_pc2 = nearest_distance(point_cloud_1, point_cloud_2)
    dist_pc2_to_pc1 = nearest_distance(point_cloud_2, point_cloud_1)
    return func(dist_pc1_to_pc2, dist_pc2_to_pc1)


def modified_hausdorff_distance(point_cloud_1, point_cloud_2):

    if len(point_cloud_1) == 0 and len(point_cloud_2) == 0:
        return 0.0

    if len(point_cloud_1) == 0 or len(point_cloud_2) == 0:
        return np.inf

    func = _modified_hausdorff_dist_by_nearest_dist
    dist_pc1_to_pc2 = nearest_distance(point_cloud_1, point_cloud_2)
    dist_pc2_to_pc1 = nearest_distance(point_cloud_2, point_cloud_1)
    return func(dist_pc1_to_pc2, dist_pc2_to_pc1)


def median_point_distance(point_cloud_1, point_cloud_2):

    if len(point_cloud_1) == 0 and len(point_cloud_2) == 0:
        return 0.0

    if len(point_cloud_1) == 0 or len(point_cloud_2) == 0:
        return np.inf

    func = _median_point_dist_by_nearest_dist
    dist_pc1_to_pc2 = nearest_distance(point_cloud_1, point_cloud_2)
    dist_pc2_to_pc1 = nearest_distance(point_cloud_2, point_cloud_1)
    return func(dist_pc1_to_pc2, dist_pc2_to_pc1)


def all_point_cloud_metrics(point_cloud_1, point_cloud_2):
    """
    Compute all distance metrics at once and return as a dictionary.
    """

    if len(point_cloud_1) == 0 and len(point_cloud_2) == 0:
        return {
            "chamfer_dist": 0,
            "hausdorff_dist": 0,
            "modified_hausdorff_dist": 0,
            "median_point_dist": 0,
        }

    if len(point_cloud_1) == 0 or len(point_cloud_2) == 0:
        return {
            "chamfer_dist": np.inf,
            "hausdorff_dist": np.inf,
            "modified_hausdorff_dist": np.inf,
            "median_point_dist": np.inf,
        }

    dist_pc1_to_pc2 = nearest_distance(point_cloud_1, point_cloud_2)
    dist_pc2_to_pc1 = nearest_distance(point_cloud_2, point_cloud_1)

    M = {
        "chamfer_dist": _chamfer_dist_by_nearest_dist,
        "hausdorff_dist": _hausdorff_dist_by_nearest_dist,
        "modified_hausdorff_dist": _modified_hausdorff_dist_by_nearest_dist,
        "median_point_dist": _median_point_dist_by_nearest_dist,
    }

    results = {}
    for key, func in M.items():
        results[key] = func(dist_pc1_to_pc2, dist_pc2_to_pc1)

    return results
