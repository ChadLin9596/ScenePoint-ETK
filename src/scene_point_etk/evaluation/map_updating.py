import numpy as np
import scipy.spatial

try:
    import pptk.kdtree

    _PPTK_IS_IMPORTED = True
except:
    _PPTK_IS_IMPORTED = False

#####################################
# DISTANCE METRICS FOR POINT CLOUDS #
######################################


def nearest_distance(point_cloud_1, point_cloud_2):

    global _PPTK_IS_IMPORTED
    if _PPTK_IS_IMPORTED:

        point_cloud_1 = point_cloud_1.astype(np.float64)
        point_cloud_2 = point_cloud_2.astype(np.float64)

        tree = pptk.kdtree._build(point_cloud_2)
        Is = pptk.kdtree._query(tree, point_cloud_1, k=1)
        I = [i[0] for i in Is]
        distances = point_cloud_1 - point_cloud_2[I]
        distances = np.sqrt(np.sum(distances**2, axis=-1))

    else:
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


def all_point_cloud_metrics(
    point_cloud_1,
    point_cloud_2,
    return_details=False,
):
    """
    Compute all distance metrics at once and return as a dictionary.
    """

    details = {
        "pts_1_to_2": np.full((len(point_cloud_1),), np.inf, dtype=np.float64),
        "pts_2_to_1": np.full((len(point_cloud_2),), np.inf, dtype=np.float64),
    }

    if len(point_cloud_1) == 0 and len(point_cloud_2) == 0:

        default = {
            "chamfer_dist": 0,
            "hausdorff_dist": 0,
            "modified_hausdorff_dist": 0,
            "median_point_dist": 0,
        }

        if return_details:
            return default, details
        return default

    if len(point_cloud_1) == 0 or len(point_cloud_2) == 0:
        default = {
            "chamfer_dist": np.inf,
            "hausdorff_dist": np.inf,
            "modified_hausdorff_dist": np.inf,
            "median_point_dist": np.inf,
        }
        if return_details:
            return default, details
        return default

    dist_pc1_to_pc2 = nearest_distance(point_cloud_1, point_cloud_2)
    dist_pc2_to_pc1 = nearest_distance(point_cloud_2, point_cloud_1)

    details["pts_1_to_2"][:] = dist_pc1_to_pc2
    details["pts_2_to_1"][:] = dist_pc2_to_pc1

    M = {
        "chamfer_dist": _chamfer_dist_by_nearest_dist,
        "hausdorff_dist": _hausdorff_dist_by_nearest_dist,
        "modified_hausdorff_dist": _modified_hausdorff_dist_by_nearest_dist,
        "median_point_dist": _median_point_dist_by_nearest_dist,
    }

    results = {}
    for key, func in M.items():
        results[key] = func(dist_pc1_to_pc2, dist_pc2_to_pc1)

    if return_details:
        return results, details
    return results


########################################
# VOXEL-BASED METRICS FOR POINT CLOUDS #
########################################


def _voxel_confusion_matrix(
    pred_voxels,
    gt_voxels,
    threshold=0.1,
    return_details=False,
):

    distances_from_pred_to_gt = np.full(len(pred_voxels), np.inf)
    distances_from_gt_to_pred = np.full(len(gt_voxels), np.inf)
    if len(gt_voxels) > 0:
        distances_from_pred_to_gt = nearest_distance(pred_voxels, gt_voxels)
    if len(pred_voxels) > 0:
        distances_from_gt_to_pred = nearest_distance(gt_voxels, pred_voxels)

    tp_voxels = pred_voxels[distances_from_pred_to_gt <= threshold]
    fp_voxels = pred_voxels[distances_from_pred_to_gt > threshold]
    fn_voxels = gt_voxels[distances_from_gt_to_pred > threshold]

    tp = int(len(tp_voxels))
    fp = int(len(fp_voxels))
    fn = int(len(fn_voxels))

    results = dict(tp=tp, fp=fp, fn=fn)
    details = dict(tp=tp_voxels, fp=fp_voxels, fn=fn_voxels)

    if return_details:
        return results, details
    return results


def voxel_classification_metrics(
    pred_voxels,
    gt_voxels,
    threshold=0.1,
    return_details=False,
):

    d, details = _voxel_confusion_matrix(
        pred_voxels,
        gt_voxels,
        threshold=threshold,
        return_details=True,
    )
    tp, fp, fn = d["tp"], d["fp"], d["fn"]

    precision = np.nan
    if tp + fp > 0:
        precision = tp / (tp + fp)

    recall = np.nan
    if tp + fn > 0:
        recall = tp / (tp + fn)

    f1 = np.nan
    if (tp + fp + fn) > 0:
        f1 = 2 * tp / (2 * tp + fp + fn)

    results = dict(precision=precision, recall=recall, f1=f1)
    if return_details:
        return results, details
    return results
