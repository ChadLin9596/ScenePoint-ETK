import numpy as np
import pptk

import py_utils.visualization_pptk as visualization_pptk


def _check_rgb(rgb, msg="RGB values must be in the range [0, 1]"):

    if np.any(rgb < 0) or np.any(rgb > 1) or np.shape(rgb)[-1] != 3:
        raise ValueError(msg)


def _check_alpha(alpha, msg="Alpha values must be in the range [0, 1]"):

    if np.max(alpha) > 1 or np.min(alpha) < 0:
        raise ValueError(msg)


def _normalized_attributes(attributes, scale=[0, 100], color_map=None):

    make_color = visualization_pptk.make_color(*scale, color_map=color_map)

    rgbas = []
    for attr in attributes:

        attr = np.asarray(attr)

        # 1st case: if attr is a 2D array with 4 columns, treat it as RGBA
        if attr.ndim == 2 and attr.shape[1] == 4:

            _check_rgb(attr[:, :3])
            _check_alpha(attr[:, 3])
            rgbas.append(attr)
            continue

        # 2nd case: if attr is a 2D array with 3 columns, treat it as RGB
        if attr.ndim == 2 and attr.shape[1] == 3:

            _check_rgb(attr)
            alpha = np.ones(attr.shape[0])
            rgba = np.vstack([attr.T, alpha]).T
            rgbas.append(rgba)
            continue

        # 3rd case: if attr is a 2D array with 2 columns, treat is as [attr, A]
        if attr.ndim == 2 and attr.shape[1] == 2:

            rgb = make_color(attr[:, 0])
            alpha = attr[:, 1]

            msg = "color_map should be an (n, 3) array in the range [0, 1]"
            _check_rgb(rgb, msg)
            _check_alpha(alpha)

            rgba = np.vstack([rgb.T, alpha]).T
            rgbas.append(rgba)
            continue

        # 4th case: if attr is a 2D array with 1 column or 1D array
        #           treat it as attr
        if (attr.ndim == 2 and attr.shape[1] == 1) or attr.ndim == 1:
            attr = attr.flatten()
            rgb = make_color(attr)
            msg = "color_map should be an (n, 3) array in the range [0, 1]"
            _check_rgb(rgb, msg)

            alpha = np.ones(attr.shape[0])
            rgba = np.vstack([rgb.T, alpha]).T
            rgbas.append(rgba)
            continue

        raise ValueError(
            "Attributes must be a 2D array with 1, 2, 3, or 4 columns, "
            "or a 1D array. Received shape: {}".format(np.shape(attr))
        )

    return np.vstack(rgbas)


def _normalized_bboxes(bbox_list=[], bbox_color_list=[]):

    if len(bbox_list) == 0:
        dummy_bbox = np.empty((0, 8, 3))
        dummy_bbox_color = np.empty((0, 4))
        return [dummy_bbox], [dummy_bbox_color]

    default_bbox_color = [1, 1, 0, 1]  # Yellow color for bounding boxes

    # 1st case: if bbox_color_list is an empty list, use default color
    if len(bbox_color_list) == 0:
        bbox_color_list = [default_bbox_color] * len(bbox_list)

    # 2nd case: if bbox_color_list is a single color, repeat it for each bbox
    elif len(bbox_color_list) == 1:
        bbox_color_list = [bbox_color_list[0]] * len(bbox_list)
        bbox_color_list = np.asarray(bbox_color_list)
        c = np.shape(bbox_color_list)[-1]

        if c == 3:
            alpha = np.ones(len(bbox_color_list))
            bbox_color_list = np.vstack([bbox_color_list.T, alpha]).T
        elif c == 4:
            _check_alpha(bbox_color_list[:, 3])
        else:
            raise ValueError(
                "bbox_color_list must be an (n, 3) or (n, 4) array. "
                "Received shape: {}".format(np.shape(bbox_color_list))
            )
        bbox_color_list = bbox_color_list.tolist()

    # 3rd case: if bbox_color_list is a list of colors, check each color
    elif len(bbox_list) == len(bbox_color_list):

        bbox_color_list_ = []
        for bbox_color in bbox_color_list:

            bbox_color = np.asarray(bbox_color)
            assert bbox_color.ndim == 1
            assert bbox_color.shape[0] in [3, 4]

            if len(bbox_color) == 3:
                bbox_color = np.r_[bbox_color, 1]

            _check_rgb(bbox_color[:3])
            _check_alpha(bbox_color[3])
            bbox_color_list_.append(bbox_color.tolist())
        bbox_color_list = bbox_color_list_

    else:
        N1 = len(bbox_list)
        N2 = len(bbox_color_list)
        raise ValueError(
            "bbox_list and bbox_color_list must have the same length. "
            "Received lengths: {}, {}".format(N1, N2)
        )

    bbox_color_list = np.asarray(bbox_color_list)

    new_bbox_list = []
    new_bbox_color_list = []

    for bbox, bbox_color in zip(bbox_list, bbox_color_list):

        bbox = np.asarray(bbox)

        if len(np.shape(bbox)) == 1:
            raise ValueError(
                "Bounding box vertices must be a 2D array with shape (8, 3). "
                "Received shape: {}".format(np.shape(bbox))
            )

        assert np.shape(bbox)[-2:] == (8, 3), (
            "Bounding box vertices must have shape (8, 3). "
            "Received shape: {}".format(np.shape(bbox))
        )

        # single bounding box case
        if len(np.shape(bbox)) == 2:
            new_bbox_list.append(bbox)
            new_bbox_color_list.append(bbox_color)
            continue

        # multiple bounding boxes case
        bbox = np.reshape(bbox, (-1, 8, 3))
        n = np.shape(bbox)[0]

        bbox_color = np.repeat(bbox_color[None, ...], n, axis=0)
        new_bbox_list.append(bbox)
        new_bbox_color_list.append(bbox_color)

    return new_bbox_list, new_bbox_color_list


def _process_pcds(*pcds, scale=[0, 100], color_map=None):
    """
    Process point cloud data to extract xyz and attributes.
    Returns a tuple of (xyz, attributes).
    """

    if scale is not None and len(scale) != 2:
        raise ValueError("scale must be a tuple of two floats")

    xyzs = []
    attributes = []

    for p in pcds:
        assert len(p) == 2
        assert len(p[0]) == len(p[1])
        xyzs.append(p[0])
        attributes.append(p[1])

    splits = np.r_[0, np.cumsum([len(x) for x in xyzs])]

    xyz = np.vstack(xyzs)
    rgba = _normalized_attributes(attributes, scale=scale, color_map=color_map)

    rgbas = []
    for i, j in zip(splits[:-1], splits[1:]):

        mask_rgba = rgba.copy()
        mask_rgba[:, 3] = 0  # Set alpha to 0 for the entire array
        mask_rgba[i:j, 3] = rgba[i:j, 3]
        rgbas.append(mask_rgba)

    return xyz, rgbas


def _process_bboxes(bbox_list, bbox_color_list, bbox_eps=0.05):

    bbox_list, bbox_color_list = _normalized_bboxes(bbox_list, bbox_color_list)

    assert len(bbox_list) == len(bbox_color_list)
    assert isinstance(bbox_list, list)
    assert isinstance(bbox_color_list, list)

    for bbox, bbox_color in zip(bbox_list, bbox_color_list):
        bbox = np.asarray(bbox)
        bbox_color = np.asarray(bbox_color)
        assert bbox.ndim == 3 and bbox.shape[-2:] == (8, 3)
        assert bbox_color.ndim == 2 and bbox_color.shape[-1] == 4
        assert len(bbox) == len(bbox_color)

    xyzs = []
    rgbas = []

    for bbox, bbox_color in zip(bbox_list, bbox_color_list):

        if len(bbox) == 0:
            continue

        xyz = visualization_pptk.make_bounding_boxes_lines(bbox, eps=bbox_eps)
        lengths = np.array([len(x) for x in xyz])
        rgba = np.repeat(bbox_color, lengths, axis=0)
        xyz = np.vstack(xyz)
        xyzs.append(xyz)
        rgbas.append(rgba)

    if len(xyzs) == 0:
        return np.empty((0, 3)), np.empty((0, 4))

    xyzs = np.vstack(xyzs)
    rgbas = np.vstack(rgbas)
    return xyzs, rgbas


def plot_multiple_pcds(*pcds, scale=[0, 100], color_map=None):
    """
    TODO: add docstring
    """

    xyz, rgbas = _process_pcds(*pcds, scale=scale, color_map=color_map)
    v = pptk.viewer(xyz, debug=False)
    v.attributes(*rgbas)

    return v


def plot_multiple_pcds_with_bboxes(
    *pcds,
    scale=[0, 100],
    color_map=None,
    bbox_list=[],
    bbox_color_list=[],
    bbox_eps=0.05,
):

    # if len(bbox_list) == 0:
    #     return plot_multiple_pcds(*pcds, scale=scale, color_map=color_map)

    xyz, rgbas = _process_pcds(*pcds, scale=scale, color_map=color_map)
    bbox_xyz, bbox_rgba = _process_bboxes(bbox_list, bbox_color_list, bbox_eps)

    xyz = np.vstack([xyz, bbox_xyz])
    rgbas = [np.vstack([rgba, bbox_rgba]) for rgba in rgbas]

    # Launch viewer
    v = pptk.viewer(xyz, debug=False)
    v.attributes(*rgbas)

    return v
