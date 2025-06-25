import numpy as np
import pptk

import py_utils.visualization_pptk as visualization_pptk


def plot_multiple_pcds(*pcds, scale=[0, 100], color_map=None):
    """
    TODO: add docstring
    """

    if scale is not None and len(scale) != 2:
        raise ValueError("scale must be a tuple of two floats")

    xyzs = []
    attributes = []

    for p in pcds:
        assert len(p) == 2
        xyzs.append(p[0])
        attributes.append(p[1])

    splits = np.r_[0, np.cumsum([len(x) for x in xyzs])]

    xyz = np.vstack(xyzs)

    f = visualization_pptk.make_color(*scale, color_map=color_map)

    rgb = []
    for attr in attributes:
        shp = np.shape(attr)
        if len(shp) == 2 and shp[-1] == 3:
            rgb.append(attr)
            continue
        rgb.append(f(attr))
    rgb = np.vstack(rgb)

    rgbas = []
    for i, j in zip(splits[:-1], splits[1:]):
        N = len(rgb)
        rgba = np.vstack([rgb.T, np.zeros(N)]).T
        rgba[i:j, 3] = 1
        rgbas.append(rgba)

    v = pptk.viewer(xyz, debug=True)
    v.attributes(*rgbas)

    return v


def plot_multiple_pcds_with_bboxes(
    *pcds,
    scale=[0, 100],
    color_map=None,
    bboxes=None,
    bbox_color=[1, 1, 0],
    bbox_eps=0.05,
):

    if bboxes is None:
        return plot_multiple_pcds(*pcds, scale=scale, color_map=color_map)

    if scale is not None and len(scale) != 2:
        raise ValueError("scale must be a tuple of two floats")

    xyzs = []
    attributes = []

    for p in pcds:
        assert len(p) == 2
        xyzs.append(p[0])
        attributes.append(p[1])

    splits = np.r_[0, np.cumsum([len(x) for x in xyzs])]

    xyz = np.vstack(xyzs)

    # Handle attribute coloring
    f = visualization_pptk.make_color(*scale, color_map=color_map)

    rgb = []
    for attr in attributes:
        if attr.ndim == 2 and attr.shape[1] == 3:
            rgb.append(attr)
        else:
            rgb.append(f(attr))
    rgb = np.vstack(rgb)

    # Create RGBA attributes with 1 for visible, 0 for invisible
    rgbas = []
    for i, j in zip(splits[:-1], splits[1:]):
        N = len(rgb)
        rgba = np.vstack([rgb.T, np.zeros(N)]).T
        rgba[i:j, 3] = 1
        rgbas.append(rgba)

    # Add bounding box lines if provided
    line_xyz = visualization_pptk.make_bounding_boxes_lines(
        bboxes, eps=bbox_eps
    )
    line_xyz = np.vstack(line_xyz)
    line_rgba = np.repeat(
        np.r_[bbox_color, 1].reshape(1, 4),
        len(line_xyz),
        axis=0,
    )

    xyz = np.vstack([xyz, line_xyz])
    rgbas = [np.vstack([rgba, line_rgba]) for rgba in rgbas]

    # Launch viewer
    v = pptk.viewer(xyz, debug=True)
    v.attributes(*rgbas)

    return v
