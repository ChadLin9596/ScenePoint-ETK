import numpy as np
import pptk

import py_utils.visualization_pptk as visualization_pptk


def plot_mulitple_pcds(*pcds, scale=[0, 100], color_map=None):
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

    v = pptk.viewer(xyz)
    v.attributes(*rgbas)

    return v
