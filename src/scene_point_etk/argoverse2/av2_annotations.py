import os

import numpy as np
import pyarrow.feather as feather
import scipy.spatial

import py_utils.array_data as array_data
import py_utils.visualization_pptk as visualization_pptk
import py_utils.utils as utils

from .av2_basic_tools import (
    check_log_id,
    get_root_from_log_id,
    get_poses_by_log_id,
    ArgoMixin,
)


class Annotations(ArgoMixin, array_data.TimePoseSequence):

    _columns = ["length", "width", "height", "category"]
    _dtypes = [np.float64, np.float64, np.float64, object]
    _columns = array_data.TimePoseSequence._columns + _columns
    _dtypes = array_data.TimePoseSequence._dtypes + _dtypes

    def __init__(self, log_id, coordinate="map"):

        if coordinate not in ["map", "ego"]:
            raise ValueError(f"Coordinate {coordinate} not supported")

        check_log_id(log_id)

        path = get_root_from_log_id(log_id)
        path = os.path.join(path, "annotations.feather")

        if not os.path.exists(path):
            raise ValueError(f"Path does not exist: {path}")

        # fmt: off
        columns = [
            "timestamp_ns", "category",
            "tx_m", "ty_m", "tz_m",
            "qx", "qy", "qz", "qw",
            "width_m", "length_m", "height_m",
        ]
        # fmt: on

        annotations = feather.read_feather(path, columns=columns)

        # allocate N rows for indexing annotations
        n = len(annotations)
        super().__init__(n)

        self._path = path
        self._coordinate = coordinate

        # array_data.TimePoseSequence
        self.timestamps = annotations["timestamp_ns"]
        self.xyz = np.array(annotations[["tx_m", "ty_m", "tz_m"]])
        self.quaternion = np.array(annotations[["qx", "qy", "qz", "qw"]])

        # Assigned annotations specific data columns
        self._data["width"] = annotations["width_m"]
        self._data["length"] = annotations["length_m"]
        self._data["height"] = annotations["height_m"]
        self._data["category"] = annotations["category"]

    def __repr__(self):
        N = len(self)
        coor = self._coordinate
        msg = "<Annotations contains %d objects on %s>" % (N, coor)
        return msg

    def __getitem__(self, key):

        other = super().__getitem__(key)
        other._path = self._path
        other._coordinate = self._coordinate
        return other

    def __getstate__(self):
        state = super().__getstate__()
        state["log_id"] = self.log_id
        state["coordinate"] = self._coordinate
        return state

    def __setstate__(self, state):
        super().__setstate__(state)

        log_id = state["log_id"]
        check_log_id(log_id)

        path = get_root_from_log_id(log_id)
        path = os.path.join(path, "annotations.feather")

        self._path = path
        self._coordinate = state["coordinate"]

    def __sub__(self, other):

        if not isinstance(other, Annotations):
            raise ValueError(f"Cannot subtract {type(other)} from Annotations")

        if self._path != other._path:
            raise ValueError("Paths do not match")

        # find the points that are not in the other
        index = np.setdiff1d(self.index, other.index, assume_unique=True)
        I = np.searchsorted(self.index, index)
        I = I[self.index[I] == index]
        return self[I]

    @property
    def category(self):
        return self._data["category"].to_numpy()

    @property
    def length(self):
        return self._data["length"].to_numpy()

    @property
    def width(self):
        return self._data["width"].to_numpy()

    @property
    def height(self):
        return self._data["height"].to_numpy()

    @property
    def bounding_centers(self):

        if hasattr(self, "_bounding_centers"):
            return self._bounding_centers

        self._bounding_centers = self.xyz
        if self._coordinate == "ego":
            return self._bounding_centers

        poses = get_poses_by_log_id(self.log_id)

        T = poses["timestamp"]
        I = np.searchsorted(T, self.timestamps)
        assert np.all(T[I] == self.timestamps)

        t = poses["xyz"][I]  # (n, 3)
        R = utils.Q_to_R(poses["q"][I])  # (n, 3, 3)

        xyz = self._bounding_centers[:, None, :]  # (n, 1, 3)
        xyz = np.sum(xyz * R, axis=-1) + t  # (n, 3)
        self._bounding_centers = xyz

        return self._bounding_centers

    def bounding_box_vertices(self, margin=0.0):

        # (n, 8, 3)
        vertices = visualization_pptk.make_bounding_box_vertices(
            self.length + margin, self.width + margin, self.height + margin
        )

        assert vertices.shape == (len(self), 8, 3)

        # transform to ego vehicle frame
        # (n, 8, 1, 3) * (n, 1, 3, 3) -> sum -> (n, 8, 3)
        # (n, 8, 3) + (n, 1, 3) -> (n, 8, 3)
        R = self.R[:, None, ...]
        xyz = self.xyz[:, None, :]
        vertices = np.sum(vertices[..., None, :] * R, axis=-1) + xyz

        if self._coordinate == "ego":
            return vertices

        poses = get_poses_by_log_id(self.log_id)

        T = poses["timestamp"]
        I = np.searchsorted(T, self.timestamps)
        assert np.all(T[I] == self.timestamps)

        t = poses["xyz"][I][:, None, :]  # (n, 1, 3)
        R = utils.Q_to_R(poses["q"][I])[:, None, ...]  # (n, 1, 3, 3)

        # transform to map frame
        # (n, 8, 1, 3) * (n, 1, 3, 3) -> sum -> (n, 8, 3)
        # (n, 8, 3) + (n, 1, 3) -> (n, 8, 3)
        vertices = np.sum(vertices[..., None, :] * R, axis=-1) + t
        assert vertices.shape == (len(self), 8, 3)

        return vertices

    def filter_by_category(self, category):
        I = np.isin(self.category, category)
        return self[I]

    def _kdtree_distance_filtering(self, points, margin=0.0):

        tree = scipy.spatial.KDTree(points)

        rs = []
        splits = np.r_[np.arange(0, len(self), 1024), len(self)]

        # chunkify
        for i, j in zip(splits[:-1], splits[1:]):

            # subset of self
            a = self[i:j]
            H = a.height + margin
            L = a.length + margin
            W = a.width + margin

            D = np.sqrt(H**2 + L**2 + W**2) / 2.0

            assert len(D) == len(a.bounding_centers)

            r = tree.query_ball_point(
                a.bounding_centers,
                D,
                workers=-1,
                return_sorted=True,
            )

            rs.extend(r)

        return rs

    def is_points_in_bounding_boxes(self, points, margin=0.0, separate=False):

        _results = self._kdtree_distance_filtering(points, margin=margin)

        results = []
        for r, v in zip(_results, self.bounding_box_vertices(margin=margin)):

            if len(r) == 0:
                continue

            r = np.array(r).astype(np.int64)

            I = utils.points_in_a_bounding_box(points[r], v)
            results.append(r[I])

        if len(results) == 0:
            return np.array([], dtype=np.int64)

        if separate:
            return results

        results = np.hstack(results)
        return np.unique(results)
