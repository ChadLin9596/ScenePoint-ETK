import copy
import os
import numpy as np
import pyarrow.feather as feather

import py_utils.array_data as array_data
import py_utils.utils as utils
import py_utils.voxel_grid as voxel_grid

from .av2_basic_tools import (
    check_log_id,
    list_sweep_files_by_log_id,
    get_root_from_log_id,
    get_poses_by_log_id,
    ArgoMixin,
)
from .av2_annotations import Annotations
from .av2_image_sequence import CameraSequence

CLOUD_COMPARE_DTYPE = [
    ("x", np.float32),
    ("y", np.float32),
    ("z", np.float32),
    ("rgb", np.float32),
    ("center", np.float64, 3),
    ("count", np.int64),
    ("intensity", np.uint8),
]


class Sweep(ArgoMixin, array_data.Array):

    def __init__(self, path, coordinate="map"):

        if coordinate not in ["map", "ego"]:
            raise ValueError(f"Coordinate {coordinate} not supported")

        path = os.path.realpath(path)
        if not os.path.exists(path):
            raise ValueError(f"Path does not exist: {path}")

        # use intensity field (uint8) as
        N = len(feather.read_feather(path, columns=["intensity"]))
        super().__init__(N)  # allocate N rows for indexing points

        self._path = path
        self._coordinate = coordinate
        self._is_sampled = False

    def __repr__(self):

        N = len(self)
        coor = self._coordinate
        return "<Sweep contains %d points on %s coordinate>" % (N, coor)

    def __getitem__(self, key):

        other = super().__getitem__(key)
        other._is_sampled = True
        other._coordinate = self._coordinate
        return other

    def __copy__(self):

        c = self.__class__
        other = c.__new__(c)
        other._path = self._path
        other._coordinate = self._coordinate
        other._is_sampled = self._is_sampled

        return other

    def __getstate__(self):

        state = {
            "log_id": self.log_id,
            "sweep_timestamp": self.sweep_timestamp,
            "coordinate": self._coordinate,
            "is_sampled": self._is_sampled,
        }

        if not self._is_sampled:
            return state

        state.update(super().__getstate__())
        state["index"] = state["index"].astype(np.uint32)
        return state

    def __setstate__(self, state):

        log_id = state["log_id"]
        timestamps = str(state["sweep_timestamp"])

        check_log_id(log_id)

        path = get_root_from_log_id(log_id)
        path = os.path.join(path, "sensors", "lidar", timestamps + ".feather")

        self.__init__(path, coordinate=state["coordinate"])
        self._is_sampled = state["is_sampled"]

        if self._is_sampled:
            # re-sampled
            self._allocate(len(state["index"]))
            self._data.index = state["index"]
            self._data["index"] = state["index"]

    def __sub__(self, other):

        if not isinstance(other, Sweep):
            raise ValueError(f"Cannot subtract {type(other)} from Sweep")

        if self._path != other._path:
            raise ValueError("Paths do not match")

        # find the points that are not in the other
        index = np.setdiff1d(self.index, other.index, assume_unique=True)
        I = np.searchsorted(self.index, index)
        I = I[self.index[I] == index]
        return self[I]

    @property
    def xyz(self):

        if hasattr(self, "_xyz"):
            return self._xyz

        xyz = feather.read_feather(self._path, columns=["x", "y", "z"])
        self._xyz = xyz.to_numpy()[self.index]
        if self._coordinate == "ego":
            return self._xyz

        poses = get_poses_by_log_id(self.log_id)

        I = np.searchsorted(poses["timestamp"], self.sweep_timestamp)
        position = poses["xyz"][I]  # (3,)
        quaternion = poses["q"][I]  # (4,)
        R = utils.Q_to_R(quaternion)  # (3, 3)

        # it is equivalent to
        # np.sum(self._xyz[..., None, :] * R, axis=-1) + position
        # but more efficient
        self._xyz = self._xyz @ R.T + position

        return self._xyz

    @property
    def intensity(self):

        if hasattr(self, "_intensity"):
            return self._intensity

        intensity = feather.read_feather(self._path, columns=["intensity"])
        self._intensity = intensity.to_numpy().flatten()[self.index]
        return self._intensity

    @property
    def laser_number(self):

        if hasattr(self, "_laser_number"):
            return self._laser_number

        laser_number = feather.read_feather(
            self._path, columns=["laser_number"]
        )
        self._laser_number = laser_number.to_numpy().flatten()[self.index]
        return self._laser_number

    @property
    def point_timestamps(self):

        if hasattr(self, "_point_timestamps"):
            return self._point_timestamps

        point_timestamps = feather.read_feather(
            self._path, columns=["offset_ns"]
        )
        point_timestamps = point_timestamps.to_numpy().flatten()
        self._point_timestamps = point_timestamps[self.index]
        return self._point_timestamps

    @property
    def sweep_timestamp(self):

        name = os.path.basename(self._path)
        return int(name.split(".")[0])

    def infer_color_by_camera_sequence(self, camera_sequence):

        if not isinstance(camera_sequence, CameraSequence):
            msg = f"Expect CameraSequence, got {type(camera_sequence)}"
            raise ValueError(msg)

        if not camera_sequence.log_id == self.log_id:
            raise ValueError("Log IDs do not match")

        timestamp = self.sweep_timestamp
        return camera_sequence.colour_single_sweep(timestamp, self.xyz)


class SweepSequence(ArgoMixin, array_data.Array):

    def __init__(self, log_id, coordinate="map"):

        check_log_id(log_id)

        path = get_root_from_log_id(log_id)
        sweep_files = list_sweep_files_by_log_id(log_id)

        super().__init__(len(sweep_files))

        self._path = path
        self.sweeps = [Sweep(i, coordinate) for i in sweep_files]
        self.coordinate = coordinate

    def __repr__(self):

        N = len(self)
        return f"<SweepSequence contains {N} sweeps for log {self.log_id}>"

    def __sub__(self, other):

        if not isinstance(other, SweepSequence):
            raise ValueError(
                f"Cannot subtract {type(other)} from SweepSequence"
            )

        if self._path != other._path:
            raise ValueError("Paths do not match")

        if len(self.sweeps) != len(other.sweeps):
            raise ValueError("Cannot subtract sequences of different lengths")

        sweeps = []
        for i, j in zip(self.sweeps, other.sweeps):
            if i._path != j._path:
                raise ValueError("Sweep paths do not match")

            # subtract the sweeps
            sweep = i - j
            sweeps.append(sweep)

        other = copy.copy(self)
        other._path = self._path
        other.sweeps = sweeps
        other.coordinate = self.coordinate

        return other

    @staticmethod
    def from_sweeps(sweeps):

        if not isinstance(sweeps, list):
            sweeps = [sweeps]

        for i, s in enumerate(sweeps):
            if not isinstance(s, Sweep):
                raise ValueError(f"Expect Sweep, got {type(s)} at index {i}")

        log_id = sweeps[0].log_id
        coordinate = sweeps[0]._coordinate

        seq = SweepSequence(log_id=log_id, coordinate=coordinate)
        seq.sweeps = sweeps
        return seq

    def filtered_by_annotations(
        self,
        annotations,
        bbox_margin=0.5,
        time_margin=1.05,
        verbose=True,
        prefix="",
    ):

        if isinstance(annotations, Annotations):
            annotations = [annotations]

        for a in annotations:
            if isinstance(a, Annotations):
                continue
            raise ValueError(f"Expect Annotations, got {type(a)}")

        # convert seconds to nanoseconds
        time_margin = time_margin * 1e9

        prog = utils.ProgressTimer(prefix=prefix, verbose=verbose)
        prog.tic(sum([len(s) for s in self.sweeps]))

        sweeps = []
        for sweep in self.sweeps:

            results = np.zeros(len(sweep), dtype=bool)
            for annot in annotations:

                # filter by timestamp
                T_annot = annot.timestamps
                T_sweep = sweep.sweep_timestamp
                T_marg = np.abs(time_margin)

                I = np.abs(T_annot - T_sweep) <= T_marg
                annot = annot[I]
                if len(annot) == 0:
                    continue

                # indices of sweep.xyz
                I = annot.is_points_in_bounding_boxes(
                    sweep.xyz, margin=bbox_margin
                )
                results[I] = True

            sweeps.append(sweep[results])
            prog.toc(len(sweep))

        other = copy.copy(self)
        other._path = self._path
        other.sweeps = sweeps
        other.coordinate = self.coordinate
        return other

    @property
    def xyz(self):
        return np.vstack([s.xyz for s in self.sweeps])

    @property
    def intensity(self):
        return np.hstack([s.intensity for s in self.sweeps])

    @property
    def laser_number(self):
        return np.hstack([s.laser_number for s in self.sweeps])

    @property
    def point_timestamps(self):
        return np.hstack([s.point_timestamps for s in self.sweeps])

    @property
    def sweep_timestamp(self):
        return [s.sweep_timestamp for s in self.sweeps]

    def export_to_voxel_grid(self, voxel_size=0.2, return_details=False):

        vg = voxel_grid.VoxelGrid(
            self.xyz,
            voxel_size=voxel_size,
            attributes=[self.intensity],
        )

        details = {
            "splits": vg._splits,
            "indices": vg._to_point_indices,
            "voxel_index": vg._sorted_voxel_index,
        }

        X = np.zeros(len(vg), dtype=np.dtype(CLOUD_COMPARE_DTYPE))
        X["x"] = vg.voxel_centroids[:, 0]
        X["y"] = vg.voxel_centroids[:, 1]
        X["z"] = vg.voxel_centroids[:, 2]
        X["center"] = vg.voxel_centers
        X["count"] = vg.voxel_counts
        X["intensity"] = vg.voxel_attributes[0]

        if return_details:
            return X, details
        return X

    @property
    def rgb(self):

        if hasattr(self, "_rgb"):
            return self._rgb

        times = self.sweep_timestamp
        camera_sequence = CameraSequence(self.log_id).align_timestamps(times)

        prog = utils.ProgressTimer(prefix="Inferring RGB")
        prog.tic(len(self.sweeps))
        rgbs = []
        for sweep in self.sweeps:
            rgb = sweep.infer_color_by_camera_sequence(camera_sequence)
            rgbs.append(rgb)
            prog.toc()

        self._rgb = np.vstack(rgbs)
        return self._rgb
