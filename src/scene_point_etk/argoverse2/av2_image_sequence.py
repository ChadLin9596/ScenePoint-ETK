import copy
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import py_utils.array_data as array_data
import py_utils.utils as utils
import py_utils.utils_img as utils_img

from .av2_basic_tools import (
    check_log_id,
    list_cameras,
    get_root_from_log_id,
    get_poses_by_log_id,
    get_extrinsic_by_log_id,
    get_intrinsic_by_log_id,
    ArgoMixin,
)


class ImageSequence(ArgoMixin, array_data.TimePoseSequence):

    def __init__(self, log_id, camera="ring_front_center"):

        check_log_id(log_id)

        if camera not in list_cameras():
            raise ValueError(f"Camera {camera} not supported")

        root = get_root_from_log_id(log_id)
        path = os.path.join(root, "sensors", "cameras", camera, "*.jpg")

        files = glob.glob(path)
        files = sorted(files)

        timestamps = [os.path.basename(i) for i in files]
        timestamps = [int(i.replace(".jpg", "")) for i in timestamps]

        self._path = root
        self.camera = camera
        self._files = files

        # initialize the array_data.TimePoseSequence
        n = len(files)
        super().__init__(n)

        # assign timestamps, poses, quaternions for array_data.TimePoseSequence
        poses = get_poses_by_log_id(log_id)

        T = poses["timestamp"]
        I = np.searchsorted(T, timestamps)
        assert np.all(T[I] == timestamps)

        self.timestamps = timestamps
        self.xyz = poses["xyz"][I]
        self.quaternion = poses["q"][I]
        self.is_reshaped = False

    def __repr__(self):
        N = len(self._data)
        H, W = self.figsize
        msg = f"<ImageSequences contains {N} {H}x{W} images>"
        msg += f" ({self.camera})"

        return msg

    def __getitem__(self, key):

        other = super().__getitem__(key)
        other._path = self._path
        other.camera = self.camera
        other._files = np.array(self._files)[key].tolist()

        return other

    def __getstate__(self):

        state = {
            "log_id": self.log_id,
            "camera": self.camera,
            "filenames": [os.path.basename(i) for i in self._files],
            "is_reshaped": self.is_reshaped,
        }
        state.update(super().__getstate__())

        if self.is_reshaped:
            state["figsize"] = self.figsize

        return state

    def __setstate__(self, state):

        log_id = state["log_id"]
        camera = state["camera"]
        filenames = state["filenames"]
        is_reshaped = state.get("is_reshaped", False)

        check_log_id(log_id)

        super().__setstate__(state)

        path = get_root_from_log_id(log_id)
        fs = []
        for i in filenames:
            f = os.path.join(path, "sensors", "cameras", camera, i)
            fs.append(f)

        self._path = path
        self.camera = camera
        self._files = fs
        self.is_reshaped = is_reshaped

        if is_reshaped:
            self.is_reshaped = state["is_reshaped"]
            self._figsize = state["figsize"]

    def get_an_image(self, index):

        img = plt.imread(self._files[index])
        if self.is_reshaped:
            H, W = self.figsize
            img = Image.fromarray(img).resize((W, H), Image.Resampling.BICUBIC)
            img = np.array(img)

        return img

    @property
    def figsize(self):

        if hasattr(self, "_figsize"):
            return self._figsize

        X = get_intrinsic_by_log_id(self.log_id)
        index = np.searchsorted(X["sensor_name"].to_numpy(), self.camera)
        figsize = X[["height_px", "width_px"]].to_numpy()[index]

        self._figsize = figsize
        return self._figsize

    @property
    def intrinsic(self):

        if hasattr(self, "_intrinsic"):
            return self._intrinsic

        X = get_intrinsic_by_log_id(self.log_id)
        index = np.searchsorted(X["sensor_name"].to_numpy(), self.camera)
        focal_length = X[["fx_px", "fy_px"]].to_numpy()[index]
        principal_point = X[["cx_px", "cy_px"]].to_numpy()[index]

        # fmt: off
        self._intrinsic = np.array([
            [focal_length[0],              0., principal_point[0]],
            [             0., focal_length[1], principal_point[1]],
            [             0.,              0.,                 1.],
        ])
        # fmt: on

        if self.is_reshaped:

            original_figsize = X[["height_px", "width_px"]].to_numpy()[index]

            self._intrinsic = utils_img.update_intrinsics_by_resized(
                self._intrinsic,
                original_figsize,
                self._figsize,  # should already be set
            )

        return self._intrinsic

    @property
    def extrinsic_to_ego(self):

        if hasattr(self, "_extrinsic"):
            return self._extrinsic

        X = get_extrinsic_by_log_id(self.log_id)
        index = np.searchsorted(X["sensor_name"].to_numpy(), self.camera)
        T = X[["tx_m", "ty_m", "tz_m"]].to_numpy()[index]
        Q = X[["qx", "qy", "qz", "qw"]].to_numpy()[index]

        R = utils.Q_to_R(Q)
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = T

        self._extrinsic = extrinsic
        return self._extrinsic

    @property
    def extrinsic(self):
        return self.transformation @ self.extrinsic_to_ego[None, ...]

    def filter_by_image_viewing(
        self,
        map_points,
        max_distance=50,
        min_distance=None,
        min_pts_num_in_FOV=80,
        verbose=True,
    ):

        results = np.zeros(len(self), dtype=bool)

        prog = utils.ProgressTimer(verbose=verbose)
        prog.tic(len(self))

        for n, extrinsic in enumerate(self.extrinsic):

            xyz = utils_img._trans_from_world_to_camera(map_points, extrinsic)

            mask = utils_img.is_camera_points_in_FOV(
                xyz,
                self.intrinsic,
                *self.figsize,
                max_distance=max_distance,
                min_distance=min_distance,
            )

            pts_num_in_FOV = np.sum(mask)
            if pts_num_in_FOV >= min_pts_num_in_FOV:
                results[n] = 1

            prog.toc()

        return self[results]

    def FOV_mask(
        self,
        map_points,
        max_distance=50,
        min_distance=None,
        verbose=True,
    ):

        result = np.zeros(len(map_points), dtype=bool)

        prog = utils.ProgressTimer(verbose=verbose)
        prog.tic(len(self))

        for extrinsic in self.extrinsic:

            xyz = utils_img._trans_from_world_to_camera(map_points, extrinsic)

            mask = utils_img.is_camera_points_in_FOV(
                xyz,
                self.intrinsic,
                *self.figsize,
                max_distance=max_distance,
                min_distance=min_distance,
            )

            result |= mask

            prog.toc()

        return result

    def resize(self, H, W):

        other = copy.copy(self)

        other._figsize = np.r_[H, W]
        other.is_reshaped = True
        other._intrinsic = utils_img.update_intrinsics_by_resized(
            self.intrinsic,
            self.figsize,
            (H, W),
        )
        return other


class CameraSequence(ArgoMixin, array_data.Array):

    def __init__(self, log_id, cameras=list_cameras()):

        check_log_id(log_id)

        if not np.iterable(cameras):
            cameras = [cameras]

        for camera in cameras:
            if camera not in list_cameras():
                raise ValueError(f"Camera {camera} not supported")

        super().__init__(len(cameras))

        path = get_root_from_log_id(log_id)
        self._path = path

        _cameras = []
        for camera in cameras:
            camera = ImageSequence(log_id, camera)
            _cameras.append(camera)
        self.cameras = _cameras

    def __repr__(self):
        msg = "<CameraSequence contains: %d cameras>" % len(self.cameras)
        for i in self.cameras:
            msg += "\n\t" + str(i)
        return msg

    def __getitem__(self, key):

        other = super().__getitem__(key)
        other._path = self._path

        cameras = []

        for i in np.r_[np.arange(len(self.cameras))[key]]:
            camera = self.cameras[i]
            cameras.append(camera)
        other.cameras = cameras

        return other

    def __getstate__(self):

        state = {
            "log_id": self.log_id,
            "cameras": [i.__getstate__() for i in self.cameras],
        }
        state.update(super().__getstate__())

        return state

    def __setstate__(self, state):

        log_id = state["log_id"]
        check_log_id(log_id)
        self._path = get_root_from_log_id(log_id)

        self.cameras = []
        for camera_state in state["cameras"]:
            camera = camera_state["camera"]
            camera = ImageSequence(log_id, camera)
            camera.__setstate__(camera_state)
            self.cameras.append(camera)

        super().__setstate__(state)

    def list_cameras(self):
        return [i.camera for i in self.cameras]

    def get_a_camera(self, index_or_camera_name):

        if isinstance(index_or_camera_name, int):
            return self.cameras[index_or_camera_name]

        for camera in self.cameras:
            if camera.camera == index_or_camera_name:
                return camera

        raise ValueError(f"Camera {index_or_camera_name} not found")

    def set_a_camera(self, camera):
        if not isinstance(camera, ImageSequence):
            raise ValueError("camera must be an ImageSequence")

        for i in range(len(self.cameras)):
            if self.cameras[i].camera == camera.camera:
                self.cameras[i] = camera
                return

        raise ValueError(f"Camera {camera.camera} not found")

    def filter_by_image_viewing(
        self,
        map_points,
        max_distance=50,
        min_distance=None,
        min_pts_num_in_FOV=80,
        verbose=True,
    ):

        cameras = []
        for camera in self.cameras:
            camera = camera.filter_by_image_viewing(
                map_points,
                max_distance=max_distance,
                min_distance=min_distance,
                min_pts_num_in_FOV=min_pts_num_in_FOV,
                verbose=verbose,
            )
            cameras.append(camera)

        other = copy.copy(self)
        other.cameras = cameras
        return other

    def FOV_mask(
        self,
        map_points,
        max_distance=50,
        min_distance=None,
        verbose=True,
    ):

        result = np.zeros(len(map_points), dtype=bool)
        for camera in self.cameras:
            mask = camera.FOV_mask(
                map_points,
                max_distance=max_distance,
                min_distance=min_distance,
                verbose=verbose,
            )
            result |= mask
        return result

    def resize(self, H, W):
        cameras = []
        for camera in self.cameras:
            camera = camera.resize(H, W)
            cameras.append(camera)

        other = copy.copy(self)
        other.cameras = cameras
        return other

    def align_timestamps(self, timestamps):
        cameras = []
        for camera in self.cameras:
            camera = camera.align_timestamps(timestamps)
            cameras.append(camera)

        other = copy.copy(self)
        other.cameras = cameras
        return other
