import copy
import glob
import hashlib
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import py_utils.array_data as array_data
import py_utils.utils as utils
import py_utils.utils_img as utils_img

from .av2_basic_tools import (
    check_log_id,
    list_cameras_by_log_id,
    get_root_from_log_id,
    get_poses_by_log_id,
    get_extrinsic_by_log_id,
    get_intrinsic_by_log_id,
    ArgoMixin,
)


class ImageSequence(ArgoMixin, array_data.TimePoseSequence):

    def __init__(self, log_id, camera="ring_front_center"):

        check_log_id(log_id)

        if camera not in list_cameras_by_log_id(log_id):
            raise ValueError(f"Camera {camera} not supported")

        root = get_root_from_log_id(log_id)
        path = os.path.join(root, "sensors", "cameras", camera, "*.jpg")

        files = glob.glob(path)
        files = sorted(files)

        timestamps = [os.path.basename(i) for i in files]
        timestamps = [int(i.replace(".jpg", "")) for i in timestamps]

        # member variables for ImageSequence
        self._path = root
        self.camera = camera
        self._files = files
        self._transforms = []

        # initialize the array_data.TimePoseSequence
        n = len(files)
        super().__init__(n)

        # assign timestamps, poses, quaternions for array_data.TimePoseSequence
        poses = get_poses_by_log_id(log_id)

        T = poses["timestamp"]
        I = np.searchsorted(T, timestamps)
        assert np.all(T[I] == timestamps)

        # member variables for array_data.TimePoseSequence
        self.timestamps = timestamps
        self.xyz = poses["xyz"][I]
        self.quaternion = poses["q"][I]
        # self.is_reshaped = False

    def __reprtransforms__(self):
        msg = ""
        for transform, params in self._transforms:

            if transform == "resize":
                msg += f"\n\tresized to {params['H']}x{params['W']}"

            elif transform == "crop":
                msg += f"\n\tcropped to {params['height']}x{params['width']}"
                msg += f" from top {params['top']} left {params['left']}"

            else:
                raise ValueError(f"Unknown transform {transform}")

        return msg

    def __repr__(self):
        N = len(self._data)
        H, W = self.figsize
        msg = f"<ImageSequences contains {N} {H}x{W} images>"
        msg += f" ({self.camera})"

        # explicitly show transforms if any
        msg += self.__reprtransforms__()

        return msg

    def __getitem__(self, key):

        #  `index`, `timestamps`, `xyz,` `quaternion`
        other = super().__getitem__(key)

        other._path = self._path
        other.camera = self.camera
        other._files = np.array(self._files)[key].tolist()
        other._transforms = self._transforms.copy()

        return other

    def __getstate__(self):

        state = {
            "log_id": self.log_id,
            "camera": self.camera,
            "filenames": [os.path.basename(i) for i in self._files],
            "transforms": self._transforms,
        }
        state.update(super().__getstate__())

        return state

    def __setstate__(self, state):

        log_id = state["log_id"]
        camera = state["camera"]
        filenames = state["filenames"]
        transforms = state.get("transforms", [])

        check_log_id(log_id)

        super().__setstate__(state)

        path = get_root_from_log_id(log_id)
        fs = []
        for i in filenames:
            f = os.path.join(path, "sensors", "cameras", camera, i)
            if not os.path.exists(f):
                raise FileNotFoundError(f"File {f} not found")
            fs.append(f)

        self._path = path
        self.camera = camera
        self._files = fs
        self._transforms = transforms

    def get_an_image(self, index):
        """get an image as a HxWx3 numpy array by an integer index."""

        img = plt.imread(self._files[index])

        for transform, params in self._transforms:

            if transform == "resize":
                H = params["H"]
                W = params["W"]
                mode = Image.Resampling.BICUBIC
                img = Image.fromarray(img).resize((W, H), mode)
                img = np.array(img)

            elif transform == "crop":
                top = params["top"]
                left = params["left"]
                height = params["height"]
                width = params["width"]
                img = img[top : top + height, left : left + width, :]

        return img

    @property
    def figsize(self):
        """Return the figure size as (height, width)."""

        if hasattr(self, "_figsize"):
            return self._figsize

        X = get_intrinsic_by_log_id(self.log_id)
        index = np.searchsorted(X["sensor_name"].to_numpy(), self.camera)
        figsize = X[["height_px", "width_px"]].to_numpy()[index]

        if len(self._transforms) == 0:
            self._figsize = figsize
            return self._figsize

        method, params = self._transforms[-1]
        if method == "resize":
            H = params["H"]
            W = params["W"]
            figsize = np.r_[H, W]

        elif method == "crop":
            height = params["height"]
            width = params["width"]
            figsize = np.r_[height, width]

        else:
            raise ValueError(f"Unknown transform {method}")

        self._figsize = figsize
        return self._figsize

    @property
    def intrinsic(self):
        """Return the intrinsic matrix as a 3x3 numpy array."""

        if hasattr(self, "_intrinsic"):
            return self._intrinsic

        X = get_intrinsic_by_log_id(self.log_id)
        index = np.searchsorted(X["sensor_name"].to_numpy(), self.camera)
        focal_length = X[["fx_px", "fy_px"]].to_numpy()[index]
        principal_point = X[["cx_px", "cy_px"]].to_numpy()[index]
        figsize = X[["height_px", "width_px"]].to_numpy()[index]

        # fmt: off
        self._intrinsic = np.array([
            [focal_length[0],              0., principal_point[0]],
            [             0., focal_length[1], principal_point[1]],
            [             0.,              0.,                 1.],
        ])
        # fmt: on

        if len(self._transforms) == 0:
            return self._intrinsic

        for method, params in self._transforms:

            if method == "resize":
                H = params["H"]
                W = params["W"]

                self._intrinsic = utils_img.update_intrinsics_by_resized(
                    self._intrinsic,
                    figsize,
                    (H, W),
                )
                figsize = (H, W)

            elif method == "crop":
                top = params["top"]
                left = params["left"]

                self._intrinsic = utils_img.update_intrinsics_by_crop(
                    self._intrinsic,
                    top,
                    left,
                )
                figsize = (params["height"], params["width"])

            else:
                raise ValueError(f"Unknown transform {method}")

        return self._intrinsic

    @property
    def extrinsic_to_ego(self):

        if hasattr(self, "_extrinsic_to_ego"):
            return self._extrinsic_to_ego

        X = get_extrinsic_by_log_id(self.log_id)
        index = np.searchsorted(X["sensor_name"].to_numpy(), self.camera)
        T = X[["tx_m", "ty_m", "tz_m"]].to_numpy()[index]
        Q = X[["qx", "qy", "qz", "qw"]].to_numpy()[index]

        R = utils.Q_to_R(Q)
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = T

        self._extrinsic_to_ego = extrinsic
        return self._extrinsic_to_ego

    @property
    def extrinsic(self):
        """Return the extrinsic matrices as a Nx4x4 numpy array."""

        if hasattr(self, "_extrinsic"):
            return self._extrinsic

        extrinsic = self.transformation @ self.extrinsic_to_ego[None, ...]
        self._extrinsic = extrinsic
        return self._extrinsic

    @property
    def unique_id(self):
        """Return a unique ID for the image sequence."""
        uid = pickle.dumps(self)
        uid = hashlib.sha256(uid).hexdigest()
        uid = uid[:16]
        return uid

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
        """Return a resized copy of the image sequence with new height H and width W."""

        H, W = int(H), int(W)

        other = copy.copy(self)
        other._transforms = []
        other._transforms.extend(self._transforms)
        other._transforms.append(("resize", {"H": H, "W": W}))

        return other

    def crop(self, top, left, height, width):
        """Return a cropped copy of the image sequence."""

        top = int(top)
        left = int(left)
        height = int(height)
        width = int(width)

        other = copy.copy(self)
        other._transforms = []
        other._transforms.extend(self._transforms)
        other._transforms.append(
            (
                "crop",
                {"top": top, "left": left, "height": height, "width": width},
            ),
        )

        return other

    def center_crop(self, crop_height, crop_width):
        """Return a center-cropped copy of the image sequence."""

        crop_height = int(crop_height)
        crop_width = int(crop_width)

        H, W = self.figsize
        top = (H - crop_height) // 2
        left = (W - crop_width) // 2

        return self.crop(top, left, crop_height, crop_width)

    def get_a_depth_map(
        self,
        index,
        points,
        invalid_value=-1,
        min_distance=0.0,
        max_distance=np.inf,
        return_details=False,
    ):
        """
        Get a depth map as a HxW numpy array by an integer index and 3D points.

        Args:

        - index (int):
            - index of the image in the sequence.

        - points (np.ndarray):
            - Nx3 array of 3D points in world coordinates.

        - invalid_value (float | nan):
            - value to assign to pixels where no points project.

        - min_distance (float):
            - minimum distance from the camera to consider a point valid.

        - max_distance (float):
            - maximum distance from the camera to consider a point valid.

        - return_details (bool):
            - whether to return additional details as a dictionary.

        Returns:

        - depth_map (np.ndarray):
            - HxW array representing the depth map.
        - details (dict, optional):
            - additional details if return_details is True.
        """

        extrinsic = self.extrinsic[index]
        intrinsic = self.intrinsic

        depth_map, point_map, details = utils_img.points_to_depth_image(
            points,
            intrinsic,
            extrinsic,
            *self.figsize,
            invalid_value=invalid_value,
            min_distance=min_distance,
            max_distance=max_distance,
            return_details=True,
            other_attrs=[points],
        )

        details["point_map"] = point_map

        if return_details:
            return depth_map, details
        return depth_map

    def get_a_point_map(
        self,
        index,
        points,
        invalid_value=-1,
        min_distance=0.0,
        max_distance=np.inf,
        return_details=False,
    ):

        _, details = self.get_a_depth_map(
            index,
            points,
            invalid_value=invalid_value,
            min_distance=min_distance,
            max_distance=max_distance,
            return_details=True,
        )

        point_map = details.pop("point_map", None)
        if return_details:
            return point_map, details
        return point_map

    def get_a_point_index_map(
        self,
        index,
        points,
        invalid_value=-1,
        min_distance=0.0,
        max_distance=np.inf,
        return_details=False,
    ):

        _, details = self.get_a_depth_map(
            index,
            points,
            invalid_value=invalid_value,
            min_distance=min_distance,
            max_distance=max_distance,
            return_details=True,
        )

        details.pop("point_map", None)

        u, v = details["uv"]
        FOV_mask = details["FOV_mask"]
        indices_map = np.full(self.figsize, -1, dtype=np.int32)
        indices_map[u, v] = np.arange(len(points))[FOV_mask]

        if return_details:
            return indices_map, details
        return indices_map


class CameraSequence(ArgoMixin, array_data.Array):

    def __init__(
        self,
        log_id,
        cameras=[
            "ring_front_center",
            "ring_front_left",
            "ring_front_right",
            "ring_rear_left",
            "ring_rear_right",
            "ring_side_left",
            "ring_side_right",
            "stereo_front_left",
            "stereo_front_right",
        ],
    ):

        check_log_id(log_id)

        if isinstance(cameras, str):
            cameras = [cameras]

        # guarantee that all cameras are valid
        _cameras = []
        for camera in cameras:
            valid_cameras = list_cameras_by_log_id(log_id)
            if camera not in valid_cameras:
                continue
            _cameras.append(camera)
        cameras = _cameras

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

    def list_camera_unique_ids(self):
        return [i.unique_id for i in self.cameras]

    def list_image_sequences(self):
        return self.cameras.copy()

    def get_a_camera(self, index_or_unique_id_or_camera_name):

        if isinstance(index_or_unique_id_or_camera_name, int):
            return self.cameras[index_or_unique_id_or_camera_name]

        for camera in self.cameras:
            if camera.camera == index_or_unique_id_or_camera_name:
                return camera
            if camera.unique_id == index_or_unique_id_or_camera_name:
                return camera

        msg = f"Camera {index_or_unique_id_or_camera_name} not found"
        raise ValueError(msg)

    def set_a_camera(self, camera):
        if not isinstance(camera, ImageSequence):
            raise ValueError("camera must be an ImageSequence")

        for i in range(len(self.cameras)):
            if self.cameras[i].unique_id == camera.unique_id:
                self.cameras[i] = camera
                return

        msg = f"Camera {camera.camera}, {camera.unique_id} not found"
        raise ValueError(msg)

    def append_a_camera(self, camera):
        if not isinstance(camera, ImageSequence):
            raise ValueError("camera must be an ImageSequence")

        self.cameras.append(camera)

        # previously stored index
        index = self.index

        # re-allocate with new length
        self._allocate(len(self.cameras))

        # restore previous index and add new one
        self._data.loc[: len(index) - 1, "index"] = index
        self._data.loc[len(index), "index"] = np.max(index) + 1

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

    def crop(self, top, left, height, width):
        cameras = []
        for camera in self.cameras:
            camera = camera.crop(top, left, height, width)
            cameras.append(camera)

        other = copy.copy(self)
        other.cameras = cameras
        return other

    def center_crop(self, crop_height, crop_width):
        cameras = []
        for camera in self.cameras:
            camera = camera.center_crop(crop_height, crop_width)
            cameras.append(camera)

        other = copy.copy(self)
        other.cameras = cameras
        return other

    def align_timestamps(self, timestamps):
        """
        Return a copy of the CameraSequence with cameras aligned to the
        given timestamps.

        TODO: consider case that lengths of timestamps are different.
        """
        cameras = []
        for camera in self.cameras:
            camera = camera.align_timestamps(timestamps)
            cameras.append(camera)

        other = copy.copy(self)
        other.cameras = cameras
        return other

    def colour_single_sweep(self, sweep_timestamp, points):

        rgb = np.ones_like(points, dtype=np.float32)

        for camera in self.cameras:

            index = np.argmin(np.abs(camera.timestamps - sweep_timestamp))
            index_map = utils_img.points_to_index_map(
                points,
                camera.intrinsic,
                camera.extrinsic[index],
                *camera.figsize,
            )
            image = camera.get_an_image(index)

            mask = index_map >= 0
            rgb[index_map[mask]] = image[mask]

        return rgb
