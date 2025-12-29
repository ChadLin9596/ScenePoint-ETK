import os
import glob
import hashlib
import pickle
import numpy as np
import matplotlib.pyplot as plt

from .. import argoverse2
from .. import patch_db
from .. import utils as scene_utils
from . import diff_scene

import py_utils.utils as utils
import py_utils.pcd as pcd
import py_utils.utils_img as utils_img
import py_utils.visualization_pptk as visualization_pptk
import py_utils.utils_segmentation as utils_segmentation


class ScenePCDMixin:
    """
    a mixin class for scene PCD data. To use this mixin, the child class
    must define the `scene_filepath` attribute pointing to the location
    of the `scene.pcd` file.

    directory structure::

        <root>
        ├── <Scene ID 00>
        │   └── <version name>
        │       └── scene.pcd  <- scene_filepath
        │
        ├── <Scene ID 01>
        └── ...
    """

    scene_filepath = ""

    @property
    def scene_pcd(self):
        if hasattr(self, "_scene_pcd"):
            return self._scene_pcd.copy()

        if not os.path.exists(self.scene_filepath):
            raise FileNotFoundError(f"{self.scene_filepath} does not exist")

        default_dtype = np.dtype(argoverse2.CLOUD_COMPARE_DTYPE)
        self._scene_pcd = pcd.read(self.scene_filepath)
        assert self._scene_pcd.dtype == default_dtype
        return self._scene_pcd.copy()

    @scene_pcd.setter
    def scene_pcd(self, pcd_data):
        default_dtype = np.dtype(argoverse2.CLOUD_COMPARE_DTYPE)
        if pcd_data.dtype != default_dtype:
            raise ValueError("Invalid pcd data type")

        pcd.write(self.scene_filepath, pcd_data)
        self._scene_pcd = pcd_data.copy()

    @property
    def scene_pcd_bytesize(self):
        if os.path.exists(self.scene_filepath):
            return os.path.getsize(self.scene_filepath)
        return 0

    @property
    def pcd_xyz(self):
        p = self.scene_pcd
        xyz = np.vstack([p["x"], p["y"], p["z"]]).T
        return xyz

    @property
    def pcd_intensity(self):
        p = self.scene_pcd
        return p["intensity"]

    @property
    def pcd_center(self):
        p = self.scene_pcd
        return p["center"]

    @property
    def pcd_count(self):
        p = self.scene_pcd
        return p["count"]

    @property
    def pcd_color(self):
        p = self.scene_pcd
        r, g, b, a = scene_utils.decode_rgba(p["rgb"])
        return np.vstack([r, g, b, a]).T


class SceneDetailsMixin:
    """
    a mixin class for scene details data.

    directory structure::

        <root>
        ├── <Scene ID 00>
        │   └── <version name>
        │       └── details.pkl  <- details_filepath
        │
        ├── <Scene ID 01>
        └── ...
    """

    details_filepath = ""

    @property
    def scene_details(self):
        if hasattr(self, "_scene_details"):
            return self._scene_details

        if not os.path.exists(self.details_filepath):
            return None

        with open(self.details_filepath, "rb") as fd:
            self._scene_details = pickle.load(fd)

        return self._scene_details

    @scene_details.setter
    def scene_details(self, details):
        with open(self.details_filepath, "wb") as fd:
            pickle.dump(details, fd)

    @property
    def scene_details_bytesize(self):
        """
        num of bytes of `details.pkl`. 0 if the file does not exist.
        """
        if os.path.exists(self.details_filepath):
            return os.path.getsize(self.details_filepath)
        return 0


class CameraSequenceMixin:
    """
    a mixin class for camera sequence data.

    directory structure::

        <root>
        ├── <Scene ID 00>
        │   │
        │   └── <version name>
        │       └── cameras  <- cameras_root
        │           ├── cam_sequence.pkl (optional)  <- camera_seq_filepath
        │           ├── <camera name 1>
        │           │   └── sparse_point_indices
        │           │       ├── <point indices 1>.npy
        │           │       └── ...
        │           │
        │           ├── <camera name 2>
        │           └── ...
        │
        ├── <Scene ID 01>
        └── ...
    """

    # separate `cameras_root` and `camera_seq_filepath` to allow
    # users to manage camera sequence files manually if needed
    cameras_root = ""
    camera_seq_filepath = ""

    @property
    def camera_sequence(self):
        if hasattr(self, "_camera_sequence"):
            return self._camera_sequence

        if not os.path.exists(self.camera_seq_filepath):
            msg = f"{self.camera_seq_filepath} does not exist"
            raise FileNotFoundError(msg)

        with open(self.camera_seq_filepath, "rb") as f:
            camera_sequence = pickle.load(f)

        assert isinstance(camera_sequence, argoverse2.CameraSequence)
        self._camera_sequence = camera_sequence
        return self._camera_sequence

    @camera_sequence.setter
    def camera_sequence(self, camera_sequence):
        if not isinstance(camera_sequence, argoverse2.CameraSequence):
            raise TypeError("Expected CameraSequence object")

        self._camera_sequence = camera_sequence
        with open(self.camera_seq_filepath, "wb") as f:
            pickle.dump(camera_sequence, f)

    @property
    def camera_sequence_bytesize(self):
        """
        num of bytes of `cam_sequence.pkl`. 0 if the file does not exist.
        """
        if os.path.exists(self.camera_seq_filepath):
            return os.path.getsize(self.camera_seq_filepath)
        return 0

    @property
    def cameras(self):
        return self.camera_sequence.list_cameras()

    @property
    def camera_unique_ids(self):
        return self.camera_sequence.list_camera_unique_ids()

    ##########
    # Getter #
    ##########

    def get_an_image_filename(self, index_or_unique_id_or_camera_name, index):

        arg = index_or_unique_id_or_camera_name
        file = self.camera_sequence.get_a_camera(arg)._files[index]
        file = os.path.basename(file)
        file = ".".join(file.split(".")[:-1])
        return file

    def get_an_image(self, index_or_unique_id_or_camera_name, index):

        arg = index_or_unique_id_or_camera_name
        img_seq = self.camera_sequence.get_a_camera(arg)
        return img_seq.get_an_image(index)

    def get_a_point_index_map(
        self,
        index_or_unique_id_or_camera_name,
        index,
        points,
        other_camera_sequences=None,
    ):

        arg = index_or_unique_id_or_camera_name

        cam = self.camera_sequence.get_a_camera(arg).camera
        uid = self.camera_sequence.get_a_camera(arg).unique_id
        if other_camera_sequences is not None:
            cam = other_camera_sequences.get_a_camera(arg).camera
            uid = other_camera_sequences.get_a_camera(arg).unique_id

        point_uid = hashlib.sha256(points.tobytes()).hexdigest()[:16]

        cache_file_path = os.path.join(
            self.cameras_root,
            f"{cam}.{uid}",
            point_uid,
            self.get_an_image_filename(arg, index) + ".npy",
        )

        if os.path.exists(cache_file_path):
            index_map = np.load(cache_file_path)
            return index_map

        os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)

        kwargs = {
            "index": index,
            "points": points,
            # hard code following params for constant caching
            "invalid_value": np.nan,
            "min_distance": 0,
            "max_distance": np.inf,
        }

        img_seq = self.camera_sequence.get_a_camera(arg)
        if other_camera_sequences is not None:
            img_seq = other_camera_sequences.get_a_camera(arg)
        index_map = img_seq.get_a_point_index_map(**kwargs)

        np.save(cache_file_path, index_map)
        return index_map

    def get_a_point_map(
        self,
        index_or_unique_id_or_camera_name,
        index,
        points,
        other_camera_sequences=None,
    ):
        arg = index_or_unique_id_or_camera_name
        index_map = self.get_a_point_index_map(
            arg,
            index,
            points,
            other_camera_sequences=other_camera_sequences,
        )

        point_map = np.full(index_map.shape + (3,), np.nan, dtype=np.float32)
        valid_map = index_map >= 0
        point_map[valid_map] = points[index_map[valid_map]]
        return point_map

    def get_a_depth_map(
        self,
        index_or_unique_id_or_camera_name,
        index,
        points,
        other_camera_sequences=None,
    ):
        arg = index_or_unique_id_or_camera_name
        point_map = self.get_a_point_map(
            arg,
            index,
            points,
            other_camera_sequences=other_camera_sequences,
        )

        img_seq = self.camera_sequence.get_a_camera(arg)
        if other_camera_sequences is not None:
            img_seq = other_camera_sequences.get_a_camera(arg)

        valid_map = ~np.isnan(point_map).any(axis=-1)

        xyz = utils_img._trans_from_world_to_camera(
            point_map[valid_map].reshape(-1, 3),
            img_seq.extrinsic[index],
        )

        depth = np.full(point_map.shape[:2], np.nan, dtype=np.float32)
        depth[valid_map] = xyz[:, 2]
        return depth


class EditedDetailsMixin:

    @property
    def deleted_pcds(self):
        """
        Return a list of PCD arrays for each deleted segment,
        aligned by matching deleted_indices with deleted_indices_of_target.
        """
        deleted_pcd = self.edited_details["deleted_points"]
        deleted_ind = self.edited_details["deleted_indices_of_target"]
        deleted_inds_groups = self.deleted_indices

        # Create mapping from target index to position in deleted_points
        target_to_deleted_idx = {
            tgt_idx: i for i, tgt_idx in enumerate(deleted_ind)
        }

        # Prepare output list of PCD arrays
        pcd_arrays = []
        for group in deleted_inds_groups:
            deleted_positions = [target_to_deleted_idx[idx] for idx in group]
            deleted_positions = np.array(deleted_positions)
            pcd_arrays.append(deleted_pcd[deleted_positions])

        return pcd_arrays

    @property
    def added_pcds(self):
        added_pcd = self.edited_details["added_points"]
        split = self.edited_details["added_splits"]
        added_pcds = np.split(added_pcd, split)
        return added_pcds

    @property
    def deleted_indices(self):

        if hasattr(self, "_deleted_indices"):
            return self._deleted_indices

        self._deleted_indices = []

        deleted_pcd = self.edited_details["deleted_points"]
        deleted_ind = self.edited_details["deleted_indices_of_target"]

        deleted_indices = self.scene_details["delete"].get("indices", [])
        if len(deleted_indices) > 0:
            self._deleted_indices.extend(deleted_indices)

            for del_ind in deleted_indices:
                del_ind = np.sort(del_ind)
                I = np.searchsorted(deleted_ind, del_ind, side="left")
                M = np.ones(len(deleted_ind), dtype=bool)
                M[I] = False
                deleted_pcd = deleted_pcd[M]
                deleted_ind = deleted_ind[M]

        if len(deleted_pcd) == 0:
            return self._deleted_indices

        deleted_det = self.scene_details["delete"]
        deleted_ann = deleted_det["annotations"]
        margin = deleted_det["margin"]

        assert deleted_ann is not None

        xyz = np.vstack([deleted_pcd["x"], deleted_pcd["y"], deleted_pcd["z"]])
        xyz = xyz.T

        # deleted_inds will be
        deleted_inds = []
        args = {"margin": margin, "separate": True}
        results = deleted_ann.is_points_in_bounding_boxes(xyz, **args)
        for indices in results:
            deleted_inds.append(deleted_ind[indices])

        deleted_inds = scene_utils.cluster_overlapping_lists(deleted_inds)
        deleted_inds = [np.sort(list(ind)) for ind in deleted_inds]

        self._deleted_indices.extend(deleted_inds)
        return self._deleted_indices

    @property
    def added_indices(self):
        return self.edited_details["added_segment_indices_of_source"]

    def added_bounding_boxes(self, margin=0.0):

        bounding_boxes = []
        for pcds in self.added_pcds:

            if len(pcds) == 0:
                continue

            xyz = np.vstack([pcds["x"], pcds["y"], pcds["z"]]).T

            R, mean = scene_utils.modified_pcd_projection(xyz)

            xyz = np.sum((xyz - mean)[:, None, :] * R, axis=-1)
            x_min, y_min, z_min = np.min(xyz, axis=0)
            x_max, y_max, z_max = np.max(xyz, axis=0)

            lx = x_max - x_min + margin
            ly = y_max - y_min + margin
            lz = z_max - z_min + margin

            # visualization_pptk.make_bounding_box_vertices will return bbox
            # whose center is at (0, 0, 0) so we need to offset it to the
            # correct position
            off = np.r_[x_min, y_min, z_min] - (-1.0 * np.r_[lx, ly, lz] / 2.0)

            # (8, 3)
            args = (lx, ly, lz)
            vertices = visualization_pptk.make_bounding_box_vertices(*args)
            vertices = vertices + off
            vertices = np.sum(vertices[:, None, :] * R.T, axis=-1) + mean
            bounding_boxes.append(vertices)

        return bounding_boxes

    def deleted_bounding_boxes(self, margin=0.0):

        bounding_boxes = []
        for pcds in self.deleted_pcds:

            if len(pcds) == 0:
                continue

            xyz = np.vstack([pcds["x"], pcds["y"], pcds["z"]]).T
            R, mean = scene_utils.modified_pcd_projection(xyz)

            xyz = np.sum((xyz - mean)[:, None, :] * R, axis=-1)
            x_min, y_min, z_min = np.min(xyz, axis=0)
            x_max, y_max, z_max = np.max(xyz, axis=0)

            lx = x_max - x_min + margin
            ly = y_max - y_min + margin
            lz = z_max - z_min + margin

            # visualization_pptk.make_bounding_box_vertices will return bbox
            # whose center is at (0, 0, 0) so we need to offset it to the
            # correct position
            off = np.r_[x_min, y_min, z_min] - (-1.0 * np.r_[lx, ly, lz] / 2.0)

            # (8, 3)
            args = (lx, ly, lz)
            vertices = visualization_pptk.make_bounding_box_vertices(*args)
            vertices = vertices + off
            vertices = np.sum(vertices[:, None, :] * R.T, axis=-1) + mean
            bounding_boxes.append(vertices)

        return bounding_boxes
