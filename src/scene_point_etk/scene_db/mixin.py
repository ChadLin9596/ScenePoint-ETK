import os
import glob
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
    a mixin class for scene PCD data.

    ```
    <root>
    ├── <Scene ID 00>
    │   └── <version name>
    │       └── scene.pcd  <- scene_filepath
    │
    ├── <Scene ID 01>
    └── ...
    ```
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
        return np.vstack([r, g, b]).T


class SceneDetailsMixin:
    """
    a mixin class for scene details data.

    ```
    <root>
    ├── <Scene ID 00>
    │   └── <version name>
    │       └── details.pkl  <- details_filepath
    │
    ├── <Scene ID 01>
    └── ...
    ```
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


class CameraSequenceMixin:
    """
    a mixin class for camera sequence data.

    ```
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
    ```
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
    def cameras(self):
        return self.camera_sequence.list_cameras()

    def get_camera_filenames(self, camera_name):
        files = self.camera_sequence.get_a_camera(camera_name)._files
        files = [os.path.basename(i) for i in files]
        return files

    def _initialize_cam_fnames(self, camera_name, filename_or_index):

        if isinstance(filename_or_index, int):
            ind = filename_or_index
            filename_or_index = self.get_camera_filenames(camera_name)[ind]

        assert isinstance(filename_or_index, str)
        filename = filename_or_index.replace(".npy", "")
        filename = filename.replace(".png", "")
        filename = filename.replace(".jpg", "")
        return filename

    def get_an_image(self, camera_name, index):

        img_seq = self.camera_sequence.get_a_camera(camera_name)
        return img_seq.get_an_image(index)

    def _point_indices_on_an_image(
        self,
        camera_name,
        index,
        xyz,
        min_distance=0.0,
        max_distance=np.inf,
    ):

        img_seq = self.camera_sequence.get_a_camera(camera_name)
        intrinsic = img_seq.intrinsic
        extrinsic = img_seq.extrinsic[index]
        H, W = img_seq.figsize

        args = (xyz, intrinsic, extrinsic, H, W)
        kwargs = {"min_distance": min_distance, "max_distance": max_distance}
        index_map = utils_img.points_to_index_map(*args, **kwargs)
        return index_map

    def point_indices_on_an_image(
        self,
        camera_name,
        index,
        xyz,
        min_distance=0.0,
        max_distance=np.inf,
        overwrite=False,
    ):

        root = self.cameras_root
        root = os.path.join(root, camera_name, "sparse_point_indices")
        os.makedirs(root, exist_ok=True)

        filename = self.get_camera_filenames(camera_name)[index]
        filename = filename.replace(".npy", "")
        filename = filename.replace(".png", "")
        filename = filename.replace(".jpg", "")
        filename = os.path.join(root, filename + ".npy")

        if os.path.exists(filename) and not overwrite:
            index_map = np.load(filename)
            return index_map

        args = (camera_name, index, xyz)
        kwargs = {"min_distance": min_distance, "max_distance": max_distance}
        index_map = self._point_indices_on_an_image(*args, **kwargs)

        if overwrite:
            np.save(filename, index_map)

        return index_map

    def point_on_an_image(
        self,
        camera_name,
        index,
        xyz,
        min_distance=0.0,
        max_distance=np.inf,
    ):

        img_seq = self.camera_sequence.get_a_camera(camera_name)
        intrinsic = img_seq.intrinsic
        extrinsic = img_seq.extrinsic[index]
        H, W = img_seq.figsize

        args = (xyz, intrinsic, extrinsic, H, W)
        kwargs = {
            "min_distance": min_distance,
            "max_distance": max_distance,
            "invalid_value": np.nan,
        }
        point_map = utils_img.points_to_point_map(*args, **kwargs)
        return point_map

    def depth_on_an_image(
        self,
        camera_name,
        index,
        xyz,
        min_distance=0.0,
        max_distance=np.inf,
    ):

        img_seq = self.camera_sequence.get_a_camera(camera_name)
        extrinsic = img_seq.extrinsic[index]

        kwargs = {"min_distance": min_distance, "max_distance": max_distance}
        point_map = self.point_on_an_image(camera_name, index, xyz, **kwargs)

        depth = np.full(point_map.shape[:2], np.nan, dtype=np.float32)
        valid_map = ~np.isnan(point_map).any(axis=-1)

        xyz = point_map[valid_map]
        xyz = utils_img._trans_from_world_to_camera(xyz, extrinsic)
        depth[valid_map] = xyz[:, 2]

        return depth

    @property
    def camera_point_indices_map(self):

        if hasattr(self, "_camera_point_indices_map"):
            return self._camera_point_indices_map.copy()

        point_indices = {}

        for camera in self.cameras:

            root = self.cameras_root
            root = os.path.join(
                self.cameras_root, camera, "sparse_point_indices"
            )
            point_indices_root = os.path.join(root, "sparse_point_indices")
            os.makedirs(point_indices_root, exist_ok=True)

            img_seq = self.camera_sequence.get_a_camera(camera)
            intrinsic = img_seq.intrinsic
            extrinsics = img_seq.extrinsic
            xyz = self.pcd_xyz
            H, W = img_seq.figsize

            indices_maps = []
            for idx, extrinsic in enumerate(extrinsics):

                name = self._initialize_cam_fnames(camera, idx) + ".npy"
                path = os.path.join(point_indices_root, name)
                # if os.path.exists(path):
                #     index_map = np.load(path)
                #     indices_maps.append(index_map)
                #     continue

                args = (xyz, intrinsic, extrinsic, H, W)
                kwargs = {"min_distance": 0.0, "max_distance": np.inf}
                index_map = utils_img.points_to_index_map(*args, **kwargs)
                indices_maps.append(index_map)
                np.save(path, index_map)

            indices_maps = np.array(indices_maps)
            point_indices[camera] = indices_maps

        self._camera_point_indices_map = point_indices
        return self._camera_point_indices_map.copy()

    @property
    def camera_point_map(self):

        if hasattr(self, "_camera_point_map"):
            return self._camera_point_map.copy()

        point_maps = {}

        for camera in self.cameras:
            indices_maps = self.camera_point_indices_map[camera]
            xyz = self.pcd_xyz

            shape = indices_maps.shape + (3,)
            valid_map = indices_maps != -1
            point_map = np.full(shape, np.nan, dtype=np.float32)
            point_map[valid_map] = xyz[indices_maps[valid_map]]
            point_maps[camera] = point_map

        self._camera_point_map = point_maps
        return self._camera_point_map.copy()

    @property
    def camera_depth_map(self):

        if hasattr(self, "_camera_depth_map"):
            return self._camera_depth_map.copy()

        depth_maps = {}

        for camera in self.cameras:

            img_seq = self.camera_sequence.get_a_camera(camera)

            extrinsics = img_seq.extrinsic  # (N, 4, 4)
            point_map = self.camera_point_map[camera]

            depth_map = []
            for extrinsic, point in zip(extrinsics, point_map):

                depth = np.full(point.shape[:2], np.nan, dtype=np.float32)
                valid_map = ~np.isnan(point).any(axis=-1)

                if not np.any(valid_map):
                    depth_map.append(depth)
                    continue

                xyz = point[valid_map]
                xyz = utils_img._trans_from_world_to_camera(xyz, extrinsic)
                depth[valid_map] = xyz[:, 2]
                depth_map.append(depth)

            depth_map = np.array(depth_map)
            depth_maps[camera] = depth_map

        self._camera_depth_map = depth_maps
        return self._camera_depth_map.copy()

    def _chunkify(self, camera_name, chunk_size=50, with_overlap=True):

        img_seq = self.camera_sequence.get_a_camera(camera_name)
        segments = []

        if with_overlap:

            f = utils_segmentation.compute_sliding_window_indices_with_overlap
            s_inds, e_inds = f(len(img_seq), chunk_size, overlap_ratio=0.3)

            for s, e in zip(s_inds, e_inds):
                segment = np.arange(s, e)
                segments.append(segment)

            # indices = np.arange(len(img_seq))
            # segments.extend(segment_with_overlap_full(indices, chunk_size))

        else:
            indices = np.arange(0, len(img_seq), chunk_size)
            indices = np.r_[indices, len(img_seq)]

            for start, end in zip(indices[:-1], indices[1:]):
                segment = np.arange(start, end)
                segments.append(segment)

        return segments

    def chunkify_images(self, camera_name, chunk_size=50, with_overlap=True):

        segments = self._chunkify(camera_name, chunk_size, with_overlap)

        for segment in segments:

            segment_images = []
            for idx in segment:
                image = self.get_an_image(camera_name, idx)
                segment_images.append(image)

            yield np.array(segment_images)

    def chunkify_point_map(
        self, camera_name, chunk_size=50, with_overlap=True
    ):

        point_map = self.camera_point_map[camera_name]
        segments = self._chunkify(camera_name, chunk_size, with_overlap)

        for segment in segments:
            segment_points = point_map[segment]
            yield segment_points


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
