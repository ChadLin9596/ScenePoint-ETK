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


def segment_with_overlap_full(indices, n):
    indices = np.asarray(indices)
    step = n - n // 3
    segments = []

    start = 0
    while start + n <= len(indices):
        segments.append(indices[start : start + n])
        start += step

    # Handle last segment
    if start < len(indices):
        last_start = len(indices) - n
        if len(indices) - start >= n // 3:
            segments.append(indices[last_start : last_start + n])

    return segments


class ScenePCDMixin:

    scene_filepath = ""

    @property
    def scene_pcd(self):
        if hasattr(self, "_scene_pcd"):
            return self._scene_pcd.copy()

        if not os.path.exists(self.scene_filepath):
            raise FileNotFoundError(f"{self.scene_filepath} does not exist")

        self._scene_pcd = pcd.read(self.scene_filepath)
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

    details_filepath = ""

    @property
    def scene_details(self):
        if not os.path.exists(self.details_filepath):
            return None

        with open(self.details_filepath, "rb") as fd:
            return pickle.load(fd)

    @scene_details.setter
    def scene_details(self, details):
        with open(self.details_filepath, "wb") as fd:
            pickle.dump(details, fd)


class CameraSequenceMixin:

    camera_seq_filepath = ""

    @property
    def camera_sequence(self):
        if hasattr(self, "_camera_sequence"):
            return self._camera_sequence

        if not os.path.exists(self.camera_seq_filepath):
            raise FileNotFoundError(
                f"{self.camera_seq_filepath} does not exist"
            )

        with open(self.camera_seq_filepath, "rb") as f:
            camera_sequence = pickle.load(f)

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

    def get_an_image(self, camera_name, filename_or_index):

        filename = self._initialize_cam_fnames(camera_name, filename_or_index)

        camera_root = os.path.join(self.cameras_root, camera_name)
        images_root = os.path.join(camera_root, "images")
        filepath = os.path.join(images_root, f"{filename}.png")

        return plt.imread(filepath)[..., :3]  # read as RGB

    def get_camera_filenames(self, camera_name):
        camera_sequence = self.camera_sequence.get_a_camera(camera_name)
        return list(map(str, camera_sequence.timestamps))

    def _initialize_cam_fnames(self, camera_name, filename_or_index):

        if isinstance(filename_or_index, int):
            ind = filename_or_index
            filename_or_index = self.get_camera_filenames(camera_name)[ind]

        assert isinstance(filename_or_index, str)
        filename = filename_or_index.replace(".npy", "")
        filename = filename.replace(".png", "")
        filename = filename.replace(".jpg", "")
        return filename

    @property
    def camera_point_indices_map(self):

        if hasattr(self, "_camera_point_indices_map"):
            return self._camera_point_indices_map.copy()

        point_indices = {}

        for camera in self.cameras:

            root = os.path.split(self.camera_seq_filepath)[0]
            root = os.path.join(root, camera)
            point_indices_root = os.path.join(root, "sparse_point_indices")
            os.makedirs(point_indices_root, exist_ok=True)

            fs = os.path.join(root, "sparse_point_indices", "*.npy")
            fs = sorted(glob.glob(fs), key=lambda x: os.path.basename(x))

            if len(fs) > 0:
                indices_maps = np.array([np.load(f) for f in fs])
                point_indices[camera] = indices_maps
                continue

            img_seq = self.camera_sequence.get_a_camera(camera)
            intrinsic = img_seq.intrinsic
            extrinsics = img_seq.extrinsic
            xyz = self.pcd_xyz
            H, W = img_seq.figsize

            indices_maps = []
            for idx, extrinsic in enumerate(extrinsics):
                args = (xyz, intrinsic, extrinsic, H, W)
                kwargs = {"min_distance": 0.0, "max_distance": np.inf}
                index_map = utils_img.points_to_index_map(*args, **kwargs)
                indices_maps.append(index_map)

                name = self._initialize_cam_fnames(camera, idx) + ".npy"
                np.save(os.path.join(point_indices_root, name), index_map)
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
            indices = np.arange(len(img_seq))
            segments.extend(segment_with_overlap_full(indices, chunk_size))

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
            vertice = visualization_pptk.make_bounding_box_vertices(lx, ly, lz)
            vertice += off

            vertice = np.sum(vertice[:, None, :] * R.T, axis=-1) + mean
            bounding_boxes.append(vertice)
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
            vertice = visualization_pptk.make_bounding_box_vertices(lx, ly, lz)
            vertice = vertice + off
            vertice = np.sum(vertice[:, None, :] * R.T, axis=-1) + mean
            bounding_boxes.append(vertice)
        return bounding_boxes

    def camera_change_map(
        self,
        neighborhood_size=7,
        num_valid_points=3,
        depth_threshold=0.1,
    ):

        if hasattr(self, "_camera_change_map"):
            return self._camera_change_map.copy()

        camera_change_map = {}

        for camera in self.cameras:

            img_seq = self.camera_sequence.get_a_camera(camera)
            lidar_sweeps = self.raw_lidar_sweeps

            assert len(img_seq) == len(lidar_sweeps), (
                f"Camera {camera} and lidar sweeps have different lengths: "
                f"{len(img_seq)} vs {len(lidar_sweeps)}"
            )

            cd_masks = []
            for index in range(len(img_seq)):

                scene_index_map = self.camera_point_indices_map[camera][index]
                scene_depth_map = self.camera_depth_map[camera][index]
                lidar_depth_map = img_seq.get_a_depth_map(
                    index,
                    lidar_sweeps[index].xyz,
                    invalid_value=np.nan,
                )

                cd_mask = np.zeros_like(scene_index_map, dtype=bool)
                for indices in self.deleted_indices:

                    mask = np.isin(scene_index_map, indices)
                    if not np.any(mask):
                        continue

                    mask = scene_utils.filter_visible(
                        scene_depth_map,
                        lidar_depth_map,
                        FOV_mask=mask,
                        neighborhood_size=neighborhood_size,
                        depth_threshold=depth_threshold,
                        num_valid_points=num_valid_points,
                    )
                    mask = utils_img.fill_sparse_boolean_by_convex_hull(mask)
                    cd_mask |= mask
                cd_masks.append(cd_mask)
            camera_change_map[camera] = np.array(cd_masks)

        self._camera_change_map = camera_change_map
        return self._camera_change_map.copy()
