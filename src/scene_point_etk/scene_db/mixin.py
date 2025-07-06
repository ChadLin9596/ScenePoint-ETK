import os
import glob
import pickle
import numpy as np

from .. import argoverse2
from .. import patch_db
from .. import utils as scene_utils
from . import diff_scene

import py_utils.utils as utils
import py_utils.pcd as pcd
import py_utils.utils_img as utils_img
import py_utils.visualization_pptk as visualization_pptk


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
    def point_indices_map(self):

        if hasattr(self, "_point_indices_map"):
            return self._point_indices_map.copy()

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

        self._point_indices_map = point_indices
        return self._point_indices_map.copy()


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

        deleted_pcd = self.edited_details["deleted_points"]
        deleted_ind = self.edited_details["deleted_indices_of_target"]

        if len(deleted_pcd) == 0:
            self._deleted_indices = []
            return self._deleted_indices

        deleted_det = self.scene_details["delete"]
        deleted_ann = deleted_det["annotations"]
        margin = deleted_det["margin"]

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

        self._deleted_indices = deleted_inds
        return self._deleted_indices

    @property
    def added_indices(self):
        return self.edited_details["added_segment_indices_of_source"]

    @property
    def added_bounding_boxes(self):

        bounding_boxes = []
        for pcds in self.added_pcds:

            xyz = np.vstack([pcds["x"], pcds["y"], pcds["z"]]).T
            R, mean = scene_utils.modified_pcd_projection(xyz)

            xyz = np.sum((xyz - mean)[:, None, :] * R, axis=-1)
            x_min, y_min, z_min = np.min(xyz, axis=0)
            x_max, y_max, z_max = np.max(xyz, axis=0)

            lx = x_max - x_min
            ly = y_max - y_min
            lz = z_max - z_min

            # (8, 3)
            vertice = visualization_pptk.make_bounding_box_vertices(lx, ly, lz)
            vertice = np.sum(vertice[:, None, :] * R.T, axis=-1) + mean
            bounding_boxes.append(vertice)
        return bounding_boxes

    @property
    def deleted_bounding_boxes(self):

        bounding_boxes = []
        for pcds in self.deleted_pcds:

            xyz = np.vstack([pcds["x"], pcds["y"], pcds["z"]]).T
            R, mean = scene_utils.modified_pcd_projection(xyz)

            xyz = np.sum((xyz - mean)[:, None, :] * R, axis=-1)
            x_min, y_min, z_min = np.min(xyz, axis=0)
            x_max, y_max, z_max = np.max(xyz, axis=0)

            lx = x_max - x_min
            ly = y_max - y_min
            lz = z_max - z_min

            # (8, 3)
            vertice = visualization_pptk.make_bounding_box_vertices(lx, ly, lz)
            vertice = np.sum(vertice[:, None, :] * R.T, axis=-1) + mean
            bounding_boxes.append(vertice)
        return bounding_boxes
