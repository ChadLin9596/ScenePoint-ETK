import os
import glob
import pickle
import shutil
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import scene_point_etk.argoverse2 as argoverse2
from . import diff_scene
from .. import utils as scene_utils
from .mixin import *

import py_utils.array_data as array_data
import py_utils.pcd as pcd
import py_utils.utils as utils
import py_utils.utils_img as utils_img
import py_utils.visualization_pptk as vis_pptk


class Base(ScenePCDMixin, SceneDetailsMixin, CameraSequenceMixin):
    """
    ├── <Scene ID 00>
    │   │
    │   └── <version name>
    │       ├── details.pkl
    │       ├── scene.pcd
    │       └── cameras
    │           ├── cam_sequence.pkl (optional)
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

    def __init__(self, scene_root, version):

        self.root = scene_root
        self.version = version

        join = os.path.join
        self.scene_root = join(scene_root, version)

        # setup for mixins: ScenePCD, SceneDetails, CameraSequence
        self.scene_filepath = join(self.scene_root, "scene.pcd")
        self.details_filepath = join(self.scene_root, "details.pkl")
        self.cameras_root = join(self.scene_root, "cameras")
        self.camera_seq_filepath = join(self.cameras_root, "cam_sequence.pkl")
        os.makedirs(self.cameras_root, exist_ok=True)

    @property
    def bytesize(self):
        R = {
            "scene.pcd": 0,
            "details.pkl": 0,
            "cam_sequence.pkl": 0,
            "cameras": {}
        }
        if os.path.exists(self.scene_filepath):
            R["scene.pcd"] = os.path.getsize(self.scene_filepath)
        if os.path.exists(self.details_filepath):
            R["details.pkl"] = os.path.getsize(self.details_filepath)
        if os.path.exists(self.camera_seq_filepath):
            R["cam_sequence.pkl"] = os.path.getsize(self.camera_seq_filepath)

        for camera in self.cameras:
            camera_path = os.path.join(self.cameras_root, camera)
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(camera_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.islink(fp):
                        continue

                    total_size += os.path.getsize(fp)
            R["cameras"][camera] = total_size

        return R

    def export(self, output_dir, skip_scene=False, skip_cameras=False):

        version = os.path.basename(self.scene_root)
        scene_name = os.path.basename(os.path.dirname(self.scene_root))

        output_root = os.path.join(output_dir, scene_name, version)
        os.makedirs(output_root, exist_ok=True)

        if not skip_scene:
            path = os.path.join(output_root, "details.pkl")
            shutil.copy(self.details_filepath, path)

        if not skip_cameras:
            os.makedirs(os.path.join(output_root, "cameras"), exist_ok=True)
            path = os.path.join(output_root, "cameras", "cam_sequence.pkl")
            shutil.copy(self.camera_seq_filepath, path)


class OriginalScene(Base):
    """
    based on Scene structure, but with fixed scene name "GT" and added
    the following structure:
    ├── <Scene ID 00>
    │   │
    │   └── GT <- fixed <scene name> to "GT"
    │       └── cameras
    │           ├── <camera name 1>
    │           │   └── images
    │           │       ├── <image 1>.png
    │           │       └── ...
    │           │
    │           ├── <camera name 2>
    │           └── ...
    │
    ├── <Scene ID 01>
    └── ...
    """

    def __init__(self, scene_root):

        super().__init__(scene_root, "GT")

    @property
    def scene_pcd(self):
        # override ScenePCDMixin.scene_pcd to generate scene.pcd from
        # scene_details if scene.pcd not exists

        if hasattr(self, "_scene_pcd"):
            return self._scene_pcd.copy()

        # generate scene.pcd if not exists
        # the generated scene.pcd will be cached for future use
        # the generated scene.pcd should be identical if the details.pkl
        # is not changed
        if not os.path.exists(self.scene_filepath):
            details = self.scene_details
            voxel_size = details.get("voxel_size", 0.2)

            sweeps = argoverse2.SweepSequence.from_sweeps(details["sweeps"])
            scene = sweeps.export_to_voxel_grid(voxel_size=voxel_size)
            pcd.write(self.scene_filepath, scene)

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
    def raw_lidar_sweeps(self):
        """
        a list of argoverse2.Sweep containing both static/dynamic points
        """

        if hasattr(self, "_raw_lidar_sweeps"):
            return self._raw_lidar_sweeps

        ss = self.scene_details["sweeps"]
        ss = [argoverse2.Sweep(s.filepath, coordinate="map") for s in ss]
        self._raw_lidar_sweeps = ss

        return self._raw_lidar_sweeps

    @property
    def cleaned_lidar_sweeps(self):
        """
        a list of argoverse2.Sweep containing only static points
        """
        return self.scene_details["sweeps"]



class EditedScene(Base, EditedDetailsMixin):
    """
    based on Scene structure, but assume the GT scene is already processed
    and has the following structure:
    ├── <Scene ID 00>
    │   │
    │   └── <version name> <- can be any name but not "GT"
    │       ├── details.pkl
    │       ├── scene.pcd
    │       └── cameras <- generated by GT's camera_sequence.pkl
    │           ├── <camera name 1>
    │           │   ├── sparse_depths
    │           │   │   ├── <depth 1>.npy
    │           │   │   └── ...
    │           │   ├── sparse_point_map
    │           │   │   ├── <point map 1>.npy
    │           │   │   └── ...
    │           │   ├── sparse_point_indices
    │           │   │   ├── <point indices 1>.npy
    │           │   │   └── ...
    │           │   ├── sparse_changed_masks
    │           │   │   ├── source
    │           │   │   │   ├── <mask 1>.png
    │           │   │   │   └── ...
    │           │   │   └── target
    │           │   │       ├── <mask 1>.png
    │           │   │       └── ...
    │           │   └── dense_changed_masks
    │           │       ├── source
    │           │       │   ├── <mask 1>.png
    │           │       │   └── ...
    │           │       └── target
    │           │           ├── <mask 1>.png
    │           │           └── ...
    │           │
    │           ├── <camera name 2>
    │           └── ...
    │
    ├── <Scene ID 01>
    └── ...
    """

    def __init__(self, scene_root, version):

        if version == "GT":
            raise ValueError("EditedScene cannot have scene name 'GT'")

        super().__init__(scene_root, version)
        if not os.path.exists(os.path.join(scene_root, "GT", "details.pkl")):
            msg = f"GT scene not found at {os.path.join(scene_root, 'GT')}"
            raise RuntimeError(msg)

        # reuse the camera sequence from the GT scene
        p = os.path.join(self.root, "GT", "cameras", "cam_sequence.pkl")
        self.camera_seq_filepath = os.path.join(p)

    @property
    def edited_details(self):

        scene = OriginalScene(self.root)
        _, details = diff_scene.apply_change_info_to_target_pcd(
            scene.scene_pcd,
            self.scene_details,
            return_details=True,
        )
        self._edited_details = details
        return self._edited_details

    @property
    def scene_pcd(self):
        # override ScenePCDMixin.scene_pcd to generate scene.pcd from
        # scene_details if scene.pcd not exists
        # note that the way to generate scene.pcd is different from
        # OriginalScene

        if hasattr(self, "_scene_pcd"):
            return self._scene_pcd.copy()

        if not os.path.exists(self.scene_filepath):

            original_scene = OriginalScene(self.root)
            scene = diff_scene.apply_change_info_to_target_pcd(
                original_scene.scene_pcd,
                self.scene_details,
            )
            pcd.write(self.scene_filepath, scene)

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
    def deleted_lidar_sweeps(self):

        if hasattr(self, "_deleted_lidar_sweeps"):
            return self._deleted_lidar_sweeps

        orig_scene = OriginalScene(self.root)
        lidar_sweeps = orig_scene.cleaned_lidar_sweeps

        if len(self.edited_details["deleted_indices_of_target"]) == 0:
            # no deleted points
            self._deleted_lidar_sweeps = [i - i for i in lidar_sweeps]
            return self._deleted_lidar_sweeps

        sweeps = argoverse2.SweepSequence.from_sweeps(lidar_sweeps)
        _, details = sweeps.export_to_voxel_grid(
            voxel_size=orig_scene.scene_details.get("voxel_size", 0.2),
            skip_color=True,
            return_details=True,
        )

        original_indices = np.arange(len(sweeps.xyz))
        original_indices = original_indices[details["indices"]]
        original_indicess = np.split(original_indices, details["splits"][1:-1])

        selected = []
        for i in self.edited_details["deleted_indices_of_target"]:
            selected.append(original_indicess[i])
        selected = np.sort(np.hstack(selected))

        sweep_starts = np.r_[0, np.cumsum([len(s) for s in sweeps.sweeps])]
        starts = np.searchsorted(sweep_starts, selected, side="right") - 1
        splits = np.where(np.diff(starts) > 0)[0] + 1
        splits = np.r_[0, splits, len(starts)]

        # to local indices
        selected = selected - sweep_starts[starts]

        self._deleted_lidar_sweeps = []
        for i, j in zip(splits[:-1], splits[1:]):

            s = starts[i]
            sweep = sweeps.sweeps[s]
            deleted_sweep = sweep[selected[i:j]]
            self._deleted_lidar_sweeps.append(deleted_sweep)

        return self._deleted_lidar_sweeps

    @property
    def added_lidar_sweeps(self):
        # how to form scan points on new inserted objects?
        raise NotImplementedError

    def camera_change_map(
        self,
        neighborhood_size=7,
        num_valid_points=3,
        depth_threshold=0.1,
    ):

        if hasattr(self, "_camera_change_map"):
            return self._camera_change_map.copy()

        origin_scene = OriginalScene(self.root)
        camera_point_indices_map = origin_scene.camera_point_indices_map
        camera_depth_map = origin_scene.camera_depth_map

        camera_change_map = {}

        for camera in self.cameras:

            img_seq = self.camera_sequence.get_a_camera(camera)
            lidar_sweeps = origin_scene.raw_lidar_sweeps
            lidar_sweeps = argoverse2.SweepSequence.from_sweeps(lidar_sweeps)
            lidar_sweeps = lidar_sweeps.align_timestamps(img_seq.timestamps)
            lidar_sweeps = lidar_sweeps.sweeps

            assert len(img_seq) == len(lidar_sweeps), (
                f"Camera {camera} and lidar sweeps have different lengths: "
                f"{len(img_seq)} vs {len(lidar_sweeps)}"
            )

            cd_masks = []
            for index in range(len(img_seq)):

                scene_index_map = camera_point_indices_map[camera][index]
                scene_depth_map = camera_depth_map[camera][index]
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
                    mask = mask > 0
                    cd_mask |= mask
                cd_masks.append(cd_mask)
            camera_change_map[camera] = np.array(cd_masks)

        self._camera_change_map = camera_change_map
        return self._camera_change_map.copy()
