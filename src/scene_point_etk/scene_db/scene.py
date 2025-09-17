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

    # def process_camera(self, camera_name):

    #     camera_root = os.path.join(self.cameras_root, camera_name)
    #     depths_root = os.path.join(camera_root, "sparse_depths")
    #     point_map_root = os.path.join(camera_root, "sparse_point_map")
    #     point_indices_root = os.path.join(camera_root, "sparse_point_indices")
    #     os.makedirs(depths_root, exist_ok=True)
    #     os.makedirs(point_map_root, exist_ok=True)
    #     os.makedirs(point_indices_root, exist_ok=True)

    #     # get the essential data
    #     # if they are not already set, error will be raised
    #     img_seq = self.camera_sequence.get_a_camera(camera_name)
    #     names = img_seq.timestamps  # (N, )
    #     intrinsic = img_seq.intrinsic  # (3, 3)
    #     extrinsics = img_seq.extrinsic  # (N, 4, 4)

    #     scene = self.scene_pcd
    #     scene_xyz = np.vstack([scene["x"], scene["y"], scene["z"]]).T

    #     N = len(img_seq)
    #     prog = utils.ProgressTimer(prefix=f"Processing camera {camera_name} ")
    #     prog.tic(N)
    #     for name, extrinsic in zip(names, extrinsics):

    #         depth_map, point_map, details = utils_img.points_to_depth_image(
    #             points=scene_xyz,
    #             intrinsic=intrinsic,
    #             extrinsic=extrinsic,
    #             H=img_seq.figsize[0],
    #             W=img_seq.figsize[1],
    #             invalid_value=-1,
    #             min_distance=0,
    #             max_distance=np.inf,
    #             return_details=True,
    #             other_attrs=[scene_xyz],  # assign scene_xyz to get point map
    #         )

    #         # save sparse point indices
    #         u, v = details["uv"]
    #         FOV_mask = details["FOV_mask"]
    #         indices_map = np.full(img_seq.figsize, -1, dtype=np.int32)
    #         indices_map[u, v] = np.arange(len(scene_xyz))[FOV_mask]

    #         name = f"{name}.npy"
    #         np.save(os.path.join(depths_root, name), depth_map)
    #         np.save(os.path.join(point_map_root, name), point_map)
    #         np.save(os.path.join(point_indices_root, name), indices_map)

    #         prog.toc()

    # def get_a_sparse_depth(self, camera_name, filename_or_index):

    #     filename = self._initialize_cam_fnames(camera_name, filename_or_index)

    #     camera_root = os.path.join(self.cameras_root, camera_name)
    #     depth_root = os.path.join(camera_root, "sparse_depths")
    #     filepath = os.path.join(depth_root, f"{filename}.npy")
    #     return np.load(filepath)

    # def get_a_sparse_point_map(self, camera_name, filename_or_index):

    #     filename = self._initialize_cam_fnames(camera_name, filename_or_index)

    #     camera_root = os.path.join(self.cameras_root, camera_name)
    #     point_root = os.path.join(camera_root, "sparse_point_map")
    #     filepath = os.path.join(point_root, f"{filename}.npy")
    #     return np.load(filepath)

    # def get_a_sparse_point_indices(self, camera_name, filename_or_index):

    #     filename = self._initialize_cam_fnames(camera_name, filename_or_index)

    #     camera_root = os.path.join(self.cameras_root, camera_name)
    #     point_indices_root = os.path.join(camera_root, "sparse_point_indices")
    #     filepath = os.path.join(point_indices_root, f"{filename}.npy")
    #     return np.load(filepath)

    # def get_a_depth_image(self, camera_name, filename_or_index):

    #     filename = self._initialize_cam_fnames(camera_name, filename_or_index)

    #     camera_root = os.path.join(self.cameras_root, camera_name)
    #     depth_img_root = os.path.join(camera_root, "sparse_depths_img")
    #     filepath = os.path.join(depth_img_root, f"{filename}.png")

    #     if not os.path.exists(filepath):
    #         raise FileNotFoundError(f"Depth image {filepath} does not exist")

    #     return plt.imread(filepath)[..., :3]

    # def process_depth_images(self, camera_name):

    #     camera_root = os.path.join(self.cameras_root, camera_name)
    #     depth_img_root = os.path.join(camera_root, "sparse_depths_img")
    #     os.makedirs(depth_img_root, exist_ok=True)

    #     img_seq = self.camera_sequence.get_a_camera(camera_name)
    #     figsize = tuple(img_seq.figsize)
    #     color_fn = vis_pptk.make_color(
    #         s_min=10,
    #         s_max=80,
    #         color_map=[[0, 0, 0], [1, 1, 1]],
    #     )
    #     figsize = tuple(img_seq.figsize)
    #     N = len(img_seq)
    #     for index in range(N):
    #         name = f"{self._initialize_cam_fnames(camera_name, index)}.png"

    #         depth_img = np.ones(figsize + (3,), dtype=np.float32)
    #         depth_map = self.get_a_sparse_depth(camera_name, index)
    #         valid_mask = depth_map > 0
    #         depth_img[valid_mask] = color_fn(depth_map[valid_mask])

    #         plt.imsave(os.path.join(depth_img_root, name), depth_img)

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

    # def get_an_image(self, camera_name, filename_or_index):

    #     filename = self._initialize_cam_fnames(camera_name, filename_or_index)

    #     camera_root = os.path.join(self.cameras_root, camera_name)
    #     images_root = os.path.join(camera_root, "images")
    #     filepath = os.path.join(images_root, f"{filename}.png")

    #     return plt.imread(filepath)[..., :3]  # read as RGB

    # def process_camera(self, camera_name):

    #     super().process_camera(camera_name)

    #     camera_root = os.path.join(self.cameras_root, camera_name)
    #     images_root = os.path.join(camera_root, "images")
    #     os.makedirs(images_root, exist_ok=True)

    #     img_seq = self.camera_sequence.get_a_camera(camera_name)
    #     names = img_seq.timestamps
    #     H, W = img_seq.figsize

    #     N = len(img_seq)
    #     prog = utils.ProgressTimer(prefix=f"Saving {camera_name} {H}x{W} img ")
    #     prog.tic(N)

    #     for index in range(N):
    #         name = f"{names[index]}.png"
    #         img = img_seq.get_an_image(index)
    #         plt.imsave(os.path.join(images_root, name), img)
    #         prog.toc()

    @property
    def scene_pcd(self):
        if hasattr(self, "_scene_pcd"):
            return self._scene_pcd.copy()

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
        if hasattr(self, "_raw_lidar_sweeps"):
            return self._raw_lidar_sweeps

        ss = self.scene_details["sweeps"]
        ss = [argoverse2.Sweep(s.filepath, coordinate="map") for s in ss]
        self._raw_lidar_sweeps = ss

        return self._raw_lidar_sweeps


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
    def raw_lidar_sweeps(self):
        if hasattr(self, "_raw_lidar_sweeps"):
            return self._raw_lidar_sweeps

        ss = OriginalScene(self.root).scene_details["sweeps"]
        ss = [argoverse2.Sweep(s.filepath, coordinate="map") for s in ss]
        self._raw_lidar_sweeps = ss

        return self._raw_lidar_sweeps

    @property
    def unchanged_lidar_sweeps(self):
        if hasattr(self, "_unchanged_lidar_sweeps"):
            return self._unchanged_lidar_sweeps

        scene_details = OriginalScene(self.root).scene_details
        ss = argoverse2.SweepSequence.from_sweeps(scene_details["sweeps"])

        _, details = ss.export_to_voxel_grid(
            voxel_size=scene_details.get("voxel_size", 0.2),
            skip_color=True,
            return_details=True,
        )
        original_indices = np.arange(len(ss.xyz))
        original_indices = original_indices[details["indices"]]
        original_indicess = np.split(original_indices, details["splits"][1:-1])

        if len(self.edited_details["deleted_indices_of_target"]) == 0:
            self._unchanged_lidar_sweeps = ss.sweeps
            return self._unchanged_lidar_sweeps

        selected = []
        for i in self.edited_details["deleted_indices_of_target"]:
            selected.append(original_indicess[i])
        selected = np.sort(np.hstack(selected))

        sweep_starts = np.r_[0, np.cumsum([len(s) for s in ss.sweeps])]
        starts = np.searchsorted(sweep_starts, selected, side="right") - 1
        splits = np.where(np.diff(starts) > 0)[0] + 1
        splits = np.r_[0, splits, len(starts)]

        # to local indices
        selected = selected - sweep_starts[starts]

        self._unchanged_lidar_sweeps = ss.sweeps

        for i, j in zip(splits[:-1], splits[1:]):

            s = starts[i]
            sweep = ss.sweeps[s]
            unchanged_sweep = sweep - sweep[selected[i:j]]
            self._unchanged_lidar_sweeps[s] = unchanged_sweep

        return self._unchanged_lidar_sweeps

    @property
    def changed_lidar_sweeps(self):
        if hasattr(self, "_changed_lidar_sweeps"):
            return self._changed_lidar_sweeps

        raws = self.raw_lidar_sweeps
        unchangeds = self.unchanged_lidar_sweeps

        self._changed_lidar_sweeps = []
        for raw, unchanged in zip(raws, unchangeds):
            changed_sweep = raw - unchanged
            self._changed_lidar_sweeps.append(changed_sweep)

        return self._changed_lidar_sweeps

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
            lidar_sweeps = self.raw_lidar_sweeps
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

    # def process_source_masks(self, camera_name):

    #     camera_root = os.path.join(self.cameras_root, camera_name)
    #     sparse_cd_mask_root = os.path.join(camera_root, "sparse_changed_masks")
    #     dense_cd_mask_root = os.path.join(camera_root, "dense_changed_masks")
    #     sparse_cd_mask_src_root = os.path.join(sparse_cd_mask_root, "source")
    #     dense_cd_mask_src_root = os.path.join(dense_cd_mask_root, "source")

    #     os.makedirs(sparse_cd_mask_src_root, exist_ok=True)
    #     os.makedirs(dense_cd_mask_src_root, exist_ok=True)

    #     image_sequence = self.camera_sequence.get_a_camera(camera_name)
    #     seg_indices = self.edited_details["added_segment_indices_of_source"]

    #     for index in range(len(image_sequence)):

    #         name = self._initialize_cam_fnames(camera_name, index)
    #         ind_map_src = self.get_a_sparse_point_indices(camera_name, index)
    #         valid_src = ind_map_src > 0

    #         sparse_change_mask = np.zeros_like(ind_map_src, dtype=bool)
    #         dense_change_mask = np.zeros_like(ind_map_src, dtype=bool)

    #         for indices in seg_indices:
    #             # compare indices and ind_map_src

    #             # sparse mask
    #             sp_mask = np.zeros_like(ind_map_src, dtype=bool)
    #             valid = np.isin(ind_map_src[valid_src], indices)
    #             sp_mask[valid_src] |= valid

    #             # dense mask
    #             de_mask = utils_img.fill_sparse_boolean_by_convex_hull(sp_mask)

    #             sparse_change_mask |= sp_mask
    #             dense_change_mask |= de_mask > 0

    #         file = f"{name}.png"
    #         f_sparse_mask = os.path.join(sparse_cd_mask_src_root, file)
    #         f_dense_mask = os.path.join(dense_cd_mask_src_root, file)
    #         plt.imsave(f_sparse_mask, sparse_change_mask, cmap="gray")
    #         plt.imsave(f_dense_mask, dense_change_mask, cmap="gray")

    # def get_source_sparse_mask(self, camera_name, filename_or_index):
    #     filename = self._initialize_cam_fnames(camera_name, filename_or_index)

    #     camera_root = os.path.join(self.cameras_root, camera_name)
    #     sparse_cd_mask_root = os.path.join(camera_root, "sparse_changed_masks")
    #     sparse_cd_mask_src_root = os.path.join(sparse_cd_mask_root, "source")
    #     filepath = os.path.join(sparse_cd_mask_src_root, f"{filename}.png")

    #     if not os.path.exists(filepath):
    #         raise FileNotFoundError(f"Sparse mask {filepath} does not exist")

    #     return np.all(plt.imread(filepath) > 0, axis=-1)

    # def get_source_dense_mask(self, camera_name, filename_or_index):
    #     filename = self._initialize_cam_fnames(camera_name, filename_or_index)

    #     camera_root = os.path.join(self.cameras_root, camera_name)
    #     dense_cd_mask_root = os.path.join(camera_root, "dense_changed_masks")
    #     dense_cd_mask_src_root = os.path.join(dense_cd_mask_root, "source")
    #     filepath = os.path.join(dense_cd_mask_src_root, f"{filename}.png")

    #     if not os.path.exists(filepath):
    #         raise FileNotFoundError(f"Dense mask {filepath} does not exist")

    #     return np.all(plt.imread(filepath) > 0, axis=-1)

    # def process_visual_source_masks(self, camera_name):

    #     # for debug
    #     camera_root = os.path.join(self.cameras_root, camera_name)
    #     output_dir = os.path.join(camera_root, "visual_source_mask_img")
    #     os.makedirs(output_dir, exist_ok=True)

    #     image_sequence = self.camera_sequence.get_a_camera(camera_name)
    #     segment_indices = self.edited_details[
    #         "added_segment_indices_of_source"
    #     ]

    #     for index in range(len(image_sequence)):

    #         name = self._initialize_cam_fnames(camera_name, index)
    #         depth_img = self.get_a_depth_image(camera_name, index)
    #         ind_map_src = self.get_a_sparse_point_indices(camera_name, index)

    #         valid_src = ind_map_src > 0
    #         overall_dense_mask = np.zeros_like(ind_map_src, dtype=bool)

    #         for indices in segment_indices:
    #             # compare indices and ind_map_src
    #             sp_mask = np.zeros_like(ind_map_src, dtype=bool)
    #             valid = np.isin(ind_map_src[valid_src], indices)
    #             sp_mask[valid_src] |= valid

    #             de_mask = utils_img.fill_sparse_boolean_by_convex_hull(sp_mask)
    #             overall_dense_mask |= de_mask > 0

    #         depth_img = utils_img.overlay_image(
    #             depth_img,
    #             [1, 1, 0],
    #             ratio=0.8,
    #             mask=overall_dense_mask,
    #         )

    #         for indices in segment_indices:
    #             valid = np.isin(ind_map_src[valid_src], indices)
    #             mask = np.zeros_like(ind_map_src, dtype=bool)
    #             mask[valid_src] |= valid

    #             depth_img = utils_img.overlay_image(
    #                 depth_img,
    #                 np.random.rand(3),
    #                 ratio=0,
    #                 mask=mask,
    #             )

    #         f_sparse_depth = os.path.join(output_dir, f"{name}.png")
    #         plt.imsave(f_sparse_depth, depth_img)

    # def process_target_masks(self, camera_name):

    #     camera_root = os.path.join(self.cameras_root, camera_name)
    #     sparse_cd_mask_root = os.path.join(camera_root, "sparse_changed_masks")
    #     dense_cd_mask_root = os.path.join(camera_root, "dense_changed_masks")
    #     sparse_cd_mask_tgt_root = os.path.join(sparse_cd_mask_root, "target")
    #     dense_cd_mask_tgt_root = os.path.join(dense_cd_mask_root, "target")

    #     os.makedirs(sparse_cd_mask_tgt_root, exist_ok=True)
    #     os.makedirs(dense_cd_mask_tgt_root, exist_ok=True)

    #     ori_scene = OriginalScene(self.root)
    #     tgt_inds = self.edited_details["deleted_indices_of_target"]
    #     deleted_xyz = ori_scene.pcd_xyz[tgt_inds]
    #     deleted_ann = self.scene_details["delete"].get("annotations", None)
    #     margin = self.scene_details["delete"].get("margin", 0.0)

    #     deleted_inds = []
    #     if deleted_ann is not None:
    #         results = deleted_ann.is_points_in_bounding_boxes(
    #             deleted_xyz,
    #             margin=margin,
    #             separate=True,
    #         )
    #         for indices in results:
    #             deleted_inds.append(tgt_inds[indices])

    #     deleted_inds = scene_utils.cluster_overlapping_lists(deleted_inds)
    #     deleted_inds = [np.sort(list(ind)) for ind in deleted_inds]

    #     image_sequence = self.camera_sequence.get_a_camera(camera_name)

    #     log_id = image_sequence.log_id
    #     files = argoverse2.list_sweep_files_by_log_id(log_id)
    #     times = [os.path.basename(f).replace(".feather", "") for f in files]
    #     ts = array_data.Timestamps(len(files))
    #     ts.timestamps = times
    #     ts = ts.align_timestamps(image_sequence.timestamps)
    #     files = np.r_[files][np.searchsorted(times, ts.timestamps)]
    #     sweeps = [argoverse2.Sweep(file, coordinate="map") for file in files]

    #     for index in range(len(image_sequence)):

    #         name = self._initialize_cam_fnames(camera_name, index)
    #         depth_map = ori_scene.get_a_sparse_depth(camera_name, index)
    #         ind_map_tgt = ori_scene.get_a_sparse_point_indices(
    #             camera_name, index
    #         )

    #         sweep_depth_map = image_sequence.get_a_depth_map(
    #             index,
    #             sweeps[index].xyz,
    #         )

    #         valid_tgt = ind_map_tgt > 0
    #         sparse_change_mask = np.zeros_like(ind_map_tgt, dtype=bool)
    #         dense_change_mask = np.zeros_like(ind_map_tgt, dtype=bool)

    #         for indices in deleted_inds:
    #             # compare indices and ind_map_src
    #             sp_mask = np.zeros_like(ind_map_tgt, dtype=bool)
    #             valid = np.isin(ind_map_tgt[valid_tgt], indices)

    #             sp_mask[valid_tgt] |= valid
    #             sp_mask = scene_utils.filter_visible(
    #                 depth_map,
    #                 sweep_depth_map,
    #                 sp_mask,
    #                 neighborhood_size=7,
    #                 num_valid_points=3,
    #             )
    #             de_mask = utils_img.fill_sparse_boolean_by_convex_hull(sp_mask)

    #             sparse_change_mask |= sp_mask
    #             dense_change_mask |= de_mask > 0

    #         file = f"{name}.png"
    #         f_sparse_mask = os.path.join(sparse_cd_mask_tgt_root, file)
    #         f_dense_mask = os.path.join(dense_cd_mask_tgt_root, file)
    #         plt.imsave(f_sparse_mask, sparse_change_mask, cmap="gray")
    #         plt.imsave(f_dense_mask, dense_change_mask, cmap="gray")

    # def get_target_sparse_mask(self, camera_name, filename_or_index):
    #     filename = self._initialize_cam_fnames(camera_name, filename_or_index)

    #     camera_root = os.path.join(self.cameras_root, camera_name)
    #     sparse_cd_mask_root = os.path.join(camera_root, "sparse_changed_masks")
    #     sparse_cd_mask_tgt_root = os.path.join(sparse_cd_mask_root, "target")
    #     filepath = os.path.join(sparse_cd_mask_tgt_root, f"{filename}.png")

    #     if not os.path.exists(filepath):
    #         raise FileNotFoundError(f"Sparse mask {filepath} does not exist")

    #     return np.all(plt.imread(filepath) > 0, axis=-1)

    # def get_target_dense_mask(self, camera_name, filename_or_index):

    #     filename = self._initialize_cam_fnames(camera_name, filename_or_index)

    #     camera_root = os.path.join(self.cameras_root, camera_name)
    #     dense_cd_mask_root = os.path.join(camera_root, "dense_changed_masks")
    #     dense_cd_mask_tgt_root = os.path.join(dense_cd_mask_root, "target")
    #     filepath = os.path.join(dense_cd_mask_tgt_root, f"{filename}.png")

    #     if not os.path.exists(filepath):
    #         raise FileNotFoundError(f"Dense mask {filepath} does not exist")

    #     return np.all(plt.imread(filepath) > 0, axis=-1)

    # def process_visual_target_masks(self, camera_name):

    #     ori_scene = OriginalScene(self.root)

    #     # for debug
    #     camera_root = os.path.join(self.cameras_root, camera_name)
    #     root = os.path.join(camera_root, "visual_target_mask_img")
    #     os.makedirs(root, exist_ok=True)

    #     tgt_inds = self.edited_details["deleted_indices_of_target"]
    #     deleted_xyz = ori_scene.pcd_xyz[tgt_inds]
    #     deleted_ann = self.scene_details["delete"].get("annotations", None)
    #     margin = self.scene_details["delete"].get("margin", 0.0)

    #     deleted_inds = []
    #     if deleted_ann is not None:
    #         results = deleted_ann.is_points_in_bounding_boxes(
    #             deleted_xyz,
    #             margin=margin,
    #             separate=True,
    #         )
    #         for indices in results:
    #             deleted_inds.append(tgt_inds[indices])

    #     deleted_inds = scene_utils.cluster_overlapping_lists(deleted_inds)
    #     deleted_inds = [np.sort(list(ind)) for ind in deleted_inds]

    #     image_sequence = self.camera_sequence.get_a_camera(camera_name)

    #     files = argoverse2.list_sweep_files_by_log_id(image_sequence.log_id)
    #     times = [os.path.basename(f).replace(".feather", "") for f in files]
    #     ts = array_data.Timestamps(len(files))
    #     ts.timestamps = times
    #     ts = ts.align_timestamps(image_sequence.timestamps)
    #     files = np.r_[files][np.searchsorted(times, ts.timestamps)]
    #     sweeps = [argoverse2.Sweep(file, coordinate="map") for file in files]

    #     for index in range(len(image_sequence)):

    #         name = self._initialize_cam_fnames(camera_name, index)
    #         depth_img = image_sequence.get_an_image(index) / 255.0
    #         depth_map = ori_scene.get_a_sparse_depth(camera_name, index)
    #         ind_map_tgt = ori_scene.get_a_sparse_point_indices(
    #             camera_name, index
    #         )

    #         sweep_depth_img = image_sequence.get_a_depth_map(
    #             index,
    #             sweeps[index].xyz,
    #         )

    #         valid_tgt = ind_map_tgt > 0
    #         sparse_change_masks = []
    #         dense_change_mask = np.zeros_like(ind_map_tgt, dtype=bool)

    #         for indices in deleted_inds:
    #             # compare indices and ind_map_src
    #             sp_mask = np.zeros_like(ind_map_tgt, dtype=bool)
    #             valid = np.isin(ind_map_tgt[valid_tgt], indices)

    #             sp_mask[valid_tgt] |= valid
    #             sp_mask = scene_utils.filter_visible(
    #                 depth_map,
    #                 sweep_depth_img,
    #                 sp_mask,
    #                 neighborhood_size=7,
    #                 num_valid_points=3,
    #             )
    #             sparse_change_masks.append(sp_mask)
    #             de_mask = utils_img.fill_sparse_boolean_by_convex_hull(sp_mask)

    #             dense_change_mask |= de_mask > 0

    #         depth_img = utils_img.overlay_image(
    #             depth_img,
    #             [1, 1, 0],
    #             ratio=0.5,
    #             mask=dense_change_mask,
    #         )

    #         for sparse_change_mask in sparse_change_masks:
    #             depth_img = utils_img.overlay_image(
    #                 depth_img,
    #                 np.random.rand(3),
    #                 ratio=0.5,
    #                 mask=sparse_change_mask,
    #             )

    #         f_sparse_depth = os.path.join(root, f"{name}.png")
    #         plt.imsave(f_sparse_depth, depth_img)

    # def process_camera(self, camera_name):

    #     super().process_camera(camera_name)
    #     self.process_source_masks(camera_name)
    #     self.process_target_masks(camera_name)
    #     self.process_depth_images(camera_name)
    #     self.process_visual_source_masks(camera_name)
    #     self.process_visual_target_masks(camera_name)
