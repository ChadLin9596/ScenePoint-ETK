import os
import pickle
import shutil
import matplotlib.pyplot as plt

import numpy as np

import scene_point_etk.argoverse2 as argoverse2

import py_utils.pcd as pcd
import py_utils.utils as utils
import py_utils.utils_img as utils_img


class Scene:
    """
    ├── <Scene ID 00>
    │   │
    │   └── <version name>
    │       ├── details.pkl
    │       ├── scene.pcd
    │       └── cameras
    │           ├── cam_sequence.pkl
    │           ├── <camera name 1>
    │           │   ├── sparse_depths
    │           │   │   ├── <depth 1>.npy
    │           │   │   └── ...
    │           │   ├── sparse_point_map
    │           │   │   ├── <point map 1>.npy
    │           │   │   └── ...
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

        self.scene_root = os.path.join(scene_root, version)
        self.version = version
        self.cameras_root = os.path.join(self.scene_root, "cameras")
        os.makedirs(self.cameras_root, exist_ok=True)

    @property
    def scene_filepath(self):
        return os.path.join(self.scene_root, "scene.pcd")

    @property
    def details_filepath(self):
        return os.path.join(self.scene_root, "details.pkl")

    @property
    def camera_sequence_filepath(self):
        return os.path.join(self.cameras_root, "cam_sequence.pkl")

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
    def scene_details(self):
        if not os.path.exists(self.details_filepath):
            return None

        with open(self.details_filepath, "rb") as fd:
            return pickle.load(fd)

    @scene_details.setter
    def scene_details(self, details):
        with open(self.details_filepath, "wb") as fd:
            pickle.dump(details, fd)

    @property
    def camera_sequence(self):

        if hasattr(self, "_camera_sequence"):
            return self._camera_sequence

        # load camera sequence from file
        path = self.camera_sequence_filepath
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} does not exist")

        with open(path, "rb") as f:
            camera_sequence = pickle.load(f)

        self._camera_sequence = camera_sequence
        return self._camera_sequence

    @camera_sequence.setter
    def camera_sequence(self, camera_sequence):

        if not isinstance(camera_sequence, argoverse2.CameraSequence):
            raise TypeError("Expected CameraSequence object")

        path = self.camera_sequence_filepath
        if os.path.exists(path):
            raise FileExistsError(f"{path} already exists")

        self._camera_sequence = camera_sequence
        with open(self.camera_sequence_filepath, "wb") as f:
            pickle.dump(camera_sequence, f)

    @property
    def cameras(self):
        return self.camera_sequence.list_cameras()

    def process_camera(self, camera_name):

        camera_root = os.path.join(self.cameras_root, camera_name)
        depths_root = os.path.join(camera_root, "sparse_depths")
        point_map_root = os.path.join(camera_root, "sparse_point_map")
        point_indices_root = os.path.join(camera_root, "sparse_point_indices")
        os.makedirs(depths_root, exist_ok=True)
        os.makedirs(point_map_root, exist_ok=True)
        os.makedirs(point_indices_root, exist_ok=True)

        # get the essential data
        # if they are not already set, error will be raised
        img_seq = self.camera_sequence.get_a_camera(camera_name)
        names = img_seq.timestamps  # (N, )
        intrinsic = img_seq.intrinsic  # (3, 3)
        extrinsics = img_seq.extrinsic  # (N, 4, 4)

        scene = self.scene_pcd
        scene_xyz = np.vstack([scene["x"], scene["y"], scene["z"]]).T

        N = len(img_seq)
        prog = utils.ProgressTimer(prefix=f"Processing camera {camera_name} ")
        prog.tic(N)
        for name, extrinsic in zip(names, extrinsics):

            depth_map, point_map, details = utils_img.points_to_depth_image(
                points=scene_xyz,
                intrinsic=intrinsic,
                extrinsic=extrinsic,
                H=img_seq.figsize[0],
                W=img_seq.figsize[1],
                invalid_value=-1,
                min_distance=0,
                max_distance=np.inf,
                return_details=True,
                other_attrs=[scene_xyz],  # assign scene_xyz to get point map
            )

            # save sparse point indices
            u, v = details["uv"]
            FOV_mask = details["FOV_mask"]
            indices_map = np.full(img_seq.figsize, -1, dtype=np.int32)
            indices_map[u, v] = np.arange(len(scene_xyz))[FOV_mask]

            name = f"{name}.npy"
            np.save(os.path.join(depths_root, name), depth_map)
            np.save(os.path.join(point_map_root, name), point_map)
            np.save(os.path.join(point_indices_root, name), indices_map)

            prog.toc()

    def export(self, output_dir):

        version = os.path.basename(self.scene_root)
        scene_name = os.path.basename(os.path.dirname(self.scene_root))

        output_root = os.path.join(output_dir, scene_name, version)
        os.makedirs(output_root, exist_ok=True)
        os.makedirs(os.path.join(output_root, "cameras"), exist_ok=True)

        shutil.copy(
            self.details_filepath, os.path.join(output_root, "details.pkl")
        )
        shutil.copy(
            self.camera_sequence_filepath,
            os.path.join(output_root, "cameras", "cam_sequence.pkl"),
        )


class OriginalScene(Scene):
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

    def process_camera(self, camera_name):

        super().process_camera(camera_name)

        camera_root = os.path.join(self.cameras_root, camera_name)
        images_root = os.path.join(camera_root, "images")
        os.makedirs(images_root, exist_ok=True)

        img_seq = self.camera_sequence.get_a_camera(camera_name)
        names = img_seq.timestamps
        H, W = img_seq.figsize

        N = len(img_seq)
        prog = utils.ProgressTimer(prefix=f"Saving {camera_name} {H}x{W} img ")
        prog.tic(N)

        for index in range(N):
            name = f"{names[index]}.png"
            img = img_seq.get_an_image(index)
            plt.imsave(os.path.join(images_root, name), img)
            prog.toc()


class EditedScene(Scene):
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
    │           ├── cam_sequence.pkl  <- soft link to GT's camera_sequence.pkl
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

    @property
    def camera_sequence_filepath(self):
        return os.path.join(
            self.scene_root, "GT", "cameras", "cam_sequence.pkl"
        )

    def process_camera(self, camera_name):

        super().process_camera(camera_name)

        camera_root = os.path.join(self.cameras_root, camera_name)
        sparse_cd_mask_root = os.path.join(camera_root, "sparse_changed_masks")
        dense_cd_mask_root = os.path.join(camera_root, "dense_changed_masks")
        sparse_cd_mask_src_root = os.path.join(sparse_cd_mask_root, "source")
        sparse_cd_mask_tgt_root = os.path.join(sparse_cd_mask_root, "target")
        dense_cd_mask_src_root = os.path.join(dense_cd_mask_root, "source")
        dense_cd_mask_tgt_root = os.path.join(dense_cd_mask_root, "target")

        os.makedirs(sparse_cd_mask_src_root, exist_ok=True)
        os.makedirs(sparse_cd_mask_tgt_root, exist_ok=True)
        os.makedirs(dense_cd_mask_src_root, exist_ok=True)
        os.makedirs(dense_cd_mask_tgt_root, exist_ok=True)
