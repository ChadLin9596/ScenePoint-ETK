import os
import numpy as np
from PIL import Image
from .scene import EditedScene, OriginalScene


class SceneImageDataset(EditedScene):

    VALID_KEYs = [
        # GT image
        "target_image",
        # GT pcd scene
        "target_depths",
        "target_points",
        "target_indices",
        "target_sparse_changed_masks",
        "target_dense_changed_masks",
        # <version> pcd scene
        "source_depths",
        "source_points",
        "source_indices",
        "source_sparse_changed_masks",
        "source_dense_changed_masks",
    ]

    def __init__(self, scene_root, version, return_keys=[]):

        super().__init__(scene_root, version)

        for key in return_keys:
            if key not in self.VALID_KEYs:
                msg = f"Invalid key: {key}. Valid keys are: {self.VALID_KEYs}"
                raise ValueError(msg)

        self.return_keys = return_keys
        self.samples = self._collect_samples()
        self.target_scene = OriginalScene(scene_root)
        self.source_scene = EditedScene(scene_root, version)

    def _initialize_paths(self, camera, filename):

        f = os.path.join
        t_prefix = os.path.join(self.target_scene.cameras_root, camera)
        s_prefix = os.path.join(self.source_scene.cameras_root, camera)

        scm = "sparse_changed_masks"
        dcm = "dense_changed_masks"

        png = str(filename) + ".png"
        npy = str(filename) + ".npy"

        R = {
            "target_image": f(t_prefix, "images", png),
            "target_depths": f(t_prefix, "sparse_depths", npy),
            "target_points": f(t_prefix, "sparse_point_map", npy),
            "target_indices": f(t_prefix, "sparse_point_indices", npy),
            "target_sparse_changed_masks": f(s_prefix, scm, "target", png),
            "target_dense_changed_masks": f(s_prefix, dcm, "target", png),
            "source_depths": f(s_prefix, "sparse_depths", npy),
            "source_points": f(s_prefix, "sparse_point_map", npy),
            "source_indices": f(s_prefix, "sparse_point_indices", npy),
            "source_sparse_changed_masks": f(s_prefix, scm, "source", png),
            "source_dense_changed_masks": f(s_prefix, dcm, "source", png),
        }

        return R

    def _collect_samples(self):

        samples = []

        camera_sequence = self.source_scene.camera_sequence
        for camera in camera_sequence.list_cameras():

            filenames = camera_sequence.get_a_camera(camera).timestamps
            for filename in filenames:

                paths = self._initialize_paths(camera, filename)
                sample = [paths[key] for key in self.return_keys]
                samples.append(sample)

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        paths = self.samples[idx]
        data = {}

        for key, path in zip(self.return_keys, paths):

            if key == "target_image":
                data[key] = np.array(Image.open(path).convert("RGB")) / 255.0
                continue

            if os.path.splitext(path)[1] == ".png":
                data[key] = np.array(Image.open(path).convert("L"))
                continue

            if os.path.splitext(path)[1] == ".npy":
                data[key] = np.load(path)
                continue

            raise ValueError(f"Unsupported key: {key}")

        return data
