import glob
import json
import os
from . import scene
from .diff_scene import (
    get_deleted_pcd,
    get_added_pcd,
    apply_change_info_to_target_pcd,
    infer_merge_indices,
)

SCENE_ROOT = ""
SCENE_MAP = {}


def set_scene_root(scene_root=None, overwrite=True):

    global SCENE_ROOT, SCENE_MAP

    path_this_file = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(path_this_file, "..", "config.json")

    # set the SENSOR_ROOT from either config.json or the provided patch_root
    if scene_root is not None:
        SCENE_ROOT = scene_root
    else:
        with open(config_path, "r") as f:
            config = json.load(f)
            SCENE_ROOT = config.get("scene_dataset_root", "")

    # update config.json if overwrite
    if scene_root is not None and overwrite:
        with open(config_path, "r") as f:
            config = json.load(f)

        with open(config_path, "w") as f:
            config["scene_dataset_root"] = SCENE_ROOT
            json.dump(config, f, indent=4)

    # build the PATCH_MAP
    SCENE_MAP.clear()
    paths = sorted(glob.glob(os.path.join(SCENE_ROOT, "*")))
    for path in paths:

        dirs = os.listdir(path)

        if "GT" not in dirs:
            continue

        gt_dir = os.path.join(path, "GT")
        if "details.pkl" not in os.listdir(gt_dir):
            continue

        scene_dirs = [os.path.join(path, d) for d in dirs if d != "GT"]
        scene_dirs = [d for d in scene_dirs if os.path.isdir(d)]
        scene_dirs = [d for d in scene_dirs if "details.pkl" in os.listdir(d)]
        scene_dirs = [os.path.basename(d) for d in scene_dirs]

        dir_name = os.path.basename(path)
        SCENE_MAP[dir_name] = {
            "path": path,
            "versions": scene_dirs,
        }


set_scene_root()


def list_scene_ids():
    return sorted(list(SCENE_MAP.keys()))


def list_versions_by_scene_id(scene_id):
    if scene_id not in SCENE_MAP:
        raise ValueError(f"Scene ID '{scene_id}' not found in the scene map.")
    return sorted(SCENE_MAP[scene_id]["versions"])


def get_scene_path_by_id(scene_id):
    if scene_id not in SCENE_MAP:
        raise ValueError(f"Scene ID '{scene_id}' not found in the scene map.")
    return SCENE_MAP[scene_id]["path"]


def list_scene_version_pairs(scene_ids=[], versions=[]):

    if len(scene_ids) == 0:
        scene_ids = list_scene_ids()

    raveled = []
    for scene_id in scene_ids:
        available_versions = list_versions_by_scene_id(scene_id)

        if len(versions) == 0:
            selected_versions = available_versions
        else:
            selected_versions = [
                v for v in versions if v in available_versions
            ]

        for version in selected_versions:
            raveled.append((scene_id, version))

    return raveled


class OriginalScene(scene.OriginalScene):

    def __init__(self, scene_id):
        super().__init__(get_scene_path_by_id(scene_id))


class EditedScene(scene.EditedScene):

    def __init__(self, scene_id, version):
        super().__init__(get_scene_path_by_id(scene_id), version)
