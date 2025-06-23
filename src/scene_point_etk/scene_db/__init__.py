import glob
import json
import os

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
