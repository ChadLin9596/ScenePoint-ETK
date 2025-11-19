import glob
import json
import os

import py_utils.pcd as pcd

PATCH_ROOT = ""
PATCH_MAP = {}


def set_patch_root(patch_root=None, overwrite=True):

    global PATCH_ROOT, PATCH_MAP

    path_this_file = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(path_this_file, "..", "config.json")

    # set the SENSOR_ROOT from either config.json or the provided patch_root
    if patch_root is not None:
        PATCH_ROOT = patch_root
    else:
        with open(config_path, "r") as f:
            config = json.load(f)
            PATCH_ROOT = config.get("patch_dataset_root", "")

    # update config.json if overwrite
    if patch_root is not None and overwrite:
        with open(config_path, "r") as f:
            config = json.load(f)

        with open(config_path, "w") as f:
            config["patch_dataset_root"] = PATCH_ROOT
            json.dump(config, f, indent=4)

    # build the PATCH_MAP
    patch_classes = glob.glob(os.path.join(PATCH_ROOT, "*"))
    patch_classes = [i for i in patch_classes if os.path.isdir(i)]
    patch_classes = [os.path.basename(i) for i in patch_classes]

    PATCH_MAP.clear()
    for patch_class in patch_classes:

        paths = glob.glob(os.path.join(PATCH_ROOT, patch_class, "*.pcd"))

        for path in paths:
            name = os.path.basename(path).replace(".pcd", "")
            PATCH_MAP[(patch_class, name)] = path


set_patch_root()

###############################################################################


def list_valid_patch_keys():
    return sorted(list(PATCH_MAP.keys()))


def is_valid_patch_key(key):
    return key in PATCH_MAP


def check_patch_key(key):
    if not is_valid_patch_key(key):
        raise ValueError(f"Key {key} not found")


def get_patch_path_from_key(key):
    return PATCH_MAP[key]


def get_patch_from_key(key):
    path = get_patch_path_from_key(key)
    return pcd.read(path)
