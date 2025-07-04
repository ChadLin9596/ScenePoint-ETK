import glob
import json
import os

import numpy as np
import pyarrow.feather as feather

SENSOR_ROOT = ""
SENSOR_MAP = {}


def _get_log_id(path):

    log_id = os.path.basename(path)
    log_id = log_id.replace("-", "")
    log_id = log_id.split("_")[0]
    return log_id


def set_sensor_root(sensor_root=None, overwrite=True):

    global SENSOR_ROOT, SENSOR_MAP

    path_this_file = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(path_this_file, "..", "config.json")

    # set the SENSOR_ROOT from either config.json or the provided sensor_root
    if sensor_root is not None:
        SENSOR_ROOT = sensor_root
    else:
        with open(config_path, "r") as f:
            config = json.load(f)
            SENSOR_ROOT = config.get("sensor_dataset_root", "")

    # update config.json if overwrite
    if sensor_root is not None and overwrite:
        with open(config_path, "r") as f:
            config = json.load(f)

        with open(config_path, "w") as f:
            config["sensor_dataset_root"] = SENSOR_ROOT
            json.dump(config, f, indent=4)

    # build the SENSOR_MAP
    paths = glob.glob(os.path.join(SENSOR_ROOT, "*"))
    for path in paths:
        log_id = _get_log_id(path)
        SENSOR_MAP[log_id] = path


set_sensor_root()

###############################################################################


def check_log_id(log_id):
    if log_id not in SENSOR_MAP:
        raise ValueError(f"Log id {log_id} not found")


def check_city(city):

    path_this_file = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(path_this_file, "city_2_log_id.json"), "r") as fd:
        data = json.load(fd)

    if city not in data:
        raise ValueError(f"City {city} not found")


def list_log_ids():
    return sorted(list(SENSOR_MAP.keys()))


def list_log_ids_by_mode(mode="test"):

    path_this_file = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(path_this_file, "log_splits.json"), "r") as fd:
        data = json.load(fd)

    if mode not in data:
        raise ValueError(f"Mode {mode} not found")

    return sorted(data[mode])


def list_cities():

    path_this_file = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(path_this_file, "city_2_log_id.json"), "r") as fd:
        data = json.load(fd)

    return sorted(list(data.keys()))


def list_cameras_by_log_id(log_id):

    check_log_id(log_id)

    # some log doesn't have stereo_front_left and stereo_front_right
    # return [
    #     "ring_front_center",
    #     "ring_front_left",
    #     "ring_front_right",
    #     "ring_rear_left",
    #     "ring_rear_right",
    #     "ring_side_left",
    #     "ring_side_right",
    #     "stereo_front_left",
    #     "stereo_front_right",
    # ]

    return get_intrinsic_by_log_id(log_id)["sensor_name"].tolist()


def list_sweep_files_by_log_id(log_id):

    check_log_id(log_id)

    root = get_root_from_log_id(log_id)
    files = glob.glob(os.path.join(root, "sensors", "lidar", "*.feather"))
    files = sorted(files)
    return files


def get_root_from_log_id(log_id):
    return SENSOR_MAP[log_id]


def get_log_ids_from_city(city):

    path_this_file = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(path_this_file, "city_2_log_id.json"), "r") as fd:
        data = json.load(fd)

    return data.get(city, [])


def get_city_from_log_id(log_id):

    check_log_id(log_id)

    root = get_root_from_log_id(log_id)
    fs = glob.glob(os.path.join(root, "map", "log_map_archive_*.json"))

    if len(fs) != 1:
        raise OSError

    city = fs[0].split("__")[-1]
    city = city.split("_")[0]

    return city


def get_poses_by_log_id(log_id):

    check_log_id(log_id)

    root = get_root_from_log_id(log_id)
    path = os.path.join(root, "city_SE3_egovehicle.feather")
    poses = feather.read_feather(path)

    R = {
        "timestamp": poses["timestamp_ns"].to_numpy(),
        "xyz": poses[["tx_m", "ty_m", "tz_m"]].to_numpy(),
        "q": poses[["qx", "qy", "qz", "qw"]].to_numpy(),
    }

    return R


def get_extrinsic_by_log_id(log_id):

    check_log_id(log_id)

    root = get_root_from_log_id(log_id)
    path = os.path.join(root, "calibration", "egovehicle_SE3_sensor.feather")
    extrinsic = feather.read_feather(path)

    return extrinsic


def get_intrinsic_by_log_id(log_id):

    check_log_id(log_id)

    root = get_root_from_log_id(log_id)
    path = os.path.join(root, "calibration", "intrinsics.feather")
    intrinsic = feather.read_feather(path)

    return intrinsic


def _get_ground_height_img_by_log_id(log_id):

    check_log_id(log_id)

    root = get_root_from_log_id(log_id)
    path = os.path.join(root, "map", "*_ground_height_surface____*.npy")
    path = glob.glob(path)

    assert len(path) == 1
    path = path[0]

    ground_height = np.load(path)

    return ground_height


def _get_img_Sim2_city(log_id):

    check_log_id(log_id)

    root = get_root_from_log_id(log_id)
    path = os.path.join(root, "map", "*___img_Sim2_city.json")
    path = glob.glob(path)

    assert len(path) == 1
    path = path[0]

    with open(path, "r") as fd:
        data = json.load(fd)

    info = {
        "R": np.array(data["R"]).reshape(2, 2),
        "t": np.array(data["t"]).reshape(2),
        "s": float(data["s"]),
    }
    return info


def get_ground_height_points_by_log_id(log_id):

    ground_height_img = _get_ground_height_img_by_log_id(log_id)
    img_Sim2_city = _get_img_Sim2_city(log_id)

    H, W = ground_height_img.shape

    # create a grid of (x, y, z) points
    xyz = np.empty((H, W, 3))
    xyz[:, :, 0] = np.arange(W)[None, :] + 0.5
    xyz[:, :, 1] = np.arange(H)[:, None] + 0.5
    xyz[:, :, 2] = ground_height_img

    # filter out nan values
    xyz = xyz.reshape(-1, 3)
    I = np.isnan(xyz[:, 2])
    xyz = xyz[~I]

    # transform points to city coordinate
    xy = xyz[:, :2]
    xy = xy / img_Sim2_city["s"]
    xy = xy - img_Sim2_city["t"]
    xy = np.sum(xy[:, None, :] * img_Sim2_city["R"].T, axis=-1)
    xyz[:, :2] = xy

    return xyz


def get_ground_height_by_points(log_id, xyz, filled_value=np.nan):

    ground_height_points = _get_ground_height_img_by_log_id(log_id)
    img_Sim2_city = _get_img_Sim2_city(log_id)

    xy = xyz[:, :2]

    xy = np.sum(xy[:, None, :] * img_Sim2_city["R"], axis=-1)
    xy = xy + img_Sim2_city["t"]
    xy = xy * img_Sim2_city["s"]

    xy = np.floor(xy).astype(np.int64)

    z = np.full(len(xy), filled_value)

    H, W = ground_height_points.shape

    # fmt: off
    valid_I = (
        (xy[:, 0] >= 0) & (xy[:, 0] < W) &
        (xy[:, 1] >= 0) & (xy[:, 1] < H)
    )
    # fmt: on

    z[valid_I] = ground_height_points[xy[valid_I, 1], xy[valid_I, 0]]
    return z


class ArgoMixin:

    # any path under get_root_from_log_id(log_id)
    _path = None

    @property
    def root(self):

        ps = self._path.split(os.sep)
        for i, p in enumerate(ps):
            # assume the root is the only folder exceeding 32 characters
            if len(p) < 32:
                continue
            return os.path.join(os.sep, *ps[: i + 1])
        return None

    @property
    def log_id(self):

        root = self.root
        if root is None:
            return None

        return _get_log_id(root)

    @property
    def city(self):

        log_id = self.log_id
        return get_city_from_log_id(log_id)
