import os

_pre_cwd = os.path.realpath(os.getcwd())

import argparse
import json
import yaml

import numpy as np

import scene_point_etk.argoverse2 as argoverse2
import scene_point_etk.scene_db as scene_db
import scene_point_etk.patch_db as patch_db
import scene_point_etk.utils as scene_utils
import py_utils.utils as utils

import common

_dry = False
_overwrite = False


################
# main process #
################


def make_delete_info(
    log_id,
    bbox_margin=0.5,
    FOV_cameras=["ring_front_left"],
):

    # initiate delete info
    delete_info = {"annotations": None, "margin": bbox_margin, "indices": []}

    original_scene = scene_db.OriginalScene(log_id)

    original_annotations = argoverse2.Annotations(log_id)
    dynamic_annotations = original_scene.scene_details["annotations"]
    static_annotations = original_annotations - dynamic_annotations

    if len(static_annotations) == 0:
        return delete_info

    # filter out annotations that are out of FOV
    bbox_vertices = static_annotations.bounding_box_vertices(bbox_margin)
    bbox_vertices = bbox_vertices.reshape(-1, 3)

    camera_sequence = argoverse2.CameraSequence(log_id, cameras=FOV_cameras)
    point_masks = camera_sequence.FOV_mask(
        bbox_vertices,
        min_distance=0,
        max_distance=np.inf,
        verbose=common.VERBOSE,
    )
    bbox_masks = point_masks.reshape(-1, 8)
    bbox_masks = np.all(bbox_masks, axis=-1)
    static_annotations = static_annotations[bbox_masks]

    # testify any removed points
    indices = static_annotations.is_points_in_bounding_boxes(
        original_scene.pcd_xyz,
        margin=bbox_margin,
    )
    if len(indices) > 0:
        delete_info["annotations"] = static_annotations

    return delete_info


def make_add_info(
    log_id,
    area_per_points=40,
    FOV_cameras=["ring_front_left"],
    FOV_max_distance=10.0,
    FOV_min_distance=0.0,
    ground_offset=0.1,
    voxel_size=0.2,
):

    # anchors: the 3d points locate at ground for new points mounting
    anchors = scene_utils.infer_anchor_points_by_av2_log_id(
        log_id,
        area_per_point=area_per_points,
        FOV_cameras=FOV_cameras,
        FOV_max_distance=FOV_max_distance,
        FOV_min_distance=FOV_min_distance,
    )

    # assign orientation to each anchor point
    poses = argoverse2.get_poses_by_log_id(log_id)
    poses_xyz = poses["xyz"]
    poses_euler = utils.R_to_euler(utils.Q_to_R(poses["q"]))

    # get the closest position to assign the yaw angle
    d = anchors[:, None, :2] - poses_xyz[:, :2]
    d = np.linalg.norm(d, axis=-1)
    I = np.argmin(d, axis=-1)
    yaw_angles = poses_euler[I, 2]

    # initialize anchors with yaw angles
    anchors_euler = np.zeros((len(anchors), 3))
    anchors_euler[:, 2] = yaw_angles

    # assign patches to each anchor
    N = len(anchors)
    patch_keys = patch_db.list_valid_patch_keys()
    selective_patches = np.random.randint(0, len(patch_keys), N)
    selective_patches = [patch_keys[i] for i in selective_patches]

    # form the change_info/
    add_info = {
        "patches": selective_patches,
        "anchor_xyzs": anchors,
        "anchor_eulers": anchors_euler,
        "z_offset": ground_offset,
        "voxel_size": voxel_size,
    }

    return add_info


def main(args):

    if common.VERBOSE:
        print(json.dumps(args, indent=4))

    # list parameters that will be used in the process
    bbox_margin = args["bbox_params"]["bbox_margin"]
    area_per_points = args["anchor_params"]["area_per_point"]
    FOV_cameras = args["anchor_params"]["FOV_cameras"]
    FOV_max_distance = args["anchor_params"]["FOV_max_distance"]
    FOV_min_distance = args["anchor_params"]["FOV_min_distance"]
    ground_offset = args["anchor_params"]["ground_offset"]
    voxel_size = args["pcd_params"]["voxel_size"]
    selective_logids = args.get("selective_logids", None)
    edited_scene_prefix = args.get("edit_scene_prefix", "edited.")

    # replace log_ids with selective_logids if provided
    log_ids = argoverse2.SENSOR_MAP.keys()
    if selective_logids is not None and len(selective_logids) > 0:
        log_ids = selective_logids

    for log_id in log_ids:

        common.xprint(f"Processing log_id: {log_id}")

        existing_versions = []
        if log_id in scene_db.list_scene_ids():
            existing_versions = scene_db.list_versions_by_scene_id(log_id)

        delete_scene_name = edited_scene_prefix + "delete"
        add_scene_name = edited_scene_prefix + "add"
        overall_scene_name = edited_scene_prefix + "overall"

        del_exist = delete_scene_name in existing_versions
        add_exist = add_scene_name in existing_versions
        overall_exist = overall_scene_name in existing_versions

        # ---  prepare delete_info  ---
        if del_exist and not _overwrite:

            delete_info = scene_db.EditedScene(log_id, delete_scene_name)
            delete_info = delete_info.scene_details["delete"]

        else:

            delete_info = make_delete_info(
                log_id,
                bbox_margin=bbox_margin,
                FOV_cameras=FOV_cameras,
            )

        # ---  prepare add info ---
        if add_exist and not _overwrite:

            add_info = scene_db.EditedScene(log_id, add_scene_name)
            add_info = add_info.scene_details["add"]

        else:

            add_info = make_add_info(
                log_id,
                area_per_points=area_per_points,
                FOV_cameras=FOV_cameras,
                FOV_max_distance=FOV_max_distance,
                FOV_min_distance=FOV_min_distance,
                ground_offset=ground_offset,
                voxel_size=voxel_size,
            )

        # ---  process the scene ---

        valid_delete = delete_info["annotations"] is not None
        valid_add = len(add_info["patches"]) > 0
        valid_overall = valid_add and valid_delete

        do_delete = _overwrite or not del_exist
        if valid_delete and do_delete:

            scene_delete = scene_db.EditedScene(log_id, delete_scene_name)
            scene_delete.scene_details = {
                "delete": delete_info,
                "add": {},
                "voxel_size": voxel_size,
            }
            # it will automatically write the point cloud
            scene_delete.scene_pcd

        do_add = _overwrite or not add_exist
        if valid_add and do_add:

            scene_add = scene_db.EditedScene(log_id, add_scene_name)
            scene_add.scene_details = {
                "delete": {},
                "add": add_info,
                "voxel_size": voxel_size,
            }
            # it will automatically write the point cloud
            scene_add.scene_pcd

        # if delete_info cannot remove any point, then the overall will be
        # exact same as `add`, thus skipping
        do_overall = _overwrite or not overall_exist
        if valid_overall and do_overall:

            scene_overall = scene_db.EditedScene(log_id, overall_scene_name)
            scene_overall.scene_details = {
                "delete": delete_info,
                "add": add_info,
                "voxel_size": voxel_size,
            }
            # it will automatically write the point cloud
            scene_overall.scene_pcd

        common.xprint(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        if _dry:
            break


def parser_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="YAML configuration file")
    args = parser.parse_args()

    config = args.config
    config = (
        config
        if config == os.path.abspath(config)
        else os.path.join(_pre_cwd, args.config)
    )

    with open(config, "r") as fd:
        args = yaml.safe_load(fd)

    return args


if __name__ == "__main__":

    args = parser_args()

    env = args["miscellaneous"]

    _dry = env["dry"]
    _overwrite = env["overwrite"]
    common.VERBOSE = env["verbose"]

    main(args)
