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


def make_delete_info(log_id, bbox_margin=0.5):

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

    camera_sequence = original_scene.camera_sequence
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

    original_scene = scene_db.OriginalScene(log_id)

    # anchors: the 3d points locate at ground for new points mounting
    anchors = scene_utils.infer_anchor_points_by_av2_log_id(
        log_id,
        area_per_point=area_per_points,
        FOV_cameras=FOV_cameras,
        FOV_max_distance=FOV_max_distance,
        FOV_min_distance=FOV_min_distance,
    )

    # assign orientation to each anchor point
    img_seq = original_scene.camera_sequence.get_a_camera(0)
    poses = img_seq.transformation  # (N, 4, 4)
    poses_xyz = poses[:, :3, 3]  # (N, 3)
    poses_euler = utils.R_to_euler(poses[:, :3, :3])  # (N, 3)

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

    log_ids = argoverse2.SENSOR_MAP.keys()
    if (
        args["selective_logids"] is not None
        and len(args["selective_logids"]) > 0
    ):
        log_ids = args["selective_logids"]

    for log_id in log_ids:

        common.xprint(f"Processing log_id: {log_id}")

        existing_versions = []
        if log_id in scene_db.list_scene_ids():
            existing_versions = scene_db.list_versions_by_scene_id(log_id)

        # ---  prepare delete_info  ---
        delete_scene_name = args["edit_scene_prefix"] + "delete"
        if delete_scene_name in existing_versions and not _overwrite:

            delete_info = scene_db.EditedScene(log_id, delete_scene_name)
            delete_info = delete_info.scene_details

        else:

            delete_info = make_delete_info(
                log_id,
                bbox_margin=args["bbox_params"]["bbox_margin"],
            )

        # ---  prepare add info ---
        add_scene_name = args["edit_scene_prefix"] + "add"
        if add_scene_name in existing_versions and not _overwrite:

            add_info = scene_db.EditedScene(log_id, add_scene_name)
            add_info = add_info.scene_details

        else:

            add_info = make_add_info(
                log_id,
                area_per_points=args["anchor_params"]["area_per_point"],
                FOV_cameras=args["anchor_params"]["FOV_cameras"],
                FOV_max_distance=args["anchor_params"]["FOV_max_distance"],
                FOV_min_distance=args["anchor_params"]["FOV_min_distance"],
                ground_offset=args["anchor_params"]["ground_offset"],
                voxel_size=args["pcd_params"]["voxel_size"],
            )

        original_scene = scene_db.OriginalScene(log_id)

        valid_delete = delete_info["annotations"] is not None
        do_delete = _overwrite or delete_scene_name not in existing_versions
        if valid_delete and do_delete:

            scene_delete = scene_db.EditedScene(log_id, delete_scene_name)
            scene_delete.scene_details = {
                "delete": delete_info,
                "add": {},
                "voxel_size": args["pcd_params"]["voxel_size"],
            }
            scene_delete.scene_pcd = scene_db.apply_change_info_to_target_pcd(
                original_scene.scene_pcd,
                scene_delete.scene_details,
            )

        if _overwrite or add_scene_name not in existing_versions:

            scene_add = scene_db.EditedScene(log_id, add_scene_name)
            scene_add.scene_details = {
                "delete": {},
                "add": add_info,
                "voxel_size": args["pcd_params"]["voxel_size"],
            }
            scene_add.scene_pcd = scene_db.apply_change_info_to_target_pcd(
                original_scene.scene_pcd,
                scene_add.scene_details,
            )

        # if delete_info cannot remove any point, then the overall will be
        # exact same as `add`, thus skipping
        overall_scene_name = args["edit_scene_prefix"] + "overall"
        do_overall = _overwrite or overall_scene_name not in existing_versions
        if valid_delete and do_overall:

            scene_overall = scene_db.EditedScene(log_id, overall_scene_name)
            scene_overall.scene_details = {
                "delete": delete_info,
                "add": add_info,
                "voxel_size": args["pcd_params"]["voxel_size"],
            }
            scene_add.scene_overall = scene_db.apply_change_info_to_target_pcd(
                original_scene.scene_pcd,
                scene_overall.scene_details,
            )

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
