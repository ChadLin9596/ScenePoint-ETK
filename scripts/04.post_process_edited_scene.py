import os

_pre_cwd = os.path.realpath(os.getcwd())

import argparse
import json
import yaml
import shutil

import numpy as np

import scene_point_etk.argoverse2 as argoverse2
import scene_point_etk.scene_db as scene_db
import scene_point_etk.patch_db as patch_db
import scene_point_etk.utils as scene_utils
import py_utils.utils as utils

import common

_dry = False
_overwrite = False


def main(args):

    if common.VERBOSE:
        print(json.dumps(args, indent=4))

    ground_filter_threshold = args["ground_filter_params"]["threshold"]

    for log_id in scene_db.list_scene_ids():

        for version in scene_db.list_versions_by_scene_id(log_id):

            origin_scene = scene_db.OriginalScene(log_id)
            edited_scene = scene_db.EditedScene(log_id, version)

            if len(edited_scene.added_pcds) == 0:
                print(f"Scene {log_id} -> {version} has no added patches.")
                continue

            # get the non-ground points
            _, ground_ind = scene_utils.infer_ground_points_by_av2_log_id(
                log_id,
                origin_scene.pcd_xyz,
                threshold=ground_filter_threshold,
                return_indices=True,
            )
            non_ground = np.ones(len(origin_scene.pcd_xyz), dtype=bool)
            non_ground[ground_ind] = False

            # if any non_ground points are in the added bounding boxes
            # remove the added pcd
            deleted_index = []
            for n, bbox in enumerate(edited_scene.deleted_bounding_boxes):
                M = utils.points_in_a_bounding_box(
                    origin_scene.pcd_xyz[non_ground],
                    bbox,
                )
                if np.any(M):
                    deleted_index.append(n)

            if len(deleted_index) == 0:
                continue

            scene_details = edited_scene.scene_details
            add_info = scene_details.pop("add")

            if "merge_indices" not in add_info:
                print("infer merge indices")
                add_info["merge_indices"] = scene_db.infer_merge_indices(
                    origin_scene.scene_pcd,
                    add_info,
                )

            new_add = {
                "patches": [],
                "anchor_xyzs": [],
                "anchor_eulers": [],
                "z_offset": add_info.get("z_offset", 0.0),
                "voxel_size": add_info.get("voxel_size", 0.2),
            }

            for n in range(len(add_info["patches"])):

                if n in deleted_index:
                    continue

                patch = add_info["patches"][n]
                xyz = add_info["anchor_xyzs"][n]
                euler = add_info["anchor_eulers"][n]
                new_add["patches"].append(patch)
                new_add["anchor_xyzs"].append(xyz)
                new_add["anchor_eulers"].append(euler)

            new_add["anchor_xyzs"] = np.array(new_add["anchor_xyzs"])
            new_add["anchor_eulers"] = np.array(new_add["anchor_eulers"])
            new_add["merge_indices"] = {
                p: add_info["merge_indices"][p] for p in new_add["patches"]
            }
            scene_details["add"] = new_add

            old_path = edited_scene.details_filepath
            new_path = old_path.replace("details.pkl", "details.old.pkl")
            shutil.copyfile(old_path, new_path)
            edited_scene.scene_details = scene_details

            a = len(add_info["patches"])
            b = len(new_add["patches"])
            print(f"Scene {log_id} -> {version} from {a} to {b} patches.")


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
