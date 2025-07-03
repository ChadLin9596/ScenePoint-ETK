import os

_pre_cwd = os.path.realpath(os.getcwd())

import argparse
import json
import yaml

import scene_point_etk.argoverse2 as argoverse2
import scene_point_etk.scene_db as scene_db

import common

_dry = False
_overwrite = False


################
# main process #
################


def make_cleaned_pcd(
    log_id,
    moving_objects_categories=[],
    bbox_params={},
    pcd_params={},
):

    bbox_margin = bbox_params.get("bbox_margin", 0.5)
    time_margin = bbox_params.get("time_margin", 1.05)
    voxel_size = pcd_params.get("voxel_size", 0.2)

    ori_annots = argoverse2.Annotations(log_id)
    dyn_annots = ori_annots.filter_by_category(moving_objects_categories)

    # separate original point set to static and moving objects
    sweep_seq = argoverse2.SweepSequence(log_id, coordinate="map")
    mov_sweep_seq = sweep_seq.filtered_by_annotations(
        dyn_annots,
        bbox_margin=bbox_margin,
        time_margin=time_margin,
    )
    static_sweep_seq = sweep_seq - mov_sweep_seq

    gt_pcd = static_sweep_seq.export_to_voxel_grid(voxel_size=voxel_size)
    gt_details = {
        "annotations": dyn_annots,
        "sweeps": static_sweep_seq.sweeps,
        "moving_objects_categories": moving_objects_categories,
        "bbox_margin": bbox_margin,
        "time_margin": time_margin,
        "voxel_size": voxel_size,
    }

    scene = scene_db.OriginalScene(log_id)
    scene.scene_pcd = gt_pcd
    scene.scene_details = gt_details


def main(args):

    if common.VERBOSE:
        print(json.dumps(args, indent=4))

    # list parameters that will be used in the process
    moving_objects_categories = args["dynamic_objects_categories"]
    bbox_params = args["bbox_params"]
    pcd_params = args["pcd_params"]
    selective_logids = args.get("selective_logids", None)

    # replace log_ids with selective_logids if provided
    log_ids = argoverse2.list_log_ids()
    if selective_logids is not None and len(selective_logids) > 0:
        log_ids = selective_logids

    for log_id in log_ids:

        common.xprint(f"Processing log_id: {log_id}")

        if log_id in scene_db.list_scene_ids() and not _overwrite and not _dry:
            common.xprint(">>> skip [make_target_pcd]")

        else:
            common.xprint(">>> [make_target_pcd]")

            make_cleaned_pcd(
                log_id,
                moving_objects_categories=moving_objects_categories,
                bbox_params=bbox_params,
                pcd_params=pcd_params,
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
