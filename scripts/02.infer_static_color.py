import argparse
import numpy as np
import scene_point_etk.argoverse2 as argoverse2
import scene_point_etk.scene_db as scene_db


def main(dry=False):

    log_ids = scene_db.list_scene_ids()

    for log_id in log_ids:

        scene = scene_db.OriginalScene(log_id)
        if np.any(scene.pcd_color != 0):
            print("skipping log_id:", log_id)
            continue

        print("processing log_id:", log_id)

        scene_details = scene.scene_details
        sweeps = argoverse2.SweepSequence.from_sweeps(scene_details["sweeps"])
        scene_pcd = sweeps.export_to_voxel_grid(
            voxel_size=scene_details["voxel_size"],
            skip_color=False,
        )
        scene.scene_pcd = scene_pcd

        if dry:
            break


def parser_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true", help="Dry run")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parser_args()
    dry = args.dry
    main(dry)
