import time

import scene_point_etk.argoverse2 as argoverse2
import scene_point_etk.scene_db as scene_db

import py_utils.utils as utils


def main():

    num_origin_scene = 0
    num_edited_scene = 0

    time_origin_pcd = 0
    time_edited_pcd = 0
    time_camera_map = 0

    prog = utils.ProgressTimer()
    prog.tic(len(scene_db.list_scene_ids()))

    for log_id in scene_db.list_scene_ids():

        origin_scene = scene_db.OriginalScene(log_id)
        num_origin_scene += 1

        details = origin_scene.scene_details
        sweeps = argoverse2.SweepSequence.from_sweeps(details["sweeps"])

        s = time.time()
        scene = sweeps.export_to_voxel_grid(
            voxel_size=details.get("voxel_size", 0.2),
            skip_color=False,
            return_details=False,
        )
        origin_scene.scene_pcd = scene
        time_origin_pcd += time.time() - s

        versions = scene_db.list_versions_by_scene_id(log_id)
        for version in versions:
            num_edited_scene += 1
            edited_scene = scene_db.EditedScene(log_id, version)

            origin_scene_pcd = origin_scene.scene_pcd
            details = edited_scene.scene_details

            s = time.time()
            scene = scene_db.apply_change_info_to_target_pcd(
                origin_scene_pcd,
                details,
                return_details=False,
            )
            edited_scene.scene_pcd = scene
            time_edited_pcd += time.time() - s

        prog.toc()

    print("num_origin_scene", num_origin_scene)
    print("num_edited_scene", num_edited_scene)
    print("time_origin_pcd", time_origin_pcd)
    print("time_edited_pcd", time_edited_pcd)
    print("time_camera_map", time_camera_map)


if __name__ == "__main__":
    main()
