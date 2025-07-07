import scene_point_etk.scene_db as scene_db
import py_utils.utils as utils

import numpy as np


def main():

    log_and_version = []
    for log_id in log_ids:
        versions = scene_db.list_versions_by_scene_id(log_id)
        for version in versions:
            log_and_version.append((log_id, version))

    num_raw_sweep_points = 0

    num_raw_images = 0
    num_raw_sweep = 0

    num_t1_voxelized_points = 0
    num_t1_voxelized_points2 = 0
    num_t0_voxelized_points = 0

    num_t1_scene = 0
    num_t0_scene = 0

    num_raw_unchanged_sweep_points = 0
    num_raw_changed_sweep_points = 0
    num_t1_voxelized_changed_mask = 0
    num_t0_voxelized_changed_mask = 0

    num_added_bounding_boxes = 0
    num_deled_bounding_boxes = 0

    log_ids = scene_db.list_scene_ids()

    prog = utils.ProgressTimer()
    prog.tic(len(log_ids))

    for log_id in log_ids:
        origin_scene = scene_db.OriginalScene(log_id)

        sweeps = origin_scene.raw_lidar_sweeps
        num_raw_sweep += len(sweeps)
        num_raw_sweep_points += sum([len(s) for s in sweeps])
        num_t1_voxelized_points += len(origin_scene.scene_pcd)
        num_t1_scene += 1

        cam_seq = origin_scene.camera_sequences
        for camera in cam_seq.cameras:
            num_raw_images += len(cam_seq.get_a_camera(camera))

        for version in scene_db.list_versions_by_scene_id(log_id):
            edited_scene = scene_db.EditedScene(log_id, version)
            num_t0_voxelized_points += len(edited_scene.scene_pcd)
            num_t0_scene += 1

            num_t1_voxelized_points2 += len(origin_scene.scene_pcd)

            num_raw_unchanged_sweep_points += sum(
                [len(i) for i in edited_scene.unchanged_lidar_sweeps]
            )
            num_raw_changed_sweep_points += sum(
                [len(i) for i in edited_scene.changed_lidar_sweeps]
            )

            num_t1_voxelized_changed_mask += np.unique(
                np.hstack(edited_scene.deleted_indices)
            )

            num_t0_voxelized_changed_mask += np.unique(
                np.hstack(edited_scene.added_indices)
            )

            num_added_bounding_boxes += len(
                edited_scene.added_bounding_boxes()
            )
            num_deled_bounding_boxes += len(
                edited_scene.deleted_bounding_boxes()
            )

        prog.toc()

        print("num_raw_sweep_points", num_raw_sweep_points)
        print("num_raw_images", num_raw_images)
        print("num_raw_sweep", num_raw_sweep)
        print("num_t1_voxelized_points", num_t1_voxelized_points)
        print("num_t1_voxelized_points2", num_t1_voxelized_points2)
        print("num_t0_voxelized_points", num_t0_voxelized_points)
        print("num_t1_scene", num_t1_scene)
        print("num_t0_scene", num_t0_scene)
        print("num_raw_unchanged_sweep_points", num_raw_unchanged_sweep_points)
        print("num_raw_changed_sweep_points", num_raw_changed_sweep_points)
        print("num_t1_voxelized_changed_mask", num_t1_voxelized_changed_mask)
        print("num_t0_voxelized_changed_mask", num_t0_voxelized_changed_mask)
        print("num_added_bounding_boxes", num_added_bounding_boxes)
        print("num_deled_bounding_boxes", num_deled_bounding_boxes)


if __name__ == "__main__":
    main()
