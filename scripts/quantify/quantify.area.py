import numpy as np

import scene_point_etk.scene_db as scene_db
import scene_point_etk.utils as scene_utils

import py_utils.utils as utils
import py_utils.utils_segmentation as utils_segmentation


def main():

    num_scene = 0
    total_area = 0.0
    total_traj = 0.0

    prog = utils.ProgressTimer()
    prog.tic(len(scene_db.list_scene_ids()))

    for scene_id in scene_db.list_scene_ids():

        origin_scene = scene_db.OriginalScene(scene_id)

        camera_seq = origin_scene.camera_sequence
        traj = camera_seq.get_a_camera("ring_front_left").xyz
        traj = np.sum((traj[:-1] - traj[1:]) ** 2, axis=1)
        traj = np.sqrt(traj)
        total_traj += np.sum(traj)

        log_id = camera_seq.log_id
        _, ground_ind = scene_utils.infer_ground_points_by_av2_log_id(
            log_id,
            origin_scene.pcd_xyz,
            threshold=0.5,
            return_indices=True,
        )

        ground_pcd = origin_scene.scene_pcd[ground_ind]
        ground_xy = ground_pcd["center"][:, :2]

        # no need to resort the points, they are already sorted
        splits = np.any(np.diff(ground_xy, axis=0) != 0, axis=1)
        splits = np.where(splits)[0] + 1
        splits = np.r_[0, splits, len(ground_pcd)]

        # summation the point count in the same x, y voxel index
        # in the ground points
        ground_plane_count = utils_segmentation.segmented_sum(
            ground_pcd["count"],
            s_ind=splits[:-1],
            e_ind=splits[1:],
        )

        # only the voxels with more than 5 points are considered valid
        num_valid_ground_voxel = np.sum(ground_plane_count > 5)

        voxel_size = origin_scene.scene_details["voxel_size"]
        area_per_voxel = voxel_size * voxel_size

        total_area += num_valid_ground_voxel * area_per_voxel

        num_scene += 1

        prog.toc()

    print(f"Processed {num_scene} scenes")
    print(f"Total area: {total_area}")
    print(f"Total trajectory length: {total_traj}")


if __name__ == "__main__":
    main()
