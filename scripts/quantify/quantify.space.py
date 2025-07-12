import glob
import os

import scene_point_etk.scene_db as scene_db

import py_utils.utils as utils

getsize = os.path.getsize


def main():

    num_origin_scene = 0
    num_edited_scene = 0

    size_origin_details = 0
    size_origin_pcd = 0

    size_edited_details = 0
    size_edited_pcd = 0

    size_camera_details = 0
    size_camera_map = 0

    prog = utils.ProgressTimer()
    prog.tic(len(scene_db.list_scene_ids()))

    for log_id in scene_db.list_scene_ids():

        origin_scene = scene_db.OriginalScene(log_id)
        num_origin_scene += 1
        size_origin_details += getsize(origin_scene.details_filepath)
        size_origin_pcd += getsize(origin_scene.scene_filepath)
        size_camera_details += getsize(origin_scene.camera_seq_filepath)

        for camera in origin_scene.cameras:
            path = os.path.join(
                origin_scene.cameras_root,
                camera,
                "sparse_point_indices",
                "*.npy",
            )
            for f in glob.glob(path):
                size_camera_map += getsize(f)

        versions = scene_db.list_versions_by_scene_id(log_id)
        for version in versions:
            num_edited_scene += 1
            edited_scene = scene_db.EditedScene(log_id, version)
            size_edited_details += getsize(edited_scene.details_filepath)
            size_edited_pcd += getsize(edited_scene.scene_filepath)

            for camera in edited_scene.cameras:
                path = os.path.join(
                    edited_scene.cameras_root,
                    camera,
                    "sparse_point_indices",
                    "*.npy",
                )
                for f in glob.glob(path):
                    size_camera_map += getsize(f)

        prog.toc()

    print("num_origin_scene\t", num_origin_scene)
    print("num_edited_scene\t", num_edited_scene)
    print("size_origin_details\t", round(size_origin_details / 2**30, 2), "GB")
    print("size_origin_pcd    \t", round(size_origin_pcd / 2**30, 2), "GB")
    print("size_edited_details\t", round(size_edited_details / 2**30, 2), "GB")
    print("size_edited_pcd    \t", round(size_edited_pcd / 2**30, 2), "GB")
    print("size_camera_details\t", round(size_camera_details / 2**30, 2), "GB")
    print("size_camera_map    \t", round(size_camera_map / 2**30, 2), "GB")


if __name__ == "__main__":
    main()
