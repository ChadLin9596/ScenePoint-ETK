import scene_point_etk.scene_db as scene_db


def main():

    for log_id in scene_db.list_scene_ids():
        versions = scene_db.list_versions_by_scene_id(log_id)

        for version in versions:

            origin_scene = scene_db.OriginalScene(log_id)
            edited_scene = scene_db.EditedScene(log_id, version)
            if len(edited_scene.scene_details["add"]) == 0:
                print(f"Skipping {log_id}/{version}")
                continue

            scene_details = edited_scene.scene_details
            merge_indices = scene_db.infer_merge_indices(
                origin_scene.scene_pcd,
                scene_details["add"],
            )
            scene_details["add"]["merge_indices"] = merge_indices
            edited_scene.scene_details = scene_details
            print(f"Fixed    {log_id}/{version}")


if __name__ == "__main__":
    main()
