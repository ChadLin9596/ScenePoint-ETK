import copy
import numpy as np
import scene_point_etk.scene_db as scene_db
import scene_point_etk.utils as scene_utils


def fix_a_scene_pair(scene_id, version):

    origin_scene = scene_db.OriginalScene(scene_id)
    edited_scene = scene_db.EditedScene(scene_id, version)

    if "merge_indices" not in edited_scene.scene_details["add"]:
        short_id = scene_id[:8]
        msg = f"{short_id} {version} does not have merge_indices, skipping"
        print(msg)
        return

    details_0 = copy.deepcopy(edited_scene.scene_details)
    merge_indices_0 = copy.deepcopy(details_0["add"]["merge_indices"])

    details_1 = copy.deepcopy(edited_scene.scene_details)
    details_1["add"].pop("merge_indices")

    merge_indices_1 = scene_utils.infer_merge_indices(
        origin_scene.scene_pcd,
        details_1["add"],
    )
    details_1["add"]["merge_indices"] = merge_indices_1

    o_scene_pcd = origin_scene.scene_pcd
    func = scene_db.apply_change_info_to_target_pcd
    scene_pcd_0 = func(o_scene_pcd, details_0)
    scene_pcd_1 = func(o_scene_pcd, details_1)

    msg = f"{scene_id[:8]} {version}: \n"
    msg += f"\torigin merge_indices: {len(np.concatenate(merge_indices_0))}\n"
    msg += f"\tupdate merge_indices: {len(np.concatenate(merge_indices_1))}\n"
    print(msg)

    msg = f"{scene_id[:8]} {version}: "
    msg += f"\torigin scene pcd    : {len(scene_pcd_0)}\n"
    msg += f"\tupdate scene pcd    : {len(scene_pcd_1)}\n"
    print(msg)

    edited_scene.scene_details = details_1
    edited_scene.scene_pcd = scene_pcd_1


def main():

    pairs = scene_db.list_scene_version_pairs()
    for scene_id, version in pairs:
        fix_a_scene_pair(scene_id, version)


if __name__ == "__main__":
    main()
