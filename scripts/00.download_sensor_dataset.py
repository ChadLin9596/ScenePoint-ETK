import argparse
import os

import scene_point_etk.utils.s5_cmd as s5_cmd


def main(root, output, dry_run=False):

    os.makedirs(output, exist_ok=True)

    dirs = s5_cmd.s5_ls(root)
    dirs = [i.split()[-1].replace("/", "") for i in dirs]

    for dir in dirs:
        print("downloading", dir)

        src = f"{root}{dir}/*"
        dst = os.path.join(output, dir)

        os.mkdir(dst)

        if dry_run:
            print(f"copy from {src} to {dst}")
            continue

        s5_cmd.s5_cp(src, dst)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str)
    parser.add_argument(
        "--dry",
        action="store_true",
        help="If set, will not download files, just print the commands.",
    )
    args = parser.parse_args()

    main(
        "s3://argoverse/datasets/av2/sensor/train/",
        args.output,
        dry_run=args.dry,
    )
    main(
        "s3://argoverse/datasets/av2/sensor/val/",
        args.output,
        dry_run=args.dry,
    )
