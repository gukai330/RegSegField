import os
import shutil
import argparse
import subprocess
import numpy as np


def is_image_file(filename):
    return any(filename.endswith(ext) for ext in [".png", ".PNG", ".jpg", ".JPG"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Mip NeRF 360 scenes and run COLMAP for 2D matches")
    parser.add_argument("--datadir", type=str, required=True, help="Path to the dataset root (LLFF-style)")
    parser.add_argument("--scenes", type=str, nargs="+", required=True, help="List of scene names")
    parser.add_argument("--input_views", type=int, default=12, help="Number of training views to sample")
    parser.add_argument("--llff_hold", type=int, default=8, help="LLFF holdout interval for test views")
    args = parser.parse_args()

    for scene in args.scenes:
        full_image_dir = os.path.join(args.datadir, scene, "images")
        subset_image_dir = os.path.join(args.datadir, scene, f"images_{args.input_views}views")
        os.makedirs(subset_image_dir, exist_ok=True)

        image_files = sorted([
            f for f in os.listdir(full_image_dir)
            if is_image_file(f)
        ])

        all_indices = np.arange(len(image_files))
        test_indices = all_indices[all_indices % args.llff_hold == 0]
        train_indices = all_indices[all_indices % args.llff_hold != 0]

        if args.input_views < len(train_indices):
            sub_ind = np.linspace(0, len(train_indices) - 1, args.input_views)
            sub_ind = [round(i) for i in sub_ind]
            train_indices = [train_indices[i] for i in sub_ind]

        for i in train_indices:
            shutil.copy(
                os.path.join(full_image_dir, image_files[i]),
                os.path.join(subset_image_dir, image_files[i])
            )

        # Step 1: extract 2D matches on subset (sparse input)
        db_subset = os.path.join(args.datadir, scene, f"database_{args.input_views}.db")
        if os.path.exists(db_subset):
            os.remove(db_subset)

        subprocess.run([
            "colmap", "feature_extractor",
            "--database_path", db_subset,
            "--image_path", subset_image_dir,
            "--ImageReader.camera_model", "OPENCV",
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.use_gpu", "1",
            "--SiftExtraction.max_num_features", "16384"
        ])

        subprocess.run([
            "colmap", "exhaustive_matcher",
            "--database_path", db_subset,
            "--SiftMatching.use_gpu", "1",
            "--SiftMatching.guided_matching", "1",
            "--SiftMatching.max_num_matches", "32768"
        ])

        shutil.rmtree(subset_image_dir)

        # Step 2: full COLMAP reconstruction on all images
        db_all = os.path.join(args.datadir, scene, "database.db")
        sparse_out = os.path.join(args.datadir, scene, "sparse")

        if os.path.exists(db_all) or os.path.exists(sparse_out):
            continue  # skip if already exists

        subprocess.run([
            "colmap", "feature_extractor",
            "--database_path", db_all,
            "--image_path", full_image_dir,
            "--ImageReader.camera_model", "OPENCV",
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.use_gpu", "1"
        ])

        subprocess.run([
            "colmap", "exhaustive_matcher",
            "--database_path", db_all,
            "--SiftMatching.use_gpu", "1"
        ])

        os.makedirs(sparse_out, exist_ok=True)
        subprocess.run([
            "colmap", "mapper",
            "--database_path", db_all,
            "--image_path", full_image_dir,
            "--output_path", sparse_out
        ])
