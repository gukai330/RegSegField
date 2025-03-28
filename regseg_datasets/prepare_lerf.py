import os
import shutil
import argparse
import subprocess
import numpy as np


def is_image_file(filename):
    return any(filename.endswith(ext) for ext in [".png", ".PNG", ".jpg", ".JPG"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare LERF images and run COLMAP for 2D matching")
    parser.add_argument("--datadir", type=str, required=True, help="Path to the dataset root")
    parser.add_argument("--scenes", type=str, nargs="+", required=True, help="List of scene names to process")
    parser.add_argument("--input_views", type=int, default=8, help="Number of training views to sample")
    parser.add_argument("--llff_hold", type=int, default=8, help="Hold-out interval for test views")
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

        # Step 1: extract keypoints from subset (sparse input)
        database_subset = os.path.join(args.datadir, scene, f"database_{args.input_views}.db")
        if os.path.exists(database_subset):
            os.remove(database_subset)

        subprocess.run([
            "colmap", "feature_extractor",
            "--database_path", database_subset,
            "--image_path", subset_image_dir,
            "--ImageReader.camera_model", "OPENCV",
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.use_gpu", "1",
            "--SiftExtraction.max_num_features", "16384"
        ])

        subprocess.run([
            "colmap", "exhaustive_matcher",
            "--database_path", database_subset,
            "--SiftMatching.use_gpu", "1",
            "--SiftMatching.guided_matching", "1",
            "--SiftMatching.max_num_matches", "32768"
        ])

        shutil.rmtree(subset_image_dir)

        # Step 2: run full COLMAP pipeline on all images (to get poses & 3D points)
        database_full = os.path.join(args.datadir, scene, "database_all.db")
        output_path = os.path.join(args.datadir, scene, "sparse")

        if os.path.exists(database_full) or os.path.exists(output_path):
            continue  # Skip if already processed

        subprocess.run([
            "colmap", "feature_extractor",
            "--database_path", database_full,
            "--image_path", full_image_dir,
            "--ImageReader.camera_model", "OPENCV",
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.use_gpu", "1"
        ])

        subprocess.run([
            "colmap", "exhaustive_matcher",
            "--database_path", database_full,
            "--SiftMatching.use_gpu", "1"
        ])

        os.makedirs(output_path, exist_ok=True)

        subprocess.run([
            "colmap", "mapper",
            "--database_path", database_full,
            "--image_path", full_image_dir,
            "--output_path", output_path
        ])
