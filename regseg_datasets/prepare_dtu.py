import os
import shutil
import argparse
import subprocess
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare DTU data and extract COLMAP features")
    parser.add_argument("--raw_datadir", type=str, required=True, help="Path to the original DTU dataset (Rectified)")
    parser.add_argument("--mask_dir", type=str, required=True, help="Path to IDR-style mask folders")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save converted data")
    parser.add_argument("--light_condition", type=int, default=3, help="Lighting condition index (default: 3)")
    args = parser.parse_args()

    train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13]
    exclude_idx = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
    test_idx = [i for i in np.arange(49) if i not in train_idx + exclude_idx]

    scenes = [s for s in os.listdir(args.mask_dir) if os.path.isdir(os.path.join(args.mask_dir, s))]

    for scene in scenes:
        scene_path = os.path.join(args.raw_datadir, "Rectified", scene)
        mask_path = os.path.join(args.mask_dir, scene, "mask")

        image_output = os.path.join(args.output_dir, scene, "images")
        mask_output = os.path.join(args.output_dir, scene, "idr_masks")
        os.makedirs(image_output, exist_ok=True)
        os.makedirs(mask_output, exist_ok=True)

        for split, indices in zip(["train", "test"], [train_idx, test_idx]):
            with open(os.path.join(args.output_dir, scene, f"{split}.txt"), "w") as f:
                for ind in indices:
                    image_name = f"rect_{ind+1:03d}_{args.light_condition}_r5000.png"
                    src_image = os.path.join(scene_path, image_name)
                    src_mask = os.path.join(mask_path, f"{ind:03d}.png")
                    dst_image = os.path.join(image_output, image_name)
                    dst_mask = os.path.join(mask_output, image_name)

                    shutil.copy(src_image, dst_image)
                    shutil.copy(src_mask, dst_mask)
                    f.write(f"{image_name}\n")

        # Run COLMAP
        image_dir = image_output
        database_path = os.path.join(args.output_dir, scene, "database.db")
        sparse_output = os.path.join(args.output_dir, scene, "sparse")

        if os.path.exists(database_path):
            os.remove(database_path)
        if os.path.exists(sparse_output):
            shutil.rmtree(sparse_output)
        os.makedirs(sparse_output, exist_ok=True)

        subprocess.run(["colmap", "feature_extractor",
                        "--database_path", database_path,
                        "--image_path", image_dir,
                        "--ImageReader.camera_model", "PINHOLE",
                        "--ImageReader.single_camera", "1"])

        subprocess.run(["colmap", "exhaustive_matcher",
                        "--database_path", database_path])

        subprocess.run(["colmap", "mapper", 
                        "--database_path", database_path,
                        "--image_path", image_dir,
                        "--output_path", sparse_output])
