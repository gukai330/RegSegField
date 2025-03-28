#!/usr/bin/env python
"""
Script to extract segmentation masks for all images in a dataset using Semantic-SAM.

Please run this script from the root directory of the Semantic-SAM repository,
as it depends on local imports like `semantic_sam`.

Example:
    cd /path/to/Semantic-SAM
    python extract_mask.py \
        --dataset_root /path/to/dataset_root \
        --sam_ckpt ./pretrained/swinl_only_sam_many2many.pth
"""

import os
import imageio
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse

from semantic_sam import prepare_image, build_semantic_sam, SemanticSamAutomaticMaskGenerator


def downscale_mask(mask_dict, target_size):
    original_mask = mask_dict['segmentation']
    original_size = original_mask.shape[::-1]
    scale_x, scale_y = target_size[0] / original_size[0], target_size[1] / original_size[1]

    downscaled_mask = F.interpolate(
        torch.from_numpy(original_mask).unsqueeze(0).unsqueeze(0).float(),
        size=target_size, mode='nearest'
    ).squeeze().numpy()

    mask_dict.update({
        'segmentation': downscaled_mask.astype(bool),
        'area': np.sum(downscaled_mask > 0),
        'bbox': tuple(x * scale for x, scale in zip(mask_dict['bbox'], (scale_x, scale_y, scale_x, scale_y))),
        'crop_box': tuple(x * scale for x, scale in zip(mask_dict['crop_box'], (scale_x, scale_y, scale_x, scale_y))),
        'point_coords': (np.array(mask_dict['point_coords']) / ((scale_x + scale_y) * 0.5)).tolist()
    })

    return mask_dict


def post_process(anns):
    if len(anns) == 0:
        return None
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    H, W = sorted_anns[0]['segmentation'].shape
    result = np.zeros((H, W), dtype=np.uint8)
    for i, ann in enumerate(sorted_anns):
        result[ann['segmentation']] = i + 1
    if (result > 0).any():
        result[result > 0] = np.unique(result[result > 0], return_inverse=True)[1] + 1
    return result


def main():
    parser = argparse.ArgumentParser(description="Extract semantic masks using Semantic-SAM")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root directory containing scene folders")
    parser.add_argument("--scenes", nargs="*", help="List of scene names (default: all folders with 'images')")
    parser.add_argument("--output_mask_dir", type=str, default="masks_allprompts", help="Subfolder to save masks")
    parser.add_argument("--sam_ckpt", type=str, required=True, help="Path to Semantic-SAM checkpoint")

    args = parser.parse_args()

    if args.scenes:
        scenes = args.scenes
    else:
        scenes = [d for d in os.listdir(args.dataset_root)
                  if os.path.isdir(os.path.join(args.dataset_root, d)) and
                  os.path.exists(os.path.join(args.dataset_root, d, "images"))]

    mask_generator = SemanticSamAutomaticMaskGenerator(
        build_semantic_sam(model_type='L', ckpt=args.sam_ckpt).to("cuda"),
        level=[1, 2, 3, 4, 5, 6],
        min_mask_region_area=500,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.9
    )

    for scene in tqdm(scenes, desc="Processing scenes"):
        image_folder = os.path.join(args.dataset_root, scene, "images")
        output_folder = os.path.join(args.dataset_root, scene, args.output_mask_dir)
        os.makedirs(output_folder, exist_ok=True)

        for image_file in os.listdir(image_folder):
            if not image_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            image_path = os.path.join(image_folder, image_file)
            original_image, input_image = prepare_image(image_pth=image_path)
            masks = mask_generator.generate(input_image)

            raw_image = imageio.imread(image_path)
            if masks and (masks[0]['segmentation'].shape != raw_image.shape[:2]):
                for i in range(len(masks)):
                    masks[i] = downscale_mask(masks[i], target_size=raw_image.shape[:2])

            raw_mask_path = os.path.join(output_folder, image_file.rsplit(".", 1)[0] + ".pkl")
            with open(raw_mask_path, "wb") as f:
                pickle.dump(masks, f)

            merged_mask = post_process(masks)
            if merged_mask is None:
                continue

            merged_mask = merged_mask.astype(np.uint8)
            out_file = os.path.join(output_folder, image_file.rsplit(".", 1)[0] + ".png")
            imageio.imwrite(out_file, merged_mask)


if __name__ == "__main__":
    main()
