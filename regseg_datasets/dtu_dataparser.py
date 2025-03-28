# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data parser for blender dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, Tuple, List
from typing_extensions import Literal

import imageio
import pickle
import numpy as np
import torch
import os

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json

from regseg_nerfacto.mask_regularizer import InMaskRegularizer
from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator

@dataclass
class DTUDataParserConfig(DataParserConfig):
    """Blender dataset parser config"""

    _target: Type = field(default_factory=lambda: DTU)
    """target class to instantiate"""
    data: Path = Path("data/blender/lego")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    alpha_color: str = "black"
    """alpha color of background"""
    scene_rotation_x: float = 0.0
    """rotation apply to scene along the x axis"""
    orientation_method: Literal["pca", "up", "none", "vertical"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "focus"
    """Whether to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""

    # dtu specific
    use_masks: bool = True
    """whether to use masks"""
    scan: int = 63
    """scan number"""
    input_views: int = 3
    """number of input views"""

@dataclass
class DTU(DataParser):
    """Blender Dataset
    Some of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37.
    """

    config: DTUDataParserConfig

    def __init__(self, config: DTUDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color
        self.scene_rotation_x: float = config.scene_rotation_x
        self.downsampling_factor = 1
        self.all_poses: np.ndarray = None
        self.bounds: np.ndarray = None

    def _generate_dataparser_outputs(self, split="train"):
        if self.alpha_color is not None:
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None

        n_images = 49 # 49 or 64 images
        light_condition = 3

        image_filenames = []
        mask_filenames = None
        all_poses = []
        fx = []
        fy = []
        cx = []
        cy = []

        for i in range(1, n_images+1):
            file_name = f"rect_{i:03d}_{light_condition}_r" \
                + ('5000' if i < 50 else'7000') + ".png"
            image_path = self.data / "Rectified"/ f"scan{self.config.scan}_train" / file_name

            camera_txt = self.data / "Cameras" / "train" / f"{i-1:08d}_cam.txt"
            intrinsics, extrinsics,near_far = read_cam_file(camera_txt) 
        
            intrinsics[:2] *= 1 / self.downsampling_factor
          
            c2w = np.linalg.inv(extrinsics) 
            c2w[0:3, 1:3] *= -1
            # c2w = c2w[np.array([1, 0, 2, 3]), :]
            # c2w[2, :] *= -1

            
            fx.append(intrinsics[0,0])
            fy.append(intrinsics[1,1])
            cx.append(intrinsics[0,2])
            cy.append(intrinsics[1,2])

            
            all_poses.append(c2w)
            image_filenames.append(image_path)

        all_poses = np.array(all_poses).astype(np.float32)

        # all_indices = np.arange(poses.shape[0])

        # use pixelnerf fashion split

        train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13]
        exclude_idx = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
        test_idx = [i for i in np.arange(49) if i not in train_idx + exclude_idx]

        # train_idx = [i for i in np.arange(49) if i not in exclude_idx]
  
        if split == "train":
            indices = train_idx
            if self.config.input_views < len(indices):
                indices = indices[:self.config.input_views]
        else:
            indices = test_idx
        
        # print the split and indices
        print(f"split: {split}, used indices: {indices}")

        # select image files 
        image_filenames = [image_filenames[i] for i in indices]
        
        # load poses
        if self.config.use_masks and split == "train":
            mask_filenames = []
            for image_filename in image_filenames:
                # load the mask of the same name
                mask_file = Path(image_filename).stem + ".pkl"
                # mask_folder = "Amodals"
                mask_folder = "Masks"
                mask_file = self.data / mask_folder / f"scan{self.config.scan}_train" / mask_file
                # check if file exists
                assert os.path.exists(mask_file), f"mask file {mask_file} does not exist"
                
                mask_filenames.append(mask_file)


        # meta = load_from_json(self.data / f"transforms_{split}.json")
        # image_filenames = []
        # poses = [0.

        all_poses = np.array(all_poses).astype(np.float32)
        img_0 = imageio.imread(image_filenames[0])
        image_height, image_width = img_0.shape[:2]

        print("mean pos before centering", np.mean(all_poses[:, :3, 3], axis=0))

        all_poses,transform_matrix = camera_utils.auto_orient_and_center_poses(
            torch.from_numpy(all_poses),
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )

        # auto rescale
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(all_poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor
        # self.config.scale_factor = scale_factor

        print("mean pos after centering", torch.mean(all_poses[:, :3, 3], dim=0), scale_factor)
        # apply scale
        all_poses[:, :3, 3] *= scale_factor

        # apply scales to the poses and bounds
        self.all_poses = all_poses
        # self.bounds = bounds * scale_factor

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = all_poses[idx_tensor]
        camera_to_world = poses[:, :3,:4]  # camera to world transform
        
        scene_box = SceneBox(aabb=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32))

        camera_type = CameraType.PERSPECTIVE

        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=torch.tensor(fx, dtype=torch.float32)[idx_tensor],
            fy=torch.tensor(fy, dtype=torch.float32)[idx_tensor],
            cx=torch.tensor(cx, dtype=torch.float32)[idx_tensor],
            cy=torch.tensor(cy, dtype=torch.float32)[idx_tensor],
            height=image_height,
            width=image_width,
            camera_type=camera_type,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            mask_filenames=mask_filenames,
            cameras=cameras,
            alpha_color=alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
            dataparser_transform=transform_matrix
        )

        return dataparser_outputs


def read_cam_file(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
    extrinsics = extrinsics.reshape((4, 4))


    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
    intrinsics = intrinsics.reshape((3, 3))
    intrinsics[:2] *= 4
    # extrinsics[:3, 3] *= 1/200.0
    # depth_min & depth_interval: line 11
    depth_min = float(lines[11].split()[0])
    depth_max = depth_min + float(lines[11].split()[1]) * 192
    return intrinsics, extrinsics,  [depth_min, depth_max]
