
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from nerfstudio.configs.base_config import InstantiateConfig

from dataclasses import dataclass, field

# import dataloader from torch

from torch.utils.data import DataLoader

from mask_matching.mask_matching import MaskPermutation, setup_downscale_factor, load_images, post_process_dict

from typing import List, Tuple, Type, Optional

import pickle

import cv2

from nerfstudio.cameras.cameras import Cameras, CameraType

from nerfstudio.cameras.camera_utils import fisheye624_project, fisheye624_unproject_helper

import tqdm

@dataclass
class MaskMatcherLoaderConfig(InstantiateConfig):

    _target: Type = field(default_factory=lambda: MaskMatcherLoader)

    data_dir: Path = Path("data")
    """Data directory"""

    ckpt_path: Path = Path("ckpt")
    """Checkpoint path"""

    # mask_filename: List[Path] = ()
    # """Mask filename"""

    downscale_factor: int = 8
    """Downscale factor"""

    colmap_path: Path = Path("sparse/0")
    """Colmap path"""



class MaskMatcherLoader(DataLoader):

    config: MaskMatcherLoaderConfig

    def __init__(self, config, mask_filenames: List[Path],  cameras:Optional[Cameras] = None):

        # load pkl mask
        self.config = config

        self.device = None
    
        self.mask_filenames = mask_filenames
        self.cameras = cameras

        self.load_multi_masks(mask_filenames)

        self.load_model(config.ckpt_path, self.raw_masks)

        self._get_descriptor()




    def load_model(self, path: Path,data_dict):
        
        model = MaskPermutation(data_dict=self.raw_masks)
        
        # load ckpt to model
        try:
            model.load_state_dict(torch.load(path))
        except:
            # different shape
            pass
        self.model_path = path
        self.model = model

        pass
    
    def load_multi_masks(self, paths: List[Path]):
        

        self.raw_masks = {}
        for i, mask_path in tqdm.tqdm(enumerate(paths)):


            # replace the extension with pkl
            
            mask_path = Path(mask_path)

            mask_path = mask_path.with_suffix('.pkl')
            with open(mask_path, "rb") as f:
                name = mask_path.stem
                # remove the surfix
                mask = pickle.load(f)

                mask.sort(key = lambda x: x['area'], reverse=True)

                image_mask = []
                raw_image_mask = []
                mask_score = []
                for single_mask in mask:
                    segmentation = single_mask['segmentation']
                    
                    # raw_image_mask.append(segmentation)
                    # do binary_erosion
                    # segmentation = skimage.morphology.binary_erosion(segmentation, skimage.morphology.disk(1))
                    image_mask.append(segmentation)
                    mask_score.append(single_mask['stability_score'])

                image_mask = torch.tensor(np.array(image_mask)).float()


                if self.cameras is not None:
                    camera = self.cameras[i]
                    distortion_params = camera.distortion_params.numpy()
                    K = camera.get_intrinsics_matrices().numpy()
                    K, undistort_image_mask = _undistort_masks(camera, distortion_params, image_mask.permute(1, 2, 0), K)

                    undistort_image_mask = torch.tensor(undistort_image_mask).bool().float().permute(2, 0, 1)

                else:
                    undistort_image_mask = torch.tensor(image_mask)

                self.raw_masks[name]={
                    "image_masks": undistort_image_mask,
                    "mask_scores": mask_score,
                    "max_overlap": 1,
                }


        self.raw_masks = post_process_dict(self.raw_masks, False)
        self.nested_raw_masks = self.cache_nested_raw_masks()
        self.cache_unique_masks()


    def cache_nested_raw_masks(self):


        masks = []
        for name, data in self.raw_masks.items():

            masks.append(data["image_masks"].permute(1, 2, 0))
        
        return torch.nested.nested_tensor(masks).to_padded_tensor(0)

    
    @torch.no_grad()
    def _get_descriptor(self):
        
        all_descriptors = []

        for name, data in self.raw_masks.items():
            
            masks = data["image_masks"].permute(1, 2, 0)
            if self.device is not None:
                masks = masks.to(self.device)
            scores = data["mask_scores"]

            # get the descriptor
            
            multi_levels,_ =self.model.get_features(name, masks)

            all_descriptors.append(multi_levels)
        
        self.all_descriptors = all_descriptors
    
    def __getitem__(self, index):
        
        return self.all_descriptors[index]
    
    def get_model(self):
        return self.model
    
    def set_device(self, device):
        self.device = device
        self.model.to(device)
    
    def encode_mask(self, index):
        # get the descriptor and preserve the gradient
        name = self.mask_filenames[index].stem

        masks = self.raw_masks[name]["image_masks"].permute(1, 2, 0)
        if self.device is not None:
            masks = masks.to(self.device)
        multi_levels, _ = self.model.get_features(name, masks)
        # update the cached masks
        # self.all_descriptors[index] = multi_levels.detach()
        return multi_levels

    def update_all_descriptors(self):

        before_update = self.all_descriptors

        self._get_descriptor()

        print("checking the descriptors")

    def cache_unique_masks(self):
        unique_mask_vals = []
        for name, data in self.raw_masks.items():
            masks = data["image_masks"].permute(1, 2, 0)
            flatten_masks = torch.flatten(masks, start_dim=0, end_dim=1)
            res = torch.unique(dim=0, return_inverse=True, return_counts=True, input=flatten_masks)

            unique_mask_vals.append(res)

        self.unique_mask_vals = unique_mask_vals


    def get_raw_mask(self, index):
        return self.raw_masks[self.mask_filenames[index].stem]["image_masks"].permute(1, 2, 0)

    def get_unique_mask(self, index):
        return self.unique_mask_vals[index]
    
    def sample_masks(self, ray_indices):
        num_masks = len(self.all_descriptors)
        feature_dims = self.all_descriptors[0].shape[2:]
        # create features
        sampled_masks = torch.zeros(*ray_indices.shape[:-1], *feature_dims, device=ray_indices.device)

        for i in range(num_masks):
            select_idx = ray_indices[...,0] == i

            y = ray_indices[select_idx, 1]
            x = ray_indices[select_idx, 2]

            descriptor = self.all_descriptors[i].to(ray_indices.device)

            sampled_masks[select_idx] = descriptor[y,x]

        return sampled_masks
    
    def sample_masks_with_grad(self, ray_indices):
        sampled_raw_masks, binary_indices = self.sample_raw_masks(ray_indices)

        sampled_masks_with_grad = torch.zeros(*ray_indices.shape[:-1], *self.all_descriptors[0].shape[2:],device=self.device)
        for i in range(len(sampled_raw_masks)):
            name = self.mask_filenames[i].stem
            multi_levels, _ = self.model.get_features(name, sampled_raw_masks[i].to(self.device))
            sampled_masks_with_grad[binary_indices[i]] = multi_levels

        return sampled_masks_with_grad


    def sample_raw_masks(self, ray_indices):
        num_masks = len(self.raw_masks)

        sampled_raw_masks = []

        binary_indices=  []
        for i in range(num_masks):
            select_idx = ray_indices[...,0] == i

            y = ray_indices[select_idx, 1]
            x = ray_indices[select_idx, 2]

            mask_name = self.mask_filenames[i].stem


            sampled_raw_masks.append(
                self.get_raw_mask(i)[y,x]
                # self.raw_masks[mask][y,x]
            )
            binary_indices.append(select_idx)

        return sampled_raw_masks, binary_indices
    

    def sample_nested_masks(self, ray_indices):
        
        sampled_nested = self.nested_raw_masks[ray_indices[...,0],ray_indices[...,1],ray_indices[...,2]]

        return sampled_nested

def _undistort_masks(
    camera: Cameras, distortion_params: np.ndarray, masks: torch.Tensor, K: np.ndarray
) -> Tuple[np.ndarray, Optional[torch.Tensor]]:
    masks = masks.numpy()
    masks = masks.astype(np.uint8) * 255
    if camera.camera_type.item() == CameraType.PERSPECTIVE.value:
        distortion_params = np.array(
            [
                distortion_params[0],
                distortion_params[1],
                distortion_params[4],
                distortion_params[5],
                distortion_params[2],
                distortion_params[3],
                0,
                0,
            ]
        )
        if np.any(distortion_params):
            newK, roi = cv2.getOptimalNewCameraMatrix(K, distortion_params, (masks.shape[1], masks.shape[0]), 0)
            masks = cv2.undistort(masks, K, distortion_params, None, newK)  # type: ignore
        else:
            newK = K
            roi = 0, 0, masks.shape[1], masks.shape[0]
        # crop the image and update the intrinsics accordingly
        x,y,w,h = roi
        masks = masks[y : y + h, x : x + w]
        K = newK

    elif camera.camera_type.item() == CameraType.FISHEYE.value:
        distortion_params = np.array(
            [distortion_params[0], distortion_params[1], distortion_params[2], distortion_params[3]]
        )
        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, distortion_params, (masks.shape[1], masks.shape[0]), np.eye(3), balance=0
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, distortion_params, np.eye(3), newK, (masks.shape[1], masks.shape[0]), cv2.CV_32FC1
        )
        # and then remap:
        masks = cv2.remap(masks, map1, map2, interpolation=cv2.INTER_LINEAR)

        K = newK
    elif camera.camera_type.item() == CameraType.FISHEYE624.value:
        fisheye624_params = torch.cat(
            [camera.fx, camera.fy, camera.cx, camera.cy, torch.from_numpy(distortion_params)], dim=0
        )
        assert fisheye624_params.shape == (16,)
        assert (
            camera.metadata is not None
            and "fisheye_crop_radius" in camera.metadata
            and isinstance(camera.metadata["fisheye_crop_radius"], float)
        )
        fisheye_crop_radius = camera.metadata["fisheye_crop_radius"]

        # Approximate the FOV of the unmasked region of the camera.
        upper, lower, left, right = fisheye624_unproject_helper(
            torch.tensor(
                [
                    [camera.cx, camera.cy - fisheye_crop_radius],
                    [camera.cx, camera.cy + fisheye_crop_radius],
                    [camera.cx - fisheye_crop_radius, camera.cy],
                    [camera.cx + fisheye_crop_radius, camera.cy],
                ],
                dtype=torch.float32,
            )[None],
            params=fisheye624_params[None],
        ).squeeze(dim=0)
        fov_radians = torch.max(
            torch.acos(torch.sum(upper * lower / torch.linalg.norm(upper) / torch.linalg.norm(lower))),
            torch.acos(torch.sum(left * right / torch.linalg.norm(left) / torch.linalg.norm(right))),
        )

        # Heuristics to determine parameters of an undistorted image.
        undist_h = int(fisheye_crop_radius * 2)
        undist_w = int(fisheye_crop_radius * 2)
        undistort_focal = undist_h / (2 * torch.tan(fov_radians / 2.0))
        undist_K = torch.eye(3)
        undist_K[0, 0] = undistort_focal  # fx
        undist_K[1, 1] = undistort_focal  # fy
        undist_K[0, 2] = (undist_w - 1) / 2.0  # cx; for a 1x1 image, center should be at (0, 0).
        undist_K[1, 2] = (undist_h - 1) / 2.0  # cy

        # Undistorted 2D coordinates -> rays -> reproject to distorted UV coordinates.
        undist_uv_homog = torch.stack(
            [
                *torch.meshgrid(
                    torch.arange(undist_w, dtype=torch.float32),
                    torch.arange(undist_h, dtype=torch.float32),
                ),
                torch.ones((undist_w, undist_h), dtype=torch.float32),
            ],
            dim=-1,
        )
        assert undist_uv_homog.shape == (undist_w, undist_h, 3)
        dist_uv = (
            fisheye624_project(
                xyz=(
                    torch.einsum(
                        "ij,bj->bi",
                        torch.linalg.inv(undist_K),
                        undist_uv_homog.reshape((undist_w * undist_h, 3)),
                    )[None]
                ),
                params=fisheye624_params[None, :],
            )
            .reshape((undist_w, undist_h, 2))
            .numpy()
        )
        map1 = dist_uv[..., 1]
        map2 = dist_uv[..., 0]

        # Use correspondence to undistort image.
        masks = cv2.remap(masks, map1, map2, interpolation=cv2.INTER_LINEAR)

        K = undist_K.numpy()
    else:
        raise NotImplementedError("Only perspective and fisheye cameras are supported")

    return K, masks


if __name__ == "__main__":

    def load_cameras(data,recon_dir, images_path, masks_path,
                      depths_path=None, downscale_factor=8, llff_hold = 8, input_views = 24,):

        meta = load_images(data, recon_dir, images_path, masks_path, depths_path)
        
        frames = meta["frames"]


        image_files = []
        mask_files = []
        maskimage_files = []
        depth_files = []

        for frame in frames:
            image_files.append(frame["file_path"])
            mask_files.append(frame["mask_path"])
            maskimage_files.append(frame["mask_path"].replace(".pkl", ".png"))
            if depths_path is not None:
                depth_files.append(frame["depth_path"])
        
        setup_downscale_factor(data,image_files, mask_files,maskimage_files, depth_files, downscale_factor=downscale_factor,images_path=images_path, masks_path=masks_path)

        def get_fname(filepath: Path) -> Path:
            """Returns transformed file name when downscale factor is applied"""
            parts = list(filepath.parts)
            parts[-2] += f"_{downscale_factor}"
            filepath = Path(*parts)
            return data / filepath





        data_dicts = {}

        all_indices = np.arange(len(frames))

        indices = all_indices[all_indices % llff_hold == 0]

        if input_views < len(indices):
            sub_ind = np.linspace(0, len(indices) - 1, input_views)
            sub_ind = [round(i) for i in sub_ind]
            indices = [indices[i] for i in sub_ind]


        # masks_filename = []

        maskimage_files = []

        for i in indices:
            frame = frames[i]
            maskimage_files.append(get_fname(Path(frame["mask_path"])))

        return maskimage_files
    
    scene = "counter"
    data = Path(f"/mnt/e/data/360_v2/{scene}")
    recon_dir = data / "sparse" / "0"
    llff_hold = 8
    input_views = 24
    downscales_factor = 8


    mask_files = load_cameras(data, recon_dir, "images", "masks_allprompts",
                               downscale_factor=downscales_factor, llff_hold=llff_hold, input_views=input_views)

    print(len(mask_files))

    

    res = MaskMatcherLoaderConfig(
        data_dir=data,
        ckpt_path=Path("counter.pth"),
        mask_filename=mask_files,
        downscale_factor=8,
        colmap_path=Path("sparse/0"),
    )

    loader = res.setup()

    print(type(loader))
    