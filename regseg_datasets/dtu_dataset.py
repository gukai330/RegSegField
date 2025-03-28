
from typing import Dict

import numpy as np

import torch
from torch.nn import functional as F

import pickle 

from pathlib import Path

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs, Semantics
from nerfstudio.data.datasets.base_dataset import InputDataset

def get_multi_mask_tensor_from_path(filepath: Path, scale_factor: float = 1.0) -> torch.Tensor:
    """
    Utility function to read a mask image from the given path and return a boolean tensor
    """
    with open(filepath, 'rb') as f:
        masks = pickle.load(f)
        temp = []
        mask0 = torch.from_numpy(masks[0])
        if mask0.shape[0] == 1:
            masks0 = mask0.squeeze(0)
        for m in masks:
            if m is not None:
                m = torch.from_numpy(m)
                if m.shape[0] == 1:
                    m = m.squeeze(0)
               
                temp.append(preprocess_mask(m,1))
            else:
                temp.append(torch.zeros_like(mask0).bool())
        # masks = [m[0] for m in masks]
        masks = temp
        masks = torch.stack(masks, dim=-1)
        # masks = masks.permute(1, 2, 0) # (H, W, M)
        # masks = torch.from_numpy(masks).bool()
    if scale_factor != 1.0:

        width, height = masks.size()[:2]
        newsize = (int(width * scale_factor), int(height * scale_factor), masks.size(2))
        masks = F.interpolate(masks.unsqueeze(0).float(), size=newsize, mode='nearest').squeeze(0).bool()
    
    # make the mask (M, H, W) -> (H, W, M)
    # masks = masks.permute(1, 2, 0)
    use_last = False
    if not use_last:
        masks = masks[..., :-1]

    return masks

def preprocess_mask(mask, size= 1):
    if size <=0:
        return mask.bool()
    # dilate the mask
    mask = mask.float()
    mask = F.max_pool2d(mask.unsqueeze(0).float(), kernel_size=size*2+1, stride=1, padding=size).squeeze(0).float()
    mask = mask.bool()
    return mask

    


class MultiMaskDataset(InputDataset):


    def get_data(self, image_idx: int) -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        image = self.get_image(image_idx)
        data = {"image_idx": image_idx, "image": image}
        if self._dataparser_outputs.mask_filenames is not None:
            # data["mask"] = self.get_mask(image_idx)
            data["masks"] = self.get_masks(image_idx)
            data["mask"] = torch.any(data["masks"], dim=-1).unsqueeze(-1)
            if hasattr(self._dataparser_outputs, "binary_masks"):
                data["mask"] = torch.from_numpy(
                    self._dataparser_outputs.binary_masks[image_idx]
                ).unsqueeze(-1)
            
            assert (
                data["masks"].shape[:2] == data["image"].shape[:2]
            ), f"Mask and image have different shapes. Got {data['masks'].shape[:2]} and {data['image'].shape[:2]}"
        metadata = self.get_metadata(data)
        data.update(metadata)
        return data


    def get_masks(self, image_idx: int):
        mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
        masks = get_multi_mask_tensor_from_path(filepath=mask_filepath, scale_factor=self.scale_factor)
        return masks
    
    # def get_mask(self, image_idx: int):
    #     mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
    #     masks = get_multi_mask_tensor_from_path(filepath=mask_filepath, scale_factor=self.scale_factor)
    #     return torch.any(masks, dim=-1).unsqueeze(-1)
