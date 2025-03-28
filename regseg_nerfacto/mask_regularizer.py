from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
from dataclasses import dataclass, field
import cv2
import torch
from torch.nn.functional import grid_sample

from nerfstudio.utils.poses import *
from nerfstudio.configs import base_config as cfg

# 1. get a point 
# 2. get the mask value of the point 
# reproject the point to all the other train views 
# find if the point is in the mask(of same id) of the other views
# 
def unpack_4x4_transform(transform):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return rotation, translation

DEBUG = False

def debugprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)
    pass
    # print(*args, **kwargs)

@dataclass
class AutoMatchRegularizerConfig(cfg.InstantiateConfig):

    _target: Type = field(default_factory=lambda: AutoMatchRegularizer)
    """target class to instantiate"""
    reg_strength: float = 1
    """regularization strength"""
    mask_count_thresh: float = 2
    """mask count threshold"""
    regularize_mode: Literal["frustum", "single_mask", "multi_mask"] = "multi_mask"


class AutoMatchRegularizer:

    def __init__(
            self, 
            config: AutoMatchRegularizerConfig,
            cameras,
            mask_loader,
            ):
        self.config = config
        self.regularize_mode = self.config.regularize_mode

        # check if poses are [N, 4, 4]        
        self.cameras = cameras
        # inverser all the poses 
        self.transforms_w2c = [inverse(c2w) for c2w in cameras.camera_to_worlds]
        self.transforms_w2c = torch.stack(self.transforms_w2c)
        self.fx = cameras.fx.detach().squeeze().cpu().numpy()
        self.fy = cameras.fy.detach().squeeze().cpu().numpy()
        self.cx = cameras.cx.detach().squeeze().cpu().numpy()
        self.cy = cameras.cy.detach().squeeze().cpu().numpy()

        # print the shape of fx fy cx cy      
        self.mask_loader = mask_loader
        # self.correlate_matrix = correlate_matrix
        # self.preprocess_correlate_matrix()

    def preprocess_correlate_matrix(self):
        # get_maximum size
        max_size = 0
        for mats in self.correlate_matrix:
            for mat in mats:
                if mat is None:
                    continue
                max_size = max(max_size, mat.shape[0])
                max_size = max(max_size, mat.shape[1])
        
        # pad the matrix
        correlate_matrix_list = []
        for mats in self.correlate_matrix:
            correlate_matrices = []
            for mat in mats:
                if mat is None:
                    # make diagno
                    mat = torch.eye(max_size, max_size)
                else:
                    mat = torch.from_numpy(mat)
                    print("mat bfore pad", mat.shape)
                    mat = torch.nn.functional.pad(mat, (0, max_size - mat.shape[-1], 0, max_size - mat.shape[-2]), "constant", torch.nan)
                    print(mat.shape)
                correlate_matrices.append(mat)
            correlate_matrices = torch.stack(correlate_matrices)
            correlate_matrix_list.append(correlate_matrices)
        # create torch tensor
        self.correlate_matrix = correlate_matrix_list

       

    def is_in_masks(self, transform_w2c, fx, fy, cx, 
                    cy, points_world, points_masks, masks, ray_indices):
        transform_w2c = transform_w2c.to(points_world.device)

        masks = masks.to(points_world.device)
        # TODO consider the cone of ray and uncertainty of the mask and its edges
        rotation, translation = unpack_4x4_transform(transform_w2c)
        # TODO check if the transform is correct
        points_cam = points_world @ rotation.T + translation # [N,S,3]
        debugprint("points_cam", points_cam.shape)
        x = points_cam[..., 0]
        y = points_cam[..., 1]
        z = -points_cam[..., 2]
        debugprint("x", x.shape)
        pixel_i = fx * ( x / ( z + 1e-12)) + cx
        pixel_j = -fy * ( y / ( z + 1e-12)) + cy
        # find the closest integer pixels
        pixel_i = torch.round(pixel_i).long()
        pixel_j = torch.round(pixel_j).long()

        res = torch.zeros_like(pixel_i).bool()

        # clip the pixel_i and pixel_j to the image size
        # pixel_i = torch.clip(pixel_i, min=0, max=masks.shape[1]-1)
        # pixel_j = torch.clip(pixel_j, min=0, max=masks.shape[0]-1)

        # check first if the pixel is in the image
        is_in_image = (pixel_i >= 0) & (pixel_i < masks.shape[1]) & (pixel_j >= 0) & (pixel_j < masks.shape[0])

        if self.regularize_mode == "frustum":
            res[is_in_image] = True
            return res
        
        # check if the pixel is in the mask
        debugprint("is_in_image", is_in_image.shape)
        debugprint("pixel_i", pixel_i.shape)
        debugprint("masks", masks.shape)
        debugprint("points_masks", points_masks.shape)
            
        points_is_in_masks = masks[pixel_j[is_in_image], pixel_i[is_in_image],:]

        if self.regularize_mode == "single_mask":
            res[is_in_image] = torch.any(points_is_in_masks, dim=-1)
            return res

        debugprint("points_is_in_masks",points_is_in_masks.shape)
        debugprint("point count in mask", torch.sum(points_is_in_masks))
        if ray_indices is not None:
            expanded_points_masks = points_masks[ray_indices]
        else:
            expanded_points_masks = points_masks.unsqueeze(1).expand(-1,points_cam.shape[1],-1)
        debugprint("expanded_points_masks", expanded_points_masks.shape)
        matched_masks = expanded_points_masks[is_in_image]  & points_is_in_masks

        # if any of the masks is matched, then the point is in the mask
        matched_masks = torch.any(matched_masks, dim=1)

        debugprint("res", res.shape)
        debugprint("matched_masks", matched_masks.shape)
        res[is_in_image] = matched_masks
        return res


    def is_in_masks_new(self, transform_w2c, fx, fy, cx,\
            cy, points_world, points_masks_source,
            masks_target, ray_indices):
        
        
        transform_w2c = transform_w2c.to(points_world.device)
        masks_target = masks_target.to(points_world.device)
        # TODO consider the cone of ray and uncertainty of the mask and its edges
        rotation, translation = unpack_4x4_transform(transform_w2c)
        # TODO check if the transform is correct
        points_cam = points_world @ rotation.T + translation # [N,S,3]
        debugprint("points_cam", points_cam.shape)
        x = points_cam[..., 0] # [N,S]
        y = points_cam[..., 1]
        z = -points_cam[..., 2]
        debugprint("x", x.shape)
        pixel_i = fx * ( x / ( z + 1e-12)) + cx
        pixel_j = -fy * ( y / ( z + 1e-12)) + cy
        # find the closest integer pixels
        # pixel_i = torch.round(pixel_i).long()
        # pixel_j = torch.round(pixel_j).long()
        pixel_i_normalized = pixel_i / (masks_target.shape[1]-1)*2 -1
        pixel_j_normalized = pixel_j / (masks_target.shape[0]-1)*2 -1
        pixel_i = torch.round(pixel_i).long()
        pixel_j = torch.round(pixel_j).long()

        pixel_coordinate = torch.stack([pixel_i_normalized,pixel_j_normalized], dim=-1) # [N,S,2]
        # use grid sample instead
        debugprint("pixel_coordinate", pixel_coordinate.shape)  
        # masks # [H, W]

        # ray_indices # N, 3

        # points_masks # [N,]
        is_in_image = (pixel_i >= 0) & (pixel_i < masks_target.shape[1]) & (pixel_j >= 0) & (pixel_j < masks_target.shape[0])

        mask_value_in_image = masks_target[pixel_j[is_in_image], pixel_i[is_in_image]] 

        

        # view_ind_source = ray_indices[:,0].to(points_masks_source.device)

        if points_masks_source is None or self.config.regularize_mode == "frustum":
            res = torch.zeros_like(pixel_i).bool()
            res[is_in_image] = True
            return res
        points_masks_source = points_masks_source.to(points_world.device)
        # TODO check if use row or col
        expanded_points_source_masks = points_masks_source.unsqueeze(1).expand(-1,points_cam.shape[1],-1,-1)[is_in_image]


        #get correct indices
        rows = torch.arange(mask_value_in_image.shape[0]).to(mask_value_in_image.device)
        # ind = torch.stack([rows, mask_value_in_image], dim=-1)

        t = 0.02
        
        matches = (expanded_points_source_masks - mask_value_in_image)**2
        matches = torch.mean(matches, dim=-1)

        matched_masks = torch.min(matches,dim=-1)[0] < t

        # check if there is nan value
        assert torch.sum(torch.isnan(matched_masks)) == 0
        # 
        res = torch.zeros_like(pixel_i).bool()

        res[is_in_image] = matched_masks.bool()

        return res

    
    def __call__(self, ray_samples_list, weights_list, image_masks, ray_indices, mask_count_thresh=None) -> Any:
        
        # TODO check the shape
        # get point pos from ray sample list
        pos = ray_samples_list[-1].frustums.get_positions()

        # get weight from weight list
        weights = weights_list[-1][..., 0]


        # check the shape of pos and weights
        debugprint("pos shape", pos.shape)
        debugprint("weights shape", weights.shape)


        in_mask_counts = torch.zeros_like(weights).to(pos.device)
        with torch.no_grad():
            for i in range(len(self.transforms_w2c)):

                in_mask_counts += self.is_in_masks_new(self.transforms_w2c[i],
                                                self.fx[i],
                                                self.fy[i],
                                                self.cx[i],
                                                self.cy[i],                                               
                                                pos, image_masks, 
                                                self.mask_loader[i],                                                
                                                ray_indices)

        if mask_count_thresh is None:
            mask_count_thresh = self.config.mask_count_thresh
            
        # calculate the loss
        penalty_size = mask_count_thresh - in_mask_counts
        penalty_size = torch.clip(penalty_size, min=0)
        loss = self.config.reg_strength * weights * penalty_size

        return loss.mean()
    