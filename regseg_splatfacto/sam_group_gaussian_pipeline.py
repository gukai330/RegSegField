from dataclasses import dataclass

from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig, VanillaPipeline, Pipeline
# from regseg_nerfacto.mask_regularizer import InMaskRegularizerConfig
# from regseg_nerfacto.sam_dataloader import SAMRegularizerConfig

from nerfstudio.utils import profiler  
# from regseg_nerfacto.total_variation_loss import tv_loss

from typing import Type
from dataclasses import dataclass, field
import torch

from nerfstudio.viewer.viewer_elements import ViewerDropdown

from regseg_splatfacto.mesh_visualizer import MeshVisualizer

from typing import Any, Callable, List, Literal, Optional, Tuple, Type, Union
from nerfstudio.data.datamanagers.base_datamanager import DataManager
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager

from torch.cuda.amp.grad_scaler import GradScaler
from nerfstudio.models.base_model import Model, ModelConfig
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import typing

import numpy as np

from nerfstudio.cameras.cameras import Cameras

from time import perf_counter


@dataclass
class GroupSAMGaussianPipelineConfig(VanillaPipelineConfig):

    _target: Type = field(default_factory=lambda: GroupSAMGaussianPipeline)
    """Target class for the pipeline."""
    train_rgb_until: int = 20000
    """Until which step to train rgb."""
    use_reg: bool = True
    """Whether to use the regularizer."""
    reg_steps: int = 8000
    """Number of regularizer steps."""   
    reg_from: int = 1500
    """From which step to start regularizing."""
    reg_every: int = 1
    """Regularize every N steps."""    
    reg_weight: float = 5e-4
    """Regularization weight."""
    learn_grouping: bool = True
    """Whether to learn the grouping network."""
    grouping_from: int = 1000
    """From which step to start learning the grouping network."""
    initialize_grouping_until: int = 1200
    """Until which step to initialize the grouping network."""
    grouping_unitl: int = 6000
    """Until which step to learn the grouping network."""
    train_mask_encoder: bool = False
    """Whether to train the mask encoder."""
    use_tv_loss: bool = False
    """Whether to use the total variation loss."""
    use_tv_from: int = 3000
    """From which step to start using the total variation loss."""
    tv_weight: float = 0.05
    """Total variation weight."""
    reg_expanded_views: bool = False
    """Whether to regularize the expanded views."""
    padding_size: int = 20
    """Padding size."""


    use_depth: bool = False
    """Whether to use the depth loss."""
    use_depth_from: int = 1000
    """From which step to start using the depth loss."""
    use_depth_until: int = 6500
    """Until which step to use the depth loss."""
    depth_weight: float = 0.05
    """Depth weight."""


class GroupSAMGaussianPipeline(VanillaPipeline):

    config: GroupSAMGaussianPipelineConfig
    def __init__(self, config: GroupSAMGaussianPipelineConfig, 
                 device: str,
                 test_mode: Literal["test", "val", "inference"] = "val",
                 world_size: int = 1,
                 local_rank: int = 0,
                 grad_scaler: Optional[GradScaler] = None,
                 *args, **kwargs):
        
        # initialize the Pipeline
        Pipeline.__init__(self)

        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        # TODO make cleaner
        seed_pts = None
        random_seed_pts = None

        if (
            hasattr(self.datamanager, "train_dataparser_outputs")
            and "points3D_xyz" in self.datamanager.train_dataparser_outputs.metadata
        ):
            pts = self.datamanager.train_dataparser_outputs.metadata["points3D_xyz"]
            pts_rgb = self.datamanager.train_dataparser_outputs.metadata["points3D_rgb"]
            seed_pts = (pts, pts_rgb)

            if "camera_based_points" in self.datamanager.train_dataparser_outputs.metadata:
                random_seed_pts = self.datamanager.train_dataparser_outputs.metadata["camera_based_points"]
                
        self.datamanager.to(device)
        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        # get the mask_matcher
        mask_loader = self.datamanager.train_dataset.maskloader

        self.mask_loader = mask_loader
    

        if mask_loader is not None:
            self.mask_loader.set_device(device)
            mask_encoder = mask_loader.get_model()
        else:
            mask_encoder = None

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
            seed_points=seed_pts,
            seed_points_random = random_seed_pts,
            mask_encoder = mask_encoder,
            train_rgb_until = config.train_rgb_until

        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])


        if self.config.use_reg:
            # self.regularizer = self.setup_regularizer()
            self.setup_regularizer()
            self.reg_steps = config.reg_steps
            self.reg_weight = config.reg_weight
            self.reg_from = config.reg_from

            self.init_viewer_items()

        

    def init_viewer_items(self):
        

        train_dataset = self.datamanager.train_dataset

        meshes = train_dataset._dataparser_outputs.meshes
        if meshes is not None:
            self.mesh_visualizer = MeshVisualizer(meshes)
        else:
            self.mesh_visualizer = None

    def setup_regularizer(self):
        # load all the masks
        train_dataset = self.datamanager.train_dataset
        num_views = len(train_dataset)

        if self.mask_loader is not None:
            masks_all_views = self.mask_loader.all_descriptors
        else:
            masks_all_views = [train_dataset.get_masks(i) for i in range(num_views)]
            # train_images = [train_dataset[i]["image"] for i in range(num_views)]
        self.masks_all_views = masks_all_views
        
        self.update_masks()
        self.train_cameras = self.datamanager.train_dataset.cameras

        if self.config.reg_expanded_views:
            self.expanded_train_cameras,_ = self.expand_cameras(self.train_cameras)
        

    def update_masks(self):

        if self.config.reg_expanded_views:
            
            self.padded_masks = []
            # pad the masks
            padding_size = self.config.padding_size
            for mask in self.masks_all_views:
                mask = mask.permute(2,3,0,1)                
                self.padded_masks.append(torch.nn.functional.pad(mask, (padding_size, padding_size, padding_size, padding_size), "reflect").permute(2,3,0,1))


    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
       
        do_reg_step = self.config.use_reg and step < self.reg_steps \
            and step > self.reg_from and step % self.config.reg_every == 0
        
        ray_bundle, batch = self.datamanager.next_train(step)
        
        if do_reg_step and self.config.reg_expanded_views:
            # expand the camera
            ray_bundle_expanded, padded_pixel = self.expand_cameras(ray_bundle, padding_pixels = self.config.padding_size)
            model_outputs_expanded = self._model(ray_bundle_expanded)  # train distributed data parallel model if world_size > 1


            model_outputs = {}

            padding_size = self.config.padding_size
            for key in model_outputs_expanded.keys():
                if key == "visibility":
                    model_outputs[key] = model_outputs_expanded[key]
                else:
                    model_outputs[key] = model_outputs_expanded[key][padding_size:-padding_size, padding_size:-padding_size]
            
        else:
            model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
                
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        image_idx = batch["image_idx"]
        
        if self.mask_loader is not None:
            source_mask = self.mask_loader[image_idx].to(self.device)
        else:
            source_mask = batch["masks"].to(self.device)

        if step < self.config.train_rgb_until:
            loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        
            # add mask regularization
            

            if do_reg_step:

                if self.config.reg_expanded_views:
                    source_mask_reg = self.padded_masks[image_idx].to(self.device)
                    train_cameras = self.expanded_train_cameras

                    out_image = model_outputs_expanded["rgb"]
                    out_vis = model_outputs_expanded["visibility"]
                    masks_all_views = self.padded_masks
                else:
                    source_mask_reg = source_mask
                    train_cameras = self.train_cameras
                    out_image = model_outputs["rgb"]
                    out_vis = model_outputs["visibility"]
                    masks_all_views = self.masks_all_views

                if "camera_closeness_indices" in batch:
                    camera_closeness_indices = batch["camera_closeness_indices"]
                else:
                    camera_closeness_indices = None

                loss = self.model.get_vis_loss(train_cameras, out_image, out_vis, source_mask_reg, masks_all_views, step, camera_closeness_indices)
                loss_dict["mask_visibility_loss"] = loss * self.reg_weight * np.interp(step, [0,1000, 2000, 8000], [2,0,2,0.1])
            
                

                # if step % 303 == 0:
                #     self.model.visualize_visibility(batch["image_idx"], self.mask_loader.get_raw_mask(batch['image_idx']), source_mask, self.masks_all_views, train_cameras, camera_closeness_indices, self.datamanager.train_dataset, 100)
    
        else:
            loss_dict = {}

        if self.config.use_tv_loss and step > self.config.use_tv_from and step < self.config.reg_steps:
            unique_mask = self.mask_loader.get_unique_mask(image_idx)
            tv_loss = self.model.get_tv_loss(model_outputs, unique_mask)
            loss_dict["tv_loss"] = tv_loss * self.config.tv_weight


        if self.config.use_depth and step > self.config.use_depth_from and step < self.config.use_depth_until:

            depth_loss = self.model.get_depth_loss(model_outputs, batch)
            loss_dict["depth_loss"] = depth_loss * self.config.depth_weight



        train_grouping_field = self.config.learn_grouping and step > self.config.grouping_from and step < self.config.grouping_unitl

        train_grouping_after_rgb = self.config.learn_grouping and step >= self.config.train_rgb_until

        save_grouping = (step == (self.config.train_rgb_until + 1500)) or (step == 500)

        if train_grouping_field or train_grouping_after_rgb:   
            camera = ray_bundle
            image_index = camera.metadata["cam_idx"]

            if self.mask_loader is not None:
                feature_mask = source_mask
                source_mask = self.mask_loader.get_raw_mask(image_index)


            # if step == self.config.grouping_from + 1:
               
            #     if self.mask_loader is not None:
            #         feature_mask = source_mask
            #         source_mask = self.mask_loader.get_raw_mask(image_index)


            #     else:
            #         self.model.initialize_groups()
            
            if step > self.config.grouping_from and \
                step < self.config.initialize_grouping_until\
                    and self.mask_loader is not None:
             

                group_render_loss = self.model.get_group_render_loss(model_outputs, feature_mask)
                loss_dict.update(group_render_loss) 

            elif step == self.config.grouping_from + 1:
                self.model.initialize_groups()
            
            else:
                binary_mask = None
                
                if "mask" in batch:
                    binary_mask = batch["mask"].to(self.device)

                unique_mask = self.mask_loader.get_unique_mask(image_index)
                group_loss_dict = self.model.get_group_loss(model_outputs, unique_mask, binary_mask)
                # group_loss_dict = self.model.pointwise_group_loss(model_outputs, source_mask)

                
                if self.config.train_mask_encoder:
                    feature_mask = self.mask_loader.encode_mask(image_index)
                    group_render_loss_dict = self.model.get_group_render_loss(model_outputs, feature_mask,True)
                    group_loss_dict.update(group_render_loss_dict)

                    if step % 500 == 0:
                        # update all the descriptors for the regularizer
                        self.mask_loader.update_all_descriptors()
                        self.masks_all_views = self.mask_loader.all_descriptors
                        self.update_masks()

                loss_dict.update(group_loss_dict)
        
        if save_grouping:

            initial_model_path = str(self.mask_loader.model_path)

            optimized_model_path = initial_model_path.replace(".pth", "_op.pth")
            
            print("mask path", initial_model_path, optimized_model_path)
            torch.save(self.mask_loader.get_model().state_dict(), optimized_model_path)

        # if step > 2000:
        #     self.visualize_depth(model_outputs, batch)

        # if True:
        #     # occlusion loss

        #     density = model_outputs["density"]
        #     penality = torch.zeros_like(density)
        #     penality[:,20:] = 0
        #     penality[:,:10] = 1.0
        #     loss_dict["occlusion_loss"] = torch.mean(density*penality)*1e-1
            # tv, std = tv_loss(
            #     model_outputs["depth"], 
            #     batch["masks"],
            #     patch_size=8,
            # )
            
            # loss_dict["tv_loss"] = tv
            # loss_dict["std_loss"] = std

            # print("tv_loss", loss_dict["tv_loss"])
            # print("std_loss", loss_dict["std_loss"])
            # loss_dict["sam_similarity_loss"] = self.sam_regularizer(
            #     model_outputs["ray_samples_list"],
            #     model_outputs["weights_list"],
            #     batch["indices"],
            #     model_outputs["ray_indices"] if "ray_indices" in model_outputs else None,
            # )

            

        return model_outputs, loss_dict, metrics_dict
    
    def visualize_depth(self, model_outputs, batch):
        
        with torch.no_grad():
            midas_depth = batch["depth_image"].squeeze(-1)
            rendered_depth = model_outputs["depth"].squeeze(-1)

            import matplotlib.pyplot as plt

            fig, axs = plt.subplots(1, 2)
            im1 = axs[0].imshow(midas_depth.cpu().numpy())
            axs[0].set_title("Midas Depth")
            # colorbar
            

            im2 = axs[1].imshow(rendered_depth.detach().cpu().numpy())
            axs[1].set_title("Rendered Depth")

            fig.colorbar(im1, ax=axs[0], fraction=0.046, pad=0.04)
            fig.colorbar(im2, ax=axs[1], fraction=0.046, pad=0.04)

            plt.show()

    def expand_cameras(self, cameras, padding_pixels = 20):
        
        new_height = cameras[0].image_height.item() + 2 * padding_pixels
        new_width = cameras[0].image_width.item() + 2 * padding_pixels

        wider_cameras = Cameras(
            camera_to_worlds=cameras.camera_to_worlds,
            fx=cameras.fx,
            fy=cameras.fy,
            cx=cameras.cx + padding_pixels,
            cy=cameras.cy + padding_pixels,
            height=new_height,
            width=new_width,
            camera_type = cameras.camera_type,
        )

        padding_pixel_mask = torch.zeros(new_height, new_width, dtype=torch.bool)
        padding_pixel_mask[:padding_pixels, :] = True
        padding_pixel_mask[-padding_pixels:, :] = True
        padding_pixel_mask[:, :padding_pixels] = True
        padding_pixel_mask[:, -padding_pixels:] = True

        return wider_cameras, padding_pixel_mask
    