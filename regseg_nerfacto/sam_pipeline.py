from dataclasses import dataclass

from nerfstudio.pipelines.base_pipeline import *
from regseg_splatfacto.sam_group_gaussian_pipeline import GroupSAMGaussianPipelineConfig
from regseg_nerfacto.mask_regularizer import AutoMatchRegularizerConfig
from nerfstudio.utils import profiler  
# from regseg_nerfacto.total_variation_loss import tv_loss

import torch

import numpy as np

from nerfstudio.cameras.cameras import Cameras, CameraType
import torch.nn.functional as F

from rich.console import Console

CONSOLE = Console(width=120)


@dataclass
class SAMPipelineConfig(GroupSAMGaussianPipelineConfig):

    _target: Type = field(default_factory=lambda: SAMNerfactoPipeline)
    """Target class for the pipeline."""
    regularizer: AutoMatchRegularizerConfig = AutoMatchRegularizerConfig()
    """SAM Regularizer to use for the model."""


class SAMNerfactoPipeline(VanillaPipeline):

    config: SAMPipelineConfig
    def __init__(self, config:SAMPipelineConfig,
                 device: str,
                 test_mode: Literal["test", "val", "inference"] = "val",
                 world_size: int = 1,
                 local_rank: int = 0,
                 grad_scaler: Optional[GradScaler] = None,
                  *args, **kwargs):
        
        Pipeline.__init__(self)

        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )
        # TODO make cleaner

        self.datamanager.to(device)
        # TODO(ethan): get rid of scene_bounds from the model
        assert self.datamanager.train_dataset is not None, "Missing input dataset"

        # get the mask_matcher
        mask_loader = self.datamanager.train_dataset.maskloader

        self.mask_loader = mask_loader
    
                # TODO make cleaner
        seed_pts = None
        if (
            hasattr(self.datamanager, "train_dataparser_outputs")
            and "points3D_xyz" in self.datamanager.train_dataparser_outputs.metadata
        ):
            pts = self.datamanager.train_dataparser_outputs.metadata["points3D_xyz"]
            pts_rgb = self.datamanager.train_dataparser_outputs.metadata["points3D_rgb"]
            seed_pts = (pts, pts_rgb)

        # # remove the camera closeness matrix for sampling TODO fix this for all different datamanagers
        # train_dataset = self.datamanager.train_dataset
        # eval_dataset = self.datamanager.eval_dataset

        # if "camera_closeness_matrix" in train_dataset._dataparser_outputs.metadata:
        #     # remove the camera closeness matrix
        #     del train_dataset._dataparser_outputs.metadata["camera_closeness_matrix"]
        # if eval_dataset is not None:
        #     if "camera_closeness_matrix" in eval_dataset._dataparser_outputs.metadata:
        #         # remove the camera closeness matrix
        #         del eval_dataset._dataparser_outputs.metadata["camera_closeness_matrix"]

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
            mask_encoder = mask_encoder,
            train_rgb_until = config.train_rgb_until

        )

        self.model.to(device)
        if world_size > 1:
            self._model = typing.cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))
            dist.barrier(device_ids=[local_rank])

        if self.config.use_reg:
            self.setup_regularizer()

    def setup_regularizer(self):
        # load all the masks
        train_dataset = self.datamanager.train_dataset
        # num_views = len(train_dataset)
        # masks_all_views = [train_dataset.get_masks(i) for i in range(num_views)]
        # train_images = [train_dataset[i]["image"] for i in range(num_views)]

        self.regularizer = self.config.regularizer.setup(
            cameras = self.datamanager.train_dataset.cameras,
            mask_loader = self.mask_loader,
        )
        # self.sam_regularizer = self.config.sam_regularizer.setup(
        #     cameras = self.datamanager.train_dataset.cameras,
        #     train_images = train_images,
        #     device = self.device,
        # )

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, batch = self.datamanager.next_train(step)
        model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        ray_indices = batch["indices"]
        # sample from the feature masks
        if self.mask_loader is not None:
            source_mask_bundle = self.mask_loader.sample_masks(ray_indices)

        else:
            source_mask_bundle = None

        if step < self.config.train_rgb_until:
            loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        
            # add mask regularization
            

            if self.config.use_reg and step < self.config.reg_steps \
                and step > self.config.reg_from and step % self.config.reg_every == 0:


                # add mask regularization
            
                loss_dict["mask_visibility_loss"] = self.regularizer(
                    ray_samples_list=model_outputs["ray_samples_list"],
                    weights_list=model_outputs["weights_list"],
                    image_masks=source_mask_bundle,
                    ray_indices=model_outputs["ray_indices"] if "ray_indices" in model_outputs else None,
                    mask_count_thresh=2
                ) * self.config.reg_weight * np.interp(step, [0,1000, 2000, 3000], [2,0,2,0.1])


                if step % 50 == 0:
                    loss_dict["global_visibility_loss"] = self.reg_step_all(False)
        
        else:
            loss_dict = {}

        raw_masks_padded = None
        if self.config.use_tv_loss and step > 200 and step < self.config.reg_steps:
            raw_masks_padded = self.mask_loader.sample_nested_masks(ray_indices)
            tv_loss = self.model.get_tv_loss(model_outputs, raw_masks_padded)
            loss_dict["tv_loss"] = tv_loss

        train_grouping_field = self.config.learn_grouping and step > self.config.grouping_from and step < self.config.grouping_unitl

        train_grouping_after_rgb = self.config.learn_grouping and step >= self.config.train_rgb_until

        if train_grouping_field or train_grouping_after_rgb:               

            if raw_masks_padded is None:
                
                # raw_masks_list, bundle_binary_indices = self.mask_loader.sample_raw_masks(ray_indices)
                raw_masks_padded = self.mask_loader.sample_nested_masks(ray_indices)


            # if step == self.config.grouping_from + 1:
               
            #     if self.mask_loader is not None:
            #         feature_mask = source_mask
            #         source_mask = self.mask_loader.get_raw_mask(image_index)


            #     else:
            #         self.model.initialize_groups()
            
            if step > self.config.grouping_from and \
                step < self.config.initialize_grouping_until\
                    and self.mask_loader is not None and self.config.train_mask_encoder:
             

                group_render_loss = self.model.get_group_render_loss(model_outputs, source_mask_bundle,False)
                loss_dict.update(group_render_loss) 

            # elif step == self.config.grouping_from + 1:
            #     self.model.initialize_groups()
            
            else:
                binary_mask = None
                
                if "mask" in batch:
                    binary_mask = batch["mask"].to(self.device)

                # group_loss_dict = self.model.get_group_loss(model_outputs, raw_masks_list, bundle_binary_indices)
                n_views = len(self.mask_loader.all_descriptors)

                group_loss_dict = self.model.get_group_loss(model_outputs, raw_masks_padded, ray_indices.shape[0]//n_views//4)

                
                if self.config.train_mask_encoder:
                    
                    source_mask_bundle = self.mask_loader.sample_masks_with_grad(ray_indices)
                    group_render_loss_dict = self.model.get_group_render_loss(model_outputs, source_mask_bundle,True)
                    group_loss_dict.update(group_render_loss_dict)

                    if step % 500 == 0:
                        # update all the descriptors for the regularizer
                        self.mask_loader.update_all_descriptors()
                        self.masks_all_views = self.mask_loader.all_descriptors

                loss_dict.update(group_loss_dict)
        

    

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

        for k, v in loss_dict.items():
            # check if any loss is nan
            assert not torch.isnan(v).any()

        return model_outputs, loss_dict, metrics_dict
    

    def reg_step_all(self, whole_view=False):

        cameras = self.datamanager.train_dataset.cameras


        padding_pixels = 20

        new_height = cameras[0].image_height.item() + 2 * padding_pixels
        new_width = cameras[0].image_width.item() + 2 * padding_pixels
        # create a set of omni-directional cameras 
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

        total_loss = 0.0

        if whole_view:
            padding_pixel_mask = torch.ones(new_height, new_width, dtype=torch.bool)
        else:
            padding_pixel_mask = torch.zeros(new_height, new_width, dtype=torch.bool)
            padding_pixel_mask[:padding_pixels, :] = True
            padding_pixel_mask[-padding_pixels:, :] = True
            padding_pixel_mask[:, :padding_pixels] = True
            padding_pixel_mask[:, -padding_pixels:] = True
        
        for i, camera in enumerate(wider_cameras):
            ray_bundle = camera.generate_rays(0)

            
            # get valid ray_bundle
            ray_bundle = ray_bundle[padding_pixel_mask]
            rand_idxs = torch.randperm(len(ray_bundle))[:4096]  
            # randomling sample 4096 rays from the ray_bundle
            
            
            source_mask = self.mask_loader[i]

            source_mask = source_mask.permute(2,3,0,1)
            # pad the mask
            padded_mask = F.pad(source_mask, (padding_pixels, padding_pixels, padding_pixels, padding_pixels), 
                                "reflect")
            padded_mask = padded_mask.permute(2,3,0,1)
            
            valid_mask = padded_mask[padding_pixel_mask]

            ray_bundle = ray_bundle[rand_idxs]
            valid_mask = valid_mask[rand_idxs]

            model_outputs = self._model(ray_bundle.to(self.device))  # train distributed data parallel model if world_size > 1
            loss = self.regularizer(
                ray_samples_list=model_outputs["ray_samples_list"],
                weights_list=model_outputs["weights_list"],
                image_masks=valid_mask,
                ray_indices=model_outputs["ray_indices"] if "ray_indices" in model_outputs else None,
            )

            total_loss += loss

        return total_loss * 1e-3
            


    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline.

        Returns:
            A list of dictionaries containing the pipeline's param groups.
        """
        param = super().get_param_groups()

        if self.mask_loader is not None:
            param["mask_opt"] = list(self.mask_loader.get_model().parameters())

        return param