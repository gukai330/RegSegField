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

@dataclass
class SAMGaussianPipelineConfig(VanillaPipelineConfig):

    _target: Type = field(default_factory=lambda: SAMGaussianPipeline)
    """Target class for the pipeline."""
    use_reg: bool = True
    """Whether to use the regularizer."""
    reg_steps: int = 6000
    """Number of regularizer steps."""
    reg_every: int = 1
    """Regularize every N steps."""    
    reg_weight: float = 5e-3
    """Regularization weight."""



class SAMGaussianPipeline(VanillaPipeline):

    config: SAMGaussianPipelineConfig
    def __init__(self, config: SAMGaussianPipelineConfig, 
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

        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
            seed_points=seed_pts,
            seed_points_random = random_seed_pts
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
            self.warmup_steps = 0

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
        masks_all_views = [train_dataset.get_masks(i) for i in range(num_views)]
        # train_images = [train_dataset[i]["image"] for i in range(num_views)]


        self.masks_all_views = masks_all_views

        # self.regularizer = self.config.regularizer.setup(
        #     cameras = self.datamanager.train_dataset.cameras,
        #     masks_all_views = masks_all_views,
        # )
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

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        # add mask regularization



        if self.config.use_reg and step < self.reg_steps \
            and step > self.warmup_steps and step % self.config.reg_every == 0:

            train_cameras = self.datamanager.train_dataset.cameras
            out_image = model_outputs["rgb"]
            out_vis = model_outputs["visibility"]
            source_mask = batch["masks"].to(self.device)
            

            loss = self.model.get_vis_loss(train_cameras, out_image, out_vis, source_mask, self.masks_all_views, step)
            loss_dict["mask_visibility_loss"] = loss * self.reg_weight

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