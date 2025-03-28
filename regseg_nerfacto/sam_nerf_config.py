from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.models.instant_ngp import InstantNGPModelConfig
from nerfstudio.models.vanilla_nerf import VanillaModelConfig

from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig, VanillaDataManager

from regseg_datasets.LLFF_dataparser import LLFFDataParserConfig
from regseg_datasets.LLFF_dataset import LLFFDataset
from regseg_datasets.SAM_dataparser import SAMDataParserConfig
from regseg_datasets.regseg_dataset import SAMDataset

from regseg_nerfacto.group_sam_nerfacto import GroupSAMNerfactoModelConfig
from regseg_nerfacto.sam_pipeline import SAMPipelineConfig, SAMNerfactoPipeline

from nerfstudio.data.pixel_samplers import PairPixelSamplerConfig
# from nerfstudio.models.nerfacto import NerfactoModelConfig

from regseg_nerfacto.batch_pixel_sampler import BatchPairPixelSamplerConfig


sam_nerfacto_config = MethodSpecification(
    
    TrainerConfig(
        method_name="sam-nerf",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        max_num_iterations=22000,
        steps_per_eval_all_images=2000,
        mixed_precision=True,
        pipeline=SAMPipelineConfig(
            _target = SAMNerfactoPipeline,
            datamanager=VanillaDataManagerConfig(
                _target=VanillaDataManager[SAMDataset],
                dataparser=SAMDataParserConfig(),
                pixel_sampler=BatchPairPixelSamplerConfig(),
                train_num_rays_per_batch=int(25*3*2*26),
                eval_num_rays_per_batch=int(25*3*2*26),
                
            ),
            model=GroupSAMNerfactoModelConfig(eval_num_rays_per_chunk=1 << 15,
                                              camera_optimizer=CameraOptimizerConfig(mode="off"),
                                              ),
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
            },
            "grouping":{
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15, weight_decay=1e-6,max_norm=1.0),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=200000),
            },
            "mask_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-5, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="sam nerfacto",
)
