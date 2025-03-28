from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.models.instant_ngp import InstantNGPModelConfig
from nerfstudio.models.vanilla_nerf import VanillaModelConfig

from regseg_splatfacto.sam_gaussian_splatting import SAMGaussianSplattingModelConfig, SAMGaussianSplattingModel

from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig, FullImageDatamanager
from regseg_datasets.LLFF_dataparser import LLFFDataParserConfig
from regseg_datasets.LLFF_dataset import LLFFDataset
from regseg_datasets.SAM_dataparser import SAMDataParserConfig
from regseg_datasets.SAM_dataset import SAMDataset
from regseg_splatfacto.sam_gaussian_pipeline import SAMGaussianPipelineConfig, SAMGaussianPipeline
from regseg_splatfacto.sam_gaussian import SAMSplatfactoModel, SAMSplatfactoModelConfig

from regseg_splatfacto.sam_gaussian_group import GroupSAMSplatfactoModel, GroupSAMSplatfactoModelConfig
from regseg_splatfacto.sam_group_gaussian_pipeline import GroupSAMGaussianPipelineConfig, GroupSAMGaussianPipeline

sam_gaussian_config = MethodSpecification(
    TrainerConfig(
        method_name="sam-gaussian",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        gradient_accumulation_steps={"camera_opt": 100},
        pipeline=SAMGaussianPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                _target = FullImageDatamanager[LLFFDataset],
                dataparser=LLFFDataParserConfig(load_3D_points=True),
            ),
            model=SAMSplatfactoModelConfig(),
        ),
        optimizers={
            "xyz": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacity": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scaling": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "rotation": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Train SAM with Gaussian splatting",
)


sam_arc_gaussian_config = MethodSpecification(
    TrainerConfig(
        method_name="sam-arc",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        gradient_accumulation_steps={"camera_opt": 100},
        pipeline=SAMGaussianPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                _target = FullImageDatamanager[SAMDataset],
                dataparser=SAMDataParserConfig(load_3D_points=True),
            ),
            model=SAMSplatfactoModelConfig(),
        ),
        optimizers={
            "xyz": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacity": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scaling": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "rotation": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Train SAM with Gaussian splatting",
)


sam_group_gaussian_config = MethodSpecification(
    TrainerConfig(
        method_name="regseg-splatfacto",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=22000,
        mixed_precision=False,
        gradient_accumulation_steps={"camera_opt": 100},
        pipeline=GroupSAMGaussianPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                _target = FullImageDatamanager[SAMDataset],
                dataparser=SAMDataParserConfig(load_3D_points=True),
            ),
            model=GroupSAMSplatfactoModelConfig(),
        ),
        optimizers={
            "xyz": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "features_group": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 , eps=1e-15),
                "scheduler": None,
            },
            "opacity": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scaling": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "rotation": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
            },
            "mask_opt": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Train SAM with Gaussian splatting",
)

