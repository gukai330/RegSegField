# ruff: noqa: E741
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

"""
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from gsplat._torch_impl import quat_to_rotmat
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from gsplat.sh import num_sh_bases, spherical_harmonics
from pytorch_msssim import SSIM
from torch.nn import Parameter
from typing_extensions import Literal

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers

# need following import for background color override
from nerfstudio.model_components import renderers
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.rich_utils import CONSOLE

from torchmetrics import PearsonCorrCoef
from torchmetrics.functional.regression import pearson_corrcoef

import torch.nn.functional as F

from nerfstudio.models.splatfacto import SplatfactoModel

from mask_matching.mask_matching import MaskPermutation

import tqdm

import time

from nerfstudio.utils.colormaps import apply_pca_colormap

from mask_matching.mask_utils import binary_erosion


# from cuml.cluster.hdbscan import DBSCAN

def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def projection_matrix(znear, zfar, fovx, fovy, device: Union[str, torch.device] = "cpu"):
    """
    Constructs an OpenGL-style perspective projection matrix.
    """
    t = znear * math.tan(0.5 * fovy)
    b = -t
    r = znear * math.tan(0.5 * fovx)
    l = -r
    n = znear
    f = zfar
    return torch.tensor(
        [
            [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, (f + n) / (f - n), -1.0 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=device,
    )


@dataclass
class GroupSAMSplatfactoModelConfig(ModelConfig):
    """Splatfacto Model Config, nerfstudio's implementation of Gaussian Splatting"""

    _target: Type = field(default_factory=lambda: GroupSAMSplatfactoModel)
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 250
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "random"
    """Whether to randomize the background color."""
    num_downscales: int = 0
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling huge gaussians"""
    continue_cull_post_densification: bool = True
    """If True, continue to cull gaussians post refinement"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0002
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 4000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.0
    "Size of the cube to initialize random gaussians within"
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_split_at: int = 15000
    """stop splitting at this step"""
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    use_scale_regularization: bool = False
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 10.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    mask_count_thresh: int = 1
    """threshold of mask count before applying regularization loss"""
    use_depth: bool = False    
    """Whether to use depth in the loss"""
    learn_group_feature: bool = True
    """Whether to learn the group feature"""
    group_feature_size: int = 16
    """Number of feature channels in the group feature"""
    group_feature_level: int = 4
    """Number of levels in the group feature"""
    use_group_initialization: bool = False
    """Whether to use group initialization"""
    use_binary_mask_for_training: bool = False
    """Whether to use binary mask for training"""
    use_binary_mask_for_testing: bool = False
    """Whether to use binary mask for testing"""

class GroupSAMSplatfactoModel(SplatfactoModel):
    """Nerfstudio's implementation of Gaussian Splatting

    Args:
        config: Splatfacto configuration to instantiate model
    """

    config: GroupSAMSplatfactoModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mask_encoder: Optional[MaskPermutation] = None, 
        train_rgb_until: int = 30000,
        **kwargs,
    ):
        self.seed_points = seed_points
        # check if seed_points_random in kwargs

        if "seed_points_random" in kwargs:
            self.seed_points_random = kwargs["seed_points_random"]
        else:
            self.seed_points_random = None

        Model.__init__(self, *args, **kwargs)
        self.mask_encoder = mask_encoder
        self.train_rgb_until = train_rgb_until

    def populate_modules(self):
        if self.seed_points is not None and not self.config.random_init:
            self.means = torch.nn.Parameter(self.seed_points[0])  # (Location, Color)
        elif self.seed_points_random is not None:
            self.means = torch.nn.Parameter(self.seed_points_random)
        else:
            self.means = torch.nn.Parameter((torch.rand((self.config.num_random, 3)) - 0.5) * self.config.random_scale)
        
        self.global_visibility = torch.zeros_like(self.means[..., 0])
        

        self.visibility_cull_mask = None
        self.updated_count = 0

        self.xys_grad_norm = None
        self.max_2Dsize = None
        distances, _ = self.k_nearest_sklearn(self.means.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)
        self.scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        self.quats = torch.nn.Parameter(random_quat_tensor(self.num_points))
        dim_sh = num_sh_bases(self.config.sh_degree)


        if (
            self.seed_points is not None
            and not self.config.random_init
            # We can have colors without points.
            and self.seed_points[1].shape[0] > 0
        ):
            shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3)).float().cuda()
            if self.config.sh_degree > 0:
                shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255)
                shs[:, 1:, 3:] = 0.0
            else:
                CONSOLE.log("use color only optimization with sigmoid activation")
                shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-10)
            self.features_dc = torch.nn.Parameter(shs[:, 0, :])
            self.features_rest = torch.nn.Parameter(shs[:, 1:, :])
        else:
            self.features_dc = torch.nn.Parameter(torch.rand(self.num_points, 3))
            self.features_rest = torch.nn.Parameter(torch.zeros((self.num_points, dim_sh - 1, 3)))

        self.features_group = torch.nn.Parameter(torch.rand(self.num_points, 
                                                        self.config.group_feature_level, 
                                                        self.config.group_feature_size))

        # self.features_cluster = torch.zeros_like(self.means[..., 0])

        self.features = {
            "features_dc": self.features_dc,
            "features_rest": self.features_rest,
            "features_group": self.features_group,
            # "features_cluster": self.features_cluster  
        }

        self.opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(self.num_points, 1)))

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.rand(3)
        else:
            self.background_color = get_color(self.config.background_color)


    @property
    def colors(self):
        if self.config.sh_degree > 0:
            return SH2RGB(self.features_dc)
        else:
            return torch.sigmoid(self.features_dc)

    @property
    def shs_0(self):
        return self.features_dc

    @property
    def shs_rest(self):
        return self.features_rest
    
    @property
    def features_dc(self):
        return self.features["features_dc"]
    
    # @features_dc.setter
    # def features_dc(self, value):
    #     self.features["features_dc"] = value

    @property
    def features_rest(self):
        return self.features["features_rest"]
    
    # @features_rest.setter
    # def features_rest(self, value):
    #     self.features["features_rest"] = value

    @property
    def features_group(self):
        return self.features["features_group"]
    
    # @property
    # def features_cluster(self):
    #     return self.features["features_cluster"]
    
    # @features_group.setter
    # def features_group(self, value):
    #     self.features["features_group"] = value
    

    def run_clustering(self):
        dbscan = DBSCAN(eps=0.1, min_samples=10)
        labels = dbscan.fit_predict(self.means)

        self.features["features_cluster"] = labels

        pass

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = 30000
        newp = dict["means"].shape[0]
        self.means = torch.nn.Parameter(torch.zeros(newp, 3, device=self.device))
        self.scales = torch.nn.Parameter(torch.zeros(newp, 3, device=self.device))
        self.quats = torch.nn.Parameter(torch.zeros(newp, 4, device=self.device))
        self.opacities = torch.nn.Parameter(torch.zeros(newp, 1, device=self.device))
        self.features_dc = torch.nn.Parameter(torch.zeros(newp, 3, device=self.device))
        self.features_rest = torch.nn.Parameter(
            torch.zeros(newp, num_sh_bases(self.config.sh_degree) - 1, 3, device=self.device)
        )
        self.features_group = torch.nn.Parameter(torch.zeros(newp, 
                                                             self.config.group_feature_level,
                                                             self.config.group_feature_size, device=self.device))
        super().load_state_dict(dict, **kwargs)

    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
            Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        # assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        if "exp_avg" in param_state:

            # Modify the state directly without deleting and reassigning.
            param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
            param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state

    def remove_from_all_optim(self, optimizers, deleted_mask):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)
        torch.cuda.empty_cache()

    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]

        if "exp_avg" in param_state:
            repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))
            param_state["exp_avg"] = torch.cat(
                [
                    param_state["exp_avg"],
                    torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
            param_state["exp_avg_sq"] = torch.cat(
                [
                    param_state["exp_avg_sq"],
                    torch.zeros_like(param_state["exp_avg_sq"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def dup_in_all_optim(self, optimizers, dup_mask, n):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.dup_in_optim(optimizers.optimizers[group], dup_mask, param, n)

    def after_train(self, step: int):
        assert step == self.step
        # to save some training time, we no longer need to update those stats post refinement
        if self.step >= self.config.stop_split_at or self.step>=self.train_rgb_until:
            return
        with torch.no_grad():

            
            self.visibility_cull_mask = self.update_global_visibility()
            

            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            assert self.xys.grad is not None
            grads = self.xys.grad.detach().norm(dim=-1)
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                self.xys_grad_norm = grads
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                assert self.vis_counts is not None
                self.vis_counts[visible_mask] = self.vis_counts[visible_mask] + 1
                self.xys_grad_norm[visible_mask] = grads[visible_mask] + self.xys_grad_norm[visible_mask]

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            newradii = self.radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask],
                newradii / float(max(self.last_size[0], self.last_size[1])),
            )
            

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    def refinement_after(self, optimizers: Optimizers, step):
        assert step == self.step

        if self.step >= self.train_rgb_until:
            return

        if self.step <= self.config.warmup_length:
            
            if self.visibility_cull_mask is not None:
                cull_mask = self.visibility_cull_mask
                n_bef = self.num_points

                self.means = Parameter(self.means[~cull_mask].detach())
                self.scales = Parameter(self.scales[~cull_mask].detach())
                self.quats = Parameter(self.quats[~cull_mask].detach())
                # self.features_dc = Parameter(self.features_dc[~cull_mask].detach())
                # self.features_rest = Parameter(self.features_rest[~cull_mask].detach())
                # self.features_group = Parameter(self.features_group[~cull_mask].detach())
                for name, features in self.features.items():
                    self.features[name] = Parameter(features[~cull_mask].detach())
                self.opacities = Parameter(self.opacities[~cull_mask].detach())

                CONSOLE.log(
                    f"Culled {n_bef - self.num_points} gaussians out of any views"
                )

                self.remove_from_all_optim(optimizers, cull_mask)
                
                self.xys_grad_norm = None
                self.vis_counts = None
                self.max_2Dsize = None


            return

        with torch.no_grad():
            # Offset all the opacity reset logic by refine_every so that we don't
            # save checkpoints right when the opacity is reset (saves every 2k)
            # then cull
            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.config.reset_alpha_every * self.config.refine_every
            do_densification = (
                self.step < self.config.stop_split_at
                and self.step % reset_interval > self.num_train_data + self.config.refine_every
            )
            if do_densification:
                # then we densify
                assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                avg_grad_norm = (self.xys_grad_norm / self.vis_counts) * 0.5 * max(self.last_size[0], self.last_size[1])
                high_grads = (avg_grad_norm > self.config.densify_grad_thresh).squeeze()
                splits = (self.scales.exp().max(dim=-1).values > self.config.densify_size_thresh).squeeze()
                if self.step < self.config.stop_screen_size_at:
                    splits |= (self.max_2Dsize > self.config.split_screen_size).squeeze()
                splits &= high_grads
                nsamps = self.config.n_split_samples
                (
                    split_means,
                    split_features,
                    split_opacities,
                    split_scales,
                    split_quats,
                ) = self.split_gaussians(splits, nsamps)

                dups = (self.scales.exp().max(dim=-1).values <= self.config.densify_size_thresh).squeeze()
                dups &= high_grads
                (
                    dup_means,
                    dup_features,
                    dup_opacities,
                    dup_scales,
                    dup_quats,
                ) = self.dup_gaussians(dups)
                self.means = Parameter(torch.cat([self.means.detach(), split_means, dup_means], dim=0))
                
                # update all the features
                for name, features in self.features.items():
                    self.features[name] = Parameter(torch.cat([features.detach(), 
                                                               split_features[name], 
                                                               dup_features[name]], dim=0))
                
                
                self.opacities = Parameter(torch.cat([self.opacities.detach(), split_opacities, dup_opacities], dim=0))
                self.scales = Parameter(torch.cat([self.scales.detach(), split_scales, dup_scales], dim=0))
                self.quats = Parameter(torch.cat([self.quats.detach(), split_quats, dup_quats], dim=0))
                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros_like(split_scales[:, 0]),
                        torch.zeros_like(dup_scales[:, 0]),
                    ],
                    dim=0,
                )

                split_idcs = torch.where(splits)[0]
                self.dup_in_all_optim(optimizers, split_idcs, nsamps)

                dup_idcs = torch.where(dups)[0]
                self.dup_in_all_optim(optimizers, dup_idcs, 1)

                # After a guassian is split into two new gaussians, the original one should also be pruned.
                splits_mask = torch.cat(
                    (
                        splits,
                        torch.zeros(
                            nsamps * splits.sum() + dups.sum(),
                            device=self.device,
                            dtype=torch.bool,
                        ),
                    )
                )

                deleted_mask = self.cull_gaussians(splits_mask)
            elif self.step >= self.config.stop_split_at and self.config.continue_cull_post_densification:
                deleted_mask = self.cull_gaussians()
            else:
                # if we donot allow culling post refinement, no more gaussians will be pruned.
                deleted_mask = None

            if deleted_mask is not None:
                self.remove_from_all_optim(optimizers, deleted_mask)

            if self.step < self.config.stop_split_at and self.step % reset_interval == self.config.refine_every:
                # Reset value is set to be twice of the cull_alpha_thresh
                reset_value = self.config.cull_alpha_thresh * 2.0
                self.opacities.data = torch.clamp(
                    self.opacities.data,
                    max=torch.logit(torch.tensor(reset_value, device=self.device)).item(),
                )
                # reset the exp of optimizer
                optim = optimizers.optimizers["opacity"]
                param = optim.param_groups[0]["params"][0]
                param_state = optim.state[param]
                param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])

            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

    def cull_gaussians(self, extra_cull_mask: Optional[torch.Tensor] = None):
        """
        This function deletes gaussians with under a certain opacity threshold
        extra_cull_mask: a mask indicates extra gaussians to cull besides existing culling criterion
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (torch.sigmoid(self.opacities) < self.config.cull_alpha_thresh).squeeze()
        below_alpha_count = torch.sum(culls).item()
        toobigs_count = 0
        if extra_cull_mask is not None:
            culls = culls | extra_cull_mask
        if self.step > self.config.refine_every * self.config.reset_alpha_every:
            # cull huge ones
            toobigs = (torch.exp(self.scales).max(dim=-1).values > self.config.cull_scale_thresh).squeeze()
            if self.step < self.config.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                toobigs = toobigs | (self.max_2Dsize > self.config.cull_screen_size).squeeze()
            culls = culls | toobigs
            toobigs_count = torch.sum(toobigs).item()
        self.means = Parameter(self.means[~culls].detach())
        self.scales = Parameter(self.scales[~culls].detach())
        self.quats = Parameter(self.quats[~culls].detach())

        for name, features in self.features.items():
            self.features[name] = Parameter(features[~culls].detach())


        # self.features_dc = Parameter(self.features_dc[~culls].detach())
        # self.features_rest = Parameter(self.features_rest[~culls].detach())
        self.opacities = Parameter(self.opacities[~culls].detach())

        CONSOLE.log(
            f"Culled {n_bef - self.num_points} gaussians "
            f"({below_alpha_count} below alpha thresh, {toobigs_count} too bigs, {self.num_points} remaining)"
        )

        return culls

    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """

        n_splits = split_mask.sum().item()
        CONSOLE.log(f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quats[split_mask] / self.quats[split_mask].norm(dim=-1, keepdim=True)  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        # new_features_dc = self.features_dc[split_mask].repeat(samps, 1)
        # new_features_rest = self.features_rest[split_mask].repeat(samps, 1, 1)
        # step 2, sample new features
        new_features = {}
        for name, features in self.features.items():

            repeat_dims = tuple(1 for _ in range(features.dim() - 1))
            new_features[name] = features[split_mask].repeat(samps, *repeat_dims)
                

        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(samps, 1)
        self.scales[split_mask] = torch.log(torch.exp(self.scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self.quats[split_mask].repeat(samps, 1)
        return (
            new_means,
            new_features,
            new_opacities,
            new_scales,
            new_quats,
        )

    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        CONSOLE.log(f"Duplicating {dup_mask.sum().item()/self.num_points} gaussians: {n_dups}/{self.num_points}")
        dup_means = self.means[dup_mask]

        # duplicate all the features
        dup_features = {}
        for name, features in self.features.items():
            dup_features[name] = features[dup_mask]
        # dup_features_dc = self.features_dc[dup_mask]
        # dup_features_rest = self.features_rest[dup_mask]
        dup_opacities = self.opacities[dup_mask]
        dup_scales = self.scales[dup_mask]
        dup_quats = self.quats[dup_mask]
        return (
            dup_means,
            dup_features,
            dup_opacities,
            dup_scales,
            dup_quats,
        )

    @property
    def num_points(self):
        return self.means.shape[0]

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        # The order of these matters
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.refinement_after,
                update_every_num_iters=self.config.refine_every,
                args=[training_callback_attributes.optimizers],
            )
        )
        return cbs

    def step_cb(self, step):
        self.step = step

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:

        features_all = {}

        for name, features in self.features.items():
            features_all[name] = [features]
        
        return {
            "xyz": [self.means],
            **features_all,
            "opacity": [self.opacities],
            "scaling": [self.scales],
            "rotation": [self.quats],
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()

        if self.mask_encoder is not None:
            pass
            gps["mask_opt"] = list(self.mask_encoder.parameters())


        return gps

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max(
                (self.config.num_downscales - self.step // self.config.resolution_schedule),
                0,
            )
        else:
            return 1

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}
        assert camera.shape[0] == 1, "Only one camera at a time"

        # get the background color
        if self.training:
            if self.config.background_color == "random":
                background = torch.rand(3, device=self.device)
            elif self.config.background_color == "white":
                background = torch.ones(3, device=self.device)
            elif self.config.background_color == "black":
                background = torch.zeros(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        else:
            if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
                background = renderers.BACKGROUND_COLOR_OVERRIDE.to(self.device)
            else:
                background = self.background_color.to(self.device)

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return {"rgb": background.repeat(int(camera.height.item()), int(camera.width.item()), 1)}
        else:
            crop_ids = None
        camera_downscale = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_downscale)
        # shift the camera to center of scene looking at center
        R = camera.camera_to_worlds[0, :3, :3]  # 3 x 3
        T = camera.camera_to_worlds[0, :3, 3:4]  # 3 x 1
        # flip the z and y axes to align with gsplat conventions
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
        R = R @ R_edit
        # analytic matrix inverse to get world2camera matrix
        R_inv = R.T
        T_inv = -R_inv @ T
        viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
        viewmat[:3, :3] = R_inv
        viewmat[:3, 3:4] = T_inv
        # calculate the FOV of the camera given fx and fy, width and height
        cx = camera.cx.item()
        cy = camera.cy.item()
        fovx = 2 * math.atan(camera.width / (2 * camera.fx))
        fovy = 2 * math.atan(camera.height / (2 * camera.fy))
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        projmat = projection_matrix(0.001, 1000, fovx, fovy, device=self.device)
        BLOCK_X, BLOCK_Y = 16, 16
        tile_bounds = (
            int((W + BLOCK_X - 1) // BLOCK_X),
            int((H + BLOCK_Y - 1) // BLOCK_Y),
            1,
        )

        features_crop = {}
        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            for name, features in self.features.items():
                features_crop[name] = features[crop_ids]
            # features_dc_crop = self.features_dc[crop_ids]
            # features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            for name, features in self.features.items():
                features_crop[name] = features
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat((features_crop["features_dc"][:, None, :], features_crop["features_rest"]), dim=1)

        self.xys, depths, self.radii, conics, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
            means_crop,
            torch.exp(scales_crop),
            1,
            quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            viewmat.squeeze()[:3, :],
            projmat.squeeze() @ viewmat.squeeze(),
            camera.fx.item(),
            camera.fy.item(),
            cx,
            cy,
            H,
            W,
            tile_bounds,
        )  # type: ignore

        self.depths = depths

        if (self.radii).sum() == 0:
            return {"rgb": background.repeat(int(camera.height.item()), int(camera.width.item()), 1)}

        # Important to allow xys grads to populate properly
        if self.training:
            self.xys.retain_grad()

        if self.config.sh_degree > 0:
            viewdirs = means_crop.detach() - camera.camera_to_worlds.detach()[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors_crop)
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
        else:
            rgbs = torch.sigmoid(colors_crop[:, 0, :])

        # rescale the camera back to original dimensions
        camera.rescale_output_resolution(camera_downscale)
        assert (num_tiles_hit > 0).any()  # type: ignore
        rgb, visibility, alpha = rasterize_gaussians(  # type: ignore
            self.xys,
            depths,
            self.radii,
            conics,
            num_tiles_hit,  # type: ignore
            rgbs,
            torch.sigmoid(opacities_crop),
            H,
            W,
            background=background,
            return_alpha=True
        )  # type: ignore
        rgb = torch.clamp(rgb, max=1.0)  # type: ignore
        # depth_im = None
        depth_im = rasterize_gaussians(  # type: ignore
            self.xys,
            depths,
            self.radii,
            conics,
            num_tiles_hit,  # type: ignore
            depths[:, None].repeat(1, 3),
            torch.sigmoid(opacities_crop),
            H,
            W,
            background=torch.ones(3, device=self.device) * 10,
        )[..., 0:1]  # type: ignore


        total_feature_dims = self.config.group_feature_level * self.config.group_feature_size 
        
        features_group_im = None
       

        res = {"rgb": rgb, "depth": depth_im, "visibility": visibility, "alpha": alpha}

        if self.config.learn_group_feature:

            # render it separately and downscale it!

            features_group_crop = features_crop["features_group"]
            # rgbs_group = torch.sigmoid(features_group_crop)
            features_group_crop = F.normalize(features_group_crop, p=2, dim=-1)


            # get N best index according to the visibility
            _, n_best_indices = torch.topk(visibility, k=self.xys.shape[0]//5, dim=0)
            n_best_indices = n_best_indices.squeeze(1)

            # only propagate features to the group features
            features_group_im = rasterize_gaussians(
                self.xys.detach()[n_best_indices],
                depths.detach()[n_best_indices],
                self.radii.detach()[n_best_indices],
                conics.detach()[n_best_indices],
                num_tiles_hit.detach()[n_best_indices],  # type: ignore
                # flatten the features for n-d gaussian rendering
                features_group_crop.view(-1,total_feature_dims)[n_best_indices],
                torch.sigmoid(opacities_crop.detach()[n_best_indices]),
                # torch.ones_like(opacities_crop.detach()),
                H,
                W,
                background=torch.zeros(total_feature_dims, device=self.device),
            ).view(H,W,self.config.group_feature_level, self.config.group_feature_size)
            
            # features_group_im = torch.clamp(features_group_im, max=1.0)

            features_group_im = F.normalize(features_group_im, p=2, dim=-1)
            # scale to 0-1
            features_group_im = features_group_im * 0.5 + 0.5
            

            features_group_out = {
               f"features_group_{i}": features_group_im[...,i,:] for i in range(self.config.group_feature_level) 
            }


            res.update(features_group_out)

        return res  # type: ignore

    @torch.no_grad()
    def update_global_visibility(self, mask = None):

        # update_visibility = ((self.step+1) % self.num_train_data == 0
        #                      and self.step % self.refine_every != 0)
        cull_mask = None

        if self.global_visibility.shape[0] != self.xys.shape[0]:
            # gaussians have been refined, reset the global visibility
            self.global_visibility = torch.zeros_like(self.xys[..., 0]).to(self.xys.device)
            self.updated_count = 0

        self.global_visibility = self.global_visibility.to(self.xys.device)
        
        if self.updated_count // self.num_train_data > 1:

            # n_bef = self.num_points
            # do culling
            cull_mask = self.global_visibility < 2

            # self.means = Parameter(self.means[~cull_mask].detach())
            # self.scales = Parameter(self.scales[~cull_mask].detach())
            # self.quats = Parameter(self.quats[~cull_mask].detach())
            # self.features_dc = Parameter(self.features_dc[~cull_mask].detach())
            # self.features_rest = Parameter(self.features_rest[~cull_mask].detach())
            # self.opacities = Parameter(self.opacities[~cull_mask].detach())

            # CONSOLE.log(
            #     f"Culled {n_bef - self.num_points} gaussians out of any views"
            # )
            
        if mask is not None:
            self.global_visibility += self.is_inside_mask(self.xys, mask)

        else:
            self.global_visibility += self.is_inside(self.xys, self.last_size)
            
        self.updated_count += 1 

        
        return cull_mask

    def get_vis_loss(self, cameras:Cameras, out_image,out_vis, source_mask, target_masks, step, camera_closeness_indices= None,):
        
        d = self._get_downscale_factor()

        # randomly select 6 views from the target_masks
        # camera_closeness_indices = None



        if camera_closeness_indices is not None:
            
            if np.sum(camera_closeness_indices) == 0:
                # no matching views for current view
                return 0.0
            full_rand_d = np.random.random(camera_closeness_indices.shape)
            random_idx = np.lexsort((full_rand_d, -camera_closeness_indices))[:6]
            # random_idx = np.argsort(-camera_closeness_indices)[:6]
        else:
            random_idx = torch.randperm(len(target_masks))[:6]

        if d > 1:

            import torchvision.transforms.functional as TF
            newsieze = [out_image.shape[0] // d, out_image.shape[1] // d]
            source_mask = TF.resize(source_mask, newsieze, antialias=None)

            temp = [None] * len(target_masks)
            for i in random_idx:
                target_mask = target_masks[i]
                target_mask = TF.resize(target_mask, newsieze, antialias=None)
                temp[i] = target_mask
            target_masks = temp
        else:
            pass


        # get the source masks for gaussians, do not keep grad here
               
        


        # assert cameras.shape[0] == 1, "Only one camera at a time"

        # use the same crop box 
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                raise ValueError("No gaussians in the crop box")
                # return {"rgb": background.repeat(int(camera.height.item()), int(camera.width.item()), 1)}
        else:
            crop_ids = None


        visibility = torch.zeros_like(out_vis)
        

        with torch.no_grad():
            
            
            is_inside_source = self.is_inside(self.xys, source_mask.shape)
            in_x, in_y = self.xys[is_inside_source].long().T
            source_mask_gaussians = source_mask[in_y,in_x]                        

            for i in random_idx:
                camera = cameras[int(i)].to(self.device)
                camera.rescale_output_resolution(1 / d)
                # shift the camera to center of scene looking at center
                R = camera.camera_to_worlds[:3, :3]  # 3 x 3
                T = camera.camera_to_worlds[:3, 3:4]  # 3 x 1
                # flip the z and y axes to align with gsplat conventions
                R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
                R = R @ R_edit
                # analytic matrix inverse to get world2camera matrix
                R_inv = R.T
                T_inv = -R_inv @ T
                viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
                viewmat[:3, :3] = R_inv
                viewmat[:3, 3:4] = T_inv
                # calculate the FOV of the camera given fx and fy, width and height
                cx = camera.cx.item()
                cy = camera.cy.item()
                fovx = 2 * math.atan(camera.width / (2 * camera.fx))
                fovy = 2 * math.atan(camera.height / (2 * camera.fy))
                W, H = int(camera.width.item()), int(camera.height.item())

                projmat = projection_matrix(0.001, 1000, fovx, fovy, device=self.device)
                BLOCK_X, BLOCK_Y = 16, 16
                tile_bounds = (
                    int((W + BLOCK_X - 1) // BLOCK_X),
                    int((H + BLOCK_Y - 1) // BLOCK_Y),
                    1,
                )

            
                if crop_ids is not None:
                    opacities_crop = self.opacities[crop_ids]
                    means_crop = self.means[crop_ids]
                    features_dc_crop = self.features_dc[crop_ids]
                    features_rest_crop = self.features_rest[crop_ids]
                    scales_crop = self.scales[crop_ids]
                    quats_crop = self.quats[crop_ids]
                else:
                    opacities_crop = self.opacities
                    means_crop = self.means
                    features_dc_crop = self.features_dc
                    features_rest_crop = self.features_rest
                    scales_crop = self.scales
                    quats_crop = self.quats

                colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

                xys_target, depths, radii, conics, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
                    means_crop,
                    torch.exp(scales_crop),
                    1,
                    quats_crop / quats_crop.norm(dim=-1, keepdim=True),
                    viewmat.squeeze()[:3, :],
                    projmat.squeeze() @ viewmat.squeeze(),
                    camera.fx.item(),
                    camera.fy.item(),
                    cx,
                    cy,
                    H,
                    W,
                    tile_bounds,
                )  # type: ignore

                
                    # 
                    # for debugging

                    # plt.plot(depths)
                    # render mask in second view
                    # mask_rendered = rasterize_gaussians(
                    #     xys_target,
                    #     depths,
                    #     self.radii,
                    #     conics,
                    #     num_tiles_hit,
                    #     out_vis.repeat(1,3),
                    #     torch.sigmoid(opacities_crop),
                    #     H,
                    #     W,
                    #     background=torch.zeros(3, device=self.device),
        
                    # )[...,0:1]

                    # # visualize the mask_rendered
                    # import matplotlib.pyplot as plt

                    # # out mask
                    # fig,ax = plt.subplots(1,2)
                    # ax[0].imshow(mask_rendered.cpu().numpy())
                    # ax[1].imshow(target_masks[i].cpu().numpy())                 
                    # plt.show()

                target_mask = target_masks[i].to(self.device)
                # get the target masks for gaussians, do not keep grad here
                is_inside_target = self.is_inside(xys_target, target_mask.shape)

                in_x, in_y = xys_target[is_inside_target].long().T
                
                target_mask_gaussians = torch.ones((xys_target.shape[0]), dtype=target_mask.dtype, device=target_mask.device) -10

                reg_mode = "multi_mask"

                if reg_mode == "frustum":
                    target_mask_gaussians *= 0
                    target_mask_gaussians[is_inside_target] = 1

                    visibility[is_inside_source] += target_mask_gaussians[is_inside_source].unsqueeze(1)
                
                else:


                    if len(target_mask.shape) == 4:
                        target_mask_gaussians = torch.ones((xys_target.shape[0],*target_mask.shape[-2:]), dtype=target_mask.dtype, device=target_mask.device) -10
                            # with shape [number, ]
                        # deal with multilevel 
                        target_mask_gaussians[is_inside_target] = target_mask[in_y,in_x]

                        # get l2 
                        matches = (target_mask_gaussians[is_inside_source] - source_mask_gaussians)**2
                        matches = torch.mean(matches, dim = -1)

                        t = 0.02
                        # check the coarse level
                        # matches = torch.sum(matches,dim=-1) < t
                        matches = torch.max(matches, dim = -1)[0] < t
                        visibility[is_inside_source] += matches.unsqueeze(1)
                                                

                    else:
                        target_mask_gaussians[is_inside_target] = target_mask[in_y,in_x]

                        # target_mask_gaussians = target_mask[xys_target.long()]
                        
                        matches = torch.abs(target_mask_gaussians[is_inside_source] - source_mask_gaussians) < 1e-3

                        # deal with the background
                        matches[source_mask_gaussians <0] = False
                        
                        visibility[is_inside_source] += matches.unsqueeze(1)
        

                    # if step>2000:
                    

                    #     import matplotlib.pyplot as plt
                    #     # plot xys,
                    #     # with depths as size
                    #     # source mask as color
                    #     # rescale the depth from 1 to 10
                    #     plt.imshow(target_masks[i].cpu().numpy(), alpha=0.5)
                    #     plt.scatter(xys_target[is_inside_source,0].cpu().numpy(),
                    #                 xys_target[is_inside_source,1].cpu().numpy(), 
                    #                 s=self.depths[is_inside_source].detach().cpu().numpy(),
                    #                 c=source_mask_gaussians.cpu().numpy())
                    #     plt.colorbar()
                    #     plt.show()


                    #     plt.imshow(target_masks[i].cpu().numpy(), alpha=0.5)
                    #     plt.scatter(xys_target[is_inside_source,0].cpu().numpy(),
                    #                 xys_target[is_inside_source,1].cpu().numpy(), 
                    #                 s=self.depths[is_inside_source].detach().cpu().numpy(),
                    #                 c=matches.cpu().numpy())
                    #     plt.colorbar()
                    #     plt.show()

        # to loss
                    
        penality_size = self.config.mask_count_thresh - visibility
        penality_size = torch.clip(penality_size, min=0)
        # loss = - out_vis * (visibility + 1e-2)
        loss = out_vis * penality_size

        return torch.mean(loss)
    
    @torch.no_grad()
    def visualize_visibility(self,image_idx, raw_mask, source_mask, target_masks, cameras, camera_closeness_indices, train_dataset, vis_id = None, plot_scatter = True):

        # visualize any indices
        import matplotlib.pyplot as plt
        # num masks 
        raw_mask = raw_mask.to(self.device)
        num_masks = raw_mask.shape[-1]
        
        # pick one mask
        crop_ids = None
        d = self._get_downscale_factor()

        features_source_in_gaussians = torch.zeros((self.xys.shape[0], *source_mask.shape[-2:]), 
                                                    dtype=source_mask.dtype, device=source_mask.device)

        is_inside_source = self.is_inside(self.xys, source_mask.shape)
        in_x,in_y = self.xys[is_inside_source].long().T
        features_source_in_gaussians[is_inside_source] = source_mask[in_y,in_x]

        # pick 2 target views
        if camera_closeness_indices is not None:
            
            full_rand_d = np.random.random(camera_closeness_indices.shape)
            random_idx = np.lexsort((full_rand_d, -camera_closeness_indices))[:2]
            # random_idx = np.argsort(-camera_closeness_indices)[:6]
        else:
            random_idx = torch.randperm(len(target_masks))[:2]

        feature_distance = []
        target_xys = []

        for cam_id in random_idx:
            camera = cameras[int(cam_id)].to(self.device)
            camera.rescale_output_resolution(1 / d)
            # shift the camera to center of scene looking at center
            R = camera.camera_to_worlds[:3, :3]  # 3 x 3
            T = camera.camera_to_worlds[:3, 3:4]  # 3 x 1
            # flip the z and y axes to align with gsplat conventions
            R_edit = torch.diag(torch.tensor([1, -1, -1], device=self.device, dtype=R.dtype))
            R = R @ R_edit
            # analytic matrix inverse to get world2camera matrix
            R_inv = R.T
            T_inv = -R_inv @ T
            viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
            viewmat[:3, :3] = R_inv
            viewmat[:3, 3:4] = T_inv
            # calculate the FOV of the camera given fx and fy, width and height
            cx = camera.cx.item()
            cy = camera.cy.item()
            fovx = 2 * math.atan(camera.width / (2 * camera.fx))
            fovy = 2 * math.atan(camera.height / (2 * camera.fy))
            W, H = int(camera.width.item()), int(camera.height.item())

            projmat = projection_matrix(0.001, 1000, fovx, fovy, device=self.device)
            BLOCK_X, BLOCK_Y = 16, 16
            tile_bounds = (
                int((W + BLOCK_X - 1) // BLOCK_X),
                int((H + BLOCK_Y - 1) // BLOCK_Y),
                1,
            )


            if crop_ids is not None:
                opacities_crop = self.opacities[crop_ids]
                means_crop = self.means[crop_ids]
                features_dc_crop = self.features_dc[crop_ids]
                features_rest_crop = self.features_rest[crop_ids]
                scales_crop = self.scales[crop_ids]
                quats_crop = self.quats[crop_ids]
            else:
                opacities_crop = self.opacities
                means_crop = self.means
                features_dc_crop = self.features_dc
                features_rest_crop = self.features_rest
                scales_crop = self.scales
                quats_crop = self.quats

            colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

            xys_target, depths, radii, conics, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
                means_crop,
                torch.exp(scales_crop),
                1,
                quats_crop / quats_crop.norm(dim=-1, keepdim=True),
                viewmat.squeeze()[:3, :],
                projmat.squeeze() @ viewmat.squeeze(),
                camera.fx.item(),
                camera.fy.item(),
                cx,
                cy,
                H,
                W,
                tile_bounds,
            )

            target_mask = target_masks[cam_id].to(self.device)
            is_inside_target = self.is_inside(xys_target, target_mask.shape)

            in_x, in_y = xys_target[is_inside_target].long().T

            features_target_in_gaussians = torch.zeros_like(features_source_in_gaussians) -10
            features_target_in_gaussians[is_inside_target] = target_mask[in_y,in_x]

            matches = (features_target_in_gaussians-features_source_in_gaussians)**2
            matches = torch.mean(matches, dim=-1)
            
            feature_distance.append(matches)

            target_xys.append(xys_target)
        target_image_id_1 = random_idx[0]
        target_image_id_2 = random_idx[1]
        features_target1 = target_masks[target_image_id_1].to(self.device)
        features_target2 = target_masks[target_image_id_2].to(self.device)
        # concat the source and target features
        concatenated = torch.cat((source_mask.to(self.device), features_target1, features_target2), dim=1)
        # apply color map
        rgb_concatenated = apply_pca_colormap(concatenated)
        # separate the source and target features
        width = source_mask.shape[1]

        source_feature_rgb = rgb_concatenated[:, :width]
        target_feature_rgb_1 = rgb_concatenated[:, width:width*2]
        target_feature_rgb_2 = rgb_concatenated[:, width*2:]


        for i in range(num_masks):
            if vis_id is not None and i != vis_id:
                continue

            fig,ax = plt.subplots(2,3)
            source_image = train_dataset[image_idx]["image"]
            ax[0][0].imshow(source_image.numpy(force=True))
            ax[0][0].imshow(raw_mask[...,i].numpy(force=True), alpha = 0.5)

            is_in_mask = self.is_inside_mask(self.xys, raw_mask[...,i]).bool()
            ax[0][0].scatter(self.xys[is_in_mask, 0].numpy(force=True),
                        self.xys[is_in_mask, 1].numpy(force=True),
                        c = 'r')

        
            ax[1][0].imshow(source_feature_rgb.numpy(force=True)[...,0,:])
            vmin = 0.0
            vmax = 0.2

            

            # visalize target mask 1

            matches_1 = feature_distance[0][...,0]
            xys_target_1 = target_xys[0]
            # is_in_target = self.is_inside_mask(xys_target_1, raw_mask[...,i])
            vis_mask =  is_in_mask.bool()
            target_image_id_1 = random_idx[0]
            target_image_1 = train_dataset[target_image_id_1]["image"]

            ax[0][1].imshow(target_image_1)
            ax[0][1].scatter(xys_target_1[vis_mask, 0].numpy(force=True),
                        xys_target_1[vis_mask, 1].numpy(force=True),
                        c = matches_1[vis_mask].numpy(force=True),
                         vmin=vmin, vmax=vmax, cmap='viridis')
            ax[1][1].imshow(target_feature_rgb_1.numpy(force=True)[...,0,:])

            # visalize target mask 2
            matches_2 = feature_distance[1][...,0]
            xys_target_2 = target_xys[1]
            # is_in_target = self.is_inside_mask(xys_target_2, raw_mask[...,i])
            vis_mask = is_in_mask.bool()
            
            target_image_id_2 = random_idx[1]
            target_image_2 = train_dataset[target_image_id_2]["image"]


            ax[0][2].imshow(target_image_2.numpy())
            scatter2 = ax[0][2].scatter(xys_target_2[vis_mask, 0].numpy(force=True),
                        xys_target_2[vis_mask, 1].numpy(force=True),
                        c = matches_2[vis_mask].numpy(force=True),
                         vmin=vmin, vmax=vmax, cmap='viridis')
            fig.colorbar(scatter2, ax=ax[1:], orientation='vertical', label='Feature Matches')
            ax[1][2].imshow(target_feature_rgb_2.numpy(force=True)[...,0,:])
          
            plt.show()
            
        return rgb_concatenated


    def initialize_groups(self):
        # TODO fix shape

        if self.features["features_group"].shape[-1] == 3:
            self.features["features_group"]= torch.nn.Parameter(self.features_dc.clone().unsqueeze(1).repeat(1, self.config.group_feature_level,1), requires_grad=True)

    def get_tv_loss(self, outputs, unique_mask):


        unique_mask_val, inverse_indices, counts = unique_mask
        alpha = outputs["alpha"]
        row_alpha = (alpha[1:,:] + alpha[:-1,:])/2
        col_alpha = (alpha[:,1:] + alpha[:,:-1])/2
        # get the total variation of depth and exclude the boundery
        depth = outputs["depth"].squeeze(-1)
        diff_row = depth[1:,:] - depth[:-1,:]
        diff_col = depth[:,1:] - depth[:,:-1]

        inverse_indices = inverse_indices.view(depth.shape)
        counts = counts[inverse_indices]

        background = inverse_indices == 0
        invalid_patches = counts < 20
        invalid_mask = background | invalid_patches
        invalid_row = invalid_mask[1:,:] | invalid_mask[:-1,:]
        invalid_col = invalid_mask[:,1:] | invalid_mask[:,:-1]        

        mask_sim_row = (inverse_indices[1:,:] == inverse_indices[:-1,:]) & (~invalid_row)
        mask_sim_col = (inverse_indices[:,1:] == inverse_indices[:,:-1]) & (~invalid_col)

        # erosion 
        mask_sim_row = binary_erosion(mask_sim_row,5)
        mask_sim_col = binary_erosion(mask_sim_col,5)

        valid_tv = torch.mean(torch.abs(diff_row[mask_sim_row])) + torch.mean(torch.abs(diff_col[mask_sim_col]))

        valid_diff_row = torch.abs(diff_row[mask_sim_row])
        valid_diff_col = torch.abs(diff_col[mask_sim_col])

        # t = 0.1

        # valid_diff_row[valid_diff_row > t] *=20
        # valid_diff_col[valid_diff_col > t] *=20

        valid_tv = torch.mean(valid_diff_row) + torch.mean(valid_diff_col)

        total_tv = torch.mean(torch.abs(diff_row)) + torch.mean(torch.abs(diff_col))

        return valid_tv
        # cluster_label = self.features["features_cluster"]

        # # calc mean of each cluster
        # labels_one_hot = F.one_hot(cluster_label, num_classes=cluster_label.max()+1)
        # sums = torch.matmul(labels_one_hot.t(), self.means)
        # counts = labels_one_hot.sum(dim=0).unsqueeze(-1)
        # cluster_means = sums / counts
        
        # loss = 0.0

        # for i in len(unique_mask_val):
        #     mask_pos = inverse_indices== i
        #     ind_in_mask = self.is_inside_mask(self.xys, mask_pos)
        #     clusters = self.features["features_cluster"][ind_in_mask].unique()
        #     # selected cluster
            


    def get_group_loss_old(self, outputs, source_mask, binary_mask=None):

        t1 = time.time()
        # get the group loss
        source_mask = source_mask.to(self.device)

        orignal_shape = source_mask.shape

        if binary_mask is not None:
            binary_mask = binary_mask.to(self.device).squeeze(-1).bool()

        else:
            binary_mask = torch.ones_like(source_mask[...,0]).to(self.device).bool()

        
        binary_mask = torch.flatten(binary_mask, start_dim=0, end_dim=1)

        # flatten the 0,1 dimension
        source_mask = torch.flatten(source_mask, start_dim=0, end_dim=1)
        source_feature = None

        # get_unique mask 
        unique_mask_val, inverse_indices, counts = torch.unique(dim=0, return_inverse=True, return_counts=True, input=source_mask)

        #ignore all zero mask_val
        all_zero_ind = torch.all(unique_mask_val == 0, dim=1)

        all_zero_ind = torch.where(all_zero_ind)[0].item()

        invalid_mask = torch.zeros((len(unique_mask_val), len(unique_mask_val))).to(self.device).bool()
        invalid_mask[all_zero_ind,:] = 1
        invalid_mask[:,all_zero_ind] = 1
        
        # get match label
        if len(unique_mask_val.shape) > 1:
            # multi-binary mask
            num_matches = torch.mm(unique_mask_val, unique_mask_val.T)
        else:

            # always use 1 for single-non-overlapping mask
            num_matches = torch.zeros((len(unique_mask_val), len(unique_mask_val))).to(self.device)
            # 
            # invalid_mask = torch.zeros_like(num_matches).to(self.device).bool()
            ind_background = torch.where(unique_mask_val < 0)[0]

            invalid_mask[ind_background,:] = 1
            invalid_mask[:,ind_background] = 1

            # add random aggregation
            ind_0, ind_1 = torch.where(unique_mask_val == 4)[0], torch.where(unique_mask_val == 5)[0]
            num_matches[ind_0,ind_1] = 1
            num_matches[ind_1,ind_0] = 1


        t2 = time.time()

        contrastive_loss = 0.0

        mask_groups = []
        # get mean of each mask combination

        


        unique_means = []
        unique_stds = []
        unique_counts = []

        for level in tqdm.trange(self.config.group_feature_level):
            
            level_means = []
            level_stds = []
            level_counts = []

            # scale back to -1, 1
            features_current_level = outputs[f"features_group_{level}"] * 2 - 1 
            # flatten for the index retrieve
            features_current_level = torch.flatten(features_current_level, start_dim=0, end_dim=1)

            for i in tqdm.trange(len(unique_mask_val)):
                
                # mask_group = outputs[f"features_group_{level}"][inverse_indices == i]
                mask_group = features_current_level[(inverse_indices == i) & binary_mask]
                # mask_groups_level.append(mask_group)
                if len(mask_group) == 1:
                    
                    level_means.append(mask_group.squeeze(0))
                    level_stds.append(torch.zeros_like(mask_group.squeeze(0)))
                    level_counts.append(torch.sum(inverse_indices == i))

                elif len(mask_group) == 0:

                    level_means.append(torch.zeros(3).to(self.device))
                    level_stds.append(torch.zeros(3).to(self.device))
                    level_counts.append(torch.sum(inverse_indices == i))

                else:
                    
                    level_means.append(torch.mean(mask_group, dim=0))
                    level_stds.append(torch.std(mask_group, dim=0))
                    level_counts.append(torch.sum(inverse_indices == i))

            unique_means.append(torch.stack(level_means))
            unique_stds.append(torch.stack(level_stds))
            unique_counts.append(torch.stack(level_counts))
                
            # mask_groups.append(mask_groups_level)
        t3 = time.time()

        dissim_margin = 1.0
        sim_loss = 0.0
        dissim_loss = 0.0

        hierarchy_loss = 0.0

        reg_loss = 0.0

        last_level_feature = None

        for level in range(self.config.group_feature_level):

            sim_indices = torch.triu( num_matches > level, diagonal=1) 

            dissim_indices = torch.triu(num_matches <= level, diagonal=1)
           
            if invalid_mask is not None:
                sim_indices = torch.logical_and(sim_indices, torch.logical_not(invalid_mask))
                dissim_indices = torch.logical_and(dissim_indices, torch.logical_not(invalid_mask))

            # group mean distance
            mean_features = unique_means[level]
            std_features = unique_stds[level]

            # check nan values 
            # CONSOLE.log(f"nan in mean features: {torch.sum(torch.isnan(mean_features))}, nan in std features: {torch.sum(torch.isnan(std_features))}")

            count_features = unique_counts[level]
            mean_minus_mean = torch.mean((mean_features[:,None] - mean_features)**2, dim=2)
            mean_minus_mean = torch.triu(mean_minus_mean, diagonal=1)


            # NaNb = count_features.unsqueeze(1) * count_features.unsqueeze(0)
            # NaSa_plus_NbSb = (std_features*count_features).unsqueeze(1) + (std_features*count_features).unsqueeze(0)
            # Na_plus_Nb = count_features.unsqueeze(1) + count_features.unsqueeze(0)

            # combined_variance = (NaSa_plus_NbSb + NaNb/Na_plus_Nb * mean_minus_mean) / Na_plus_Nb
            group_distance = mean_minus_mean

            if level > 0:

                if last_level_feature is not None:
                
                    supervised_unique_indices = torch.unique(torch.where(dissim_indices)[0])

                    supervised_indices = torch.zeros_like(last_level_feature).bool()
                    supervised_indices[supervised_unique_indices] = 1


                    level_difference = (last_level_feature-mean_features)**2
                    hierarchy_loss += torch.mean(level_difference[~supervised_indices]) +\
                        0.01 *torch.mean(level_difference[supervised_indices])
                    last_level_feature = mean_features.detach()

                else:

                    last_level_feature = mean_features.detach()

                # hierarchy_loss += torch.mean((unique_means[level - 1].detach() - unique_means[level])**2)


            sim_threshold = (5-level)/5
            # sim_loss += combined_variance[sim_indices].mean()
            if torch.sum(sim_indices) > 0:
                sim_loss += torch.nanmean(group_distance[sim_indices]) 
            sim_loss += torch.nansum(std_features * count_features.unsqueeze(1) / (torch.sum(count_features) + 1e-6))

            # use a soft margin for dissimilarity
            # sim_loss = torch.log(1 + torch.exp(sim_loss - level*0.1))

            if torch.sum(dissim_indices) > 0:
                # level_margin = dissim_margin/ (level/2.0+1)
                level_margin = dissim_margin
                dissim_loss += torch.nanmean(F.relu(level_margin - group_distance[dissim_indices])) * (1.8-sim_threshold)

            # reg_loss = torch.min(0.7, 1 - mean_features.mean(dim=0)

        # get the contrastive loss
        # sim loss
        
        t4 = time.time()

        CONSOLE.log("time 1", t2-t1, "time 2", t3-t2, "time 3", t4-t3)
       
        CONSOLE.log(f"sim_loss: {sim_loss}, dissim_loss: {dissim_loss}, hierarchy_loss: {hierarchy_loss}")
        return {
            "sim_loss": sim_loss*sim_threshold,
            "dissim_loss": dissim_loss,
            # "hierarchy_loss": hierarchy_loss
        }


    def get_group_loss(self, outputs, unique_mask, binary_mask=None):
        
        t1 = time.time()
        # get the group loss
        # source_mask = source_mask.to(self.device)

        # orignal_shape = source_mask.shape

        # if binary_mask is not None:
        #     binary_mask = binary_mask.to(self.device).squeeze(-1).bool()

        # else:
        #     binary_mask = torch.ones_like(source_mask[...,0]).to(self.device).bool()

        
        # binary_mask = torch.flatten(binary_mask, start_dim=0, end_dim=1)

        # flatten the 0,1 dimension
        # source_mask = torch.flatten(source_mask, start_dim=0, end_dim=1)
        # source_feature = None

        # get_unique mask 
        # unique_mask_val, inverse_indices, counts = torch.unique(dim=0, return_inverse=True, return_counts=True, input=source_mask)
        unique_mask_val, inverse_indices, counts = unique_mask
        # to device
        unique_mask_val = unique_mask_val.to(self.device)
        inverse_indices = inverse_indices.to(self.device)
        counts = counts.to(self.device)

        #ignore all zero mask_val
        all_zero_ind = torch.all(unique_mask_val == 0, dim=1)
        ind_small_patches = counts < 20

        invalid_ind = all_zero_ind | ind_small_patches

        invalid_ind = torch.where(invalid_ind)[0]

        # all_zero_ind = torch.where(all_zero_ind)[0].item()
    
        # inverse_invalid_ind = torch.any(inverse_indices.unsqueeze(1) == invalid_ind.unsqueeze(0), dim =1)

        # valid_inverse_indices = inverse_indices[~inverse_invalid_ind]

        # valid_indices = torch.where(inverse_invalid_ind == 0)[0]

        invalid_mask = torch.zeros((len(unique_mask_val), len(unique_mask_val))).to(self.device).bool()
        invalid_mask[invalid_ind,:] = 1
        invalid_mask[:,invalid_ind] = 1

        # invalid_mask[ind_small_patches,:] = 1
        # invalid_mask[:,ind_small_patches] = 1

        # get match label
        if len(unique_mask_val.shape) > 1:
            # multi-binary mask
            num_matches = torch.mm(unique_mask_val, unique_mask_val.T)
        else:

            # always use 1 for single-non-overlapping mask
            num_matches = torch.zeros((len(unique_mask_val), len(unique_mask_val))).to(self.device)
            # 
            # invalid_mask = torch.zeros_like(num_matches).to(self.device).bool()
            ind_background = torch.where(unique_mask_val < 0)[0]

            invalid_mask[ind_background,:] = 1
            invalid_mask[:,ind_background] = 1

            # add random aggregation
            ind_0, ind_1 = torch.where(unique_mask_val == 4)[0], torch.where(unique_mask_val == 5)[0]
            num_matches[ind_0,ind_1] = 1
            num_matches[ind_1,ind_0] = 1


        


        t2 = time.time()


        feature_levels = [outputs[f"features_group_{i}"].flatten(0,1) for i in range(self.config.group_feature_level)]

        feature_levels = torch.stack(feature_levels, dim=1)

        num_features = len(unique_mask_val)
        num_level_features = self.config.group_feature_level
        dim_features = self.config.group_feature_size

        # feature_means = torch.zeros((num_features, num_level_features, dim_features), device=self.device)

        # feature_mse = torch.zeros((num_features, num_level_features), device=self.device)

        # feature_counts = torch.zeros((len(unique_mask_val)), device=self.device)

        # for i in range(len(unique_mask_val)):

        #     valid_ind = (inverse_indices == i) & binary_mask

        #     feature_count = torch.sum(valid_ind)

        #     feature_counts[i] = feature_count

        #     if feature_count == 0:
        #         continue
        #     elif feature_count == 1:
        #         feature_means[i,...] = feature_levels[valid_ind]
        #     else:
        #         selected_features = feature_levels[valid_ind]

        #         feature_means[i,...] = torch.mean(selected_features, dim=0)
        #         feature_mse[i,:] = torch.mean((selected_features-feature_means[i,...])**2, dim=[0,2])


        index_expanded = inverse_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, num_level_features, dim_features)
        feature_sums = torch.zeros(num_features, num_level_features, dim_features,device=self.device).scatter_add_(0, index_expanded, feature_levels)
        feature_means = feature_sums / counts.unsqueeze(-1).unsqueeze(-1)

        dist = torch.abs(feature_levels - feature_means[inverse_indices])
        feature_dist = torch.zeros(num_features, num_level_features,device=self.device).scatter_add(0, 
                                                                            inverse_indices.unsqueeze(-1).expand(-1, num_level_features), 
                                                                            dist.mean(dim=2))/counts.unsqueeze(-1)




        t3 = time.time()
        dissim_margin = 1.0
        sim_loss = 0.0
        dissim_loss = 0.0

        # create a one 
        def get_weight(size, mean, std):
            weight = torch.arange(size, device=self.device)
            return torch.exp(-0.5 * (weight - mean) ** 2 / std ** 2)

        # mean_minus_mean = torch.mean( (feature_means[:,None] - feature_means)**2, dim =3)
        mean_minus_mean = torch.mean(torch.abs(feature_means[:,None] - feature_means), dim =3) 

        max_overlapping = torch.max(num_matches)
        scaling_factor = max_overlapping / self.config.group_feature_level

        for level in range(self.config.group_feature_level):

            # scaled_level = level * scaling_factor
            # last_scaled_level = (level-1) * scaling_factor

            scaled_level = level
            last_scaled_level = level-1

            sim_indices = torch.triu( num_matches > scaled_level, diagonal=1)
            dissim_indices = torch.triu((num_matches == scaled_level) , diagonal=1) # & (num_matches > last_scaled_level)
           
            if invalid_mask is not None:
                sim_indices = torch.logical_and(sim_indices, torch.logical_not(invalid_mask))
                dissim_indices = torch.logical_and(dissim_indices, torch.logical_not(invalid_mask))

            # group mean distance 
            mean_features_level = feature_means[:,level]
            gaussian_weights = get_weight(num_level_features, level, 0.8)         

            # gaussian_weights = torch.zeros(num_level_features, device=self.device)
            # gaussian_weights[level] = 1

            weighted_mtm= mean_minus_mean * gaussian_weights
            weighted_mtm = torch.sum(weighted_mtm, dim=2)/ torch.sum(gaussian_weights)
            std_features_level = feature_dist[:,level]


            # check nan values 
            # CONSOLE.log(f"nan in mean features: {torch.sum(torch.isnan(mean_features))}, nan in std features: {torch.sum(torch.isnan(std_features))}")

            count_features_level = counts

            # weighted_mtm = torch.mean((mean_features_level[:,None] - mean_features_level)**2, dim=2)


            group_distance = weighted_mtm
            
            # sim_threshold = (5-level)/5

            # weight = 1/ torch.log(count_features_level) 
            
            # weight = weight[:,None] * weight

            # weight = torch.triu(weight, diagonal=1)
            # normalize 

            # weight = torch.ones_like(group_distance)
            # group_distance = group_distance


            if torch.sum(sim_indices) > 0:
                sim_loss += torch.nanmean(group_distance[sim_indices])
                # sim_loss += torch.nansum(group_distance[sim_indices]*weight[sim_indices]/(torch.sum(weight[sim_indices])+1e-6))
            # sim_loss += torch.nansum(std_features_level * count_features_level.unsqueeze(1) / (torch.sum(count_features_level) + 1e-6))

            sim_loss += torch.nanmean(std_features_level[~invalid_ind]) * 2
            # sim_loss += torch.nansum(std_features_level* weight/(torch.sum(weight)+1e-6)) 

            if torch.sum(dissim_indices)>0:
                # level_margin = dissim_margin/ (level/2.0+1)
                level_margin = dissim_margin
                dissim_loss += torch.nanmean(F.relu(level_margin - group_distance[dissim_indices])) #* (level**2/3.0+1)
                # dissim_loss += torch.nansum(F.relu(level_margin - group_distance[dissim_indices])*weight[dissim_indices]/(torch.sum(weight[dissim_indices])+1e-6)) #* (level**2/2.0+1.5)

        sim_threshold = 1.00
        t4 = time.time()

        # CONSOLE.log(f"sim_loss: {sim_loss}, dissim_loss: {dissim_loss}, sim_threshold: {sim_threshold}")
        # CONSOLE.log("time 1", t2-t1, "time 2", t3-t2, "time 3", t4-t3)
       
        return {
            
            "sim_loss": sim_loss,
            "dissim_loss": dissim_loss
        }

        pass


    def pointwise_group_loss(self, outputs, source_mask,):

        feature_levels = [outputs[f"features_group_{i}"].flatten(0,1) for i in range(self.config.group_feature_level)]

        feature_levels = torch.stack(feature_levels, dim=1)

        # flatten the 0,1 dimension

        source_mask = source_mask.flatten(0,1)
        # feature_levels = feature_levels.flatten(0,1)
        # chunck size 

        chuck_size = 2048

        sim_loss = 0.0
        dissim_los = 0.0
        dissim_margin = 1.0

        # randomly select chuck size indices
        random_indices = torch.randperm(feature_levels.shape[0])
        random_indices = random_indices[:chuck_size]
        
        source_mask = source_mask[random_indices]
        feature_levels = feature_levels[random_indices]

        num_matches = torch.mm(source_mask, source_mask.t()) # N x M_i

        ptp_mse = torch.mean((feature_levels[:,None] - feature_levels)**2, dim =-1)

        # background
        # invalid_indices = torch.zeros_like(ptp_mse).bool()
        # invalid_indices[0,:] = True
        # invalid_indices[:,0] = True

        def get_weight(size, mean, std):
            weight = torch.arange(size, device=self.device)
            return torch.exp(-0.5 * (weight - mean) ** 2 / std ** 2)

        for i in range(self.config.group_feature_level):

            sim_indices = torch.triu( num_matches > i , diagonal=1)
            dissim_indices = torch.triu((num_matches == i), diagonal=1)

            # if invalid_indices is not None:
            #     sim_indices = torch.logical_and(sim_indices, torch.logical_not(invalid_indices))
            #     dissim_indices = torch.logical_and(dissim_indices, torch.logical_not(invalid_indices))

            weight = get_weight(ptp_mse.shape[-1], i, 0.5)

            sim_loss += torch.nanmean(ptp_mse[sim_indices])
            dissim_los += torch.nanmean(F.relu(dissim_margin - ptp_mse[dissim_indices][i]))

        return {
            "sim_loss": sim_loss,
            "dissim_loss": dissim_los
        }


    def get_group_render_loss(self, outputs, encoded_mask, learn_encoder=False):

        render_loss = 0.0

        for i in range(self.config.group_feature_level):
            rendered_feature_map = outputs[f"features_group_{i}"]*2-1

            if learn_encoder:
                rendered_feature_map = rendered_feature_map.detach()
            # get mse,  as anchor and update the anchor
            render_loss += torch.mean((rendered_feature_map - encoded_mask[...,i,:]) ** 2)

        return {
            "render_loss": render_loss
        }


    def is_inside(self, xys, size):
        return (xys[..., 0] > 0) & (xys[..., 0] < size[1]) & (xys[..., 1] > 0) & (xys[..., 1] < size[0])


    
    def is_inside_mask(self, xys, mask):

        inside = torch.zeros_like(xys[..., 0])

        inside_view = self.is_inside(xys, mask.shape)

        inside[inside_view] = mask[xys[inside_view, 1].long(), xys[inside_view, 0].long()]

        return inside
    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        d = self._get_downscale_factor()
        if d > 1:
            newsize = [image.shape[0] // d, image.shape[1] // d]

            # torchvision can be slow to import, so we do it lazily.
            import torchvision.transforms.functional as TF

            gt_img = TF.resize(image.permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        else:
            gt_img = image
        return gt_img.to(self.device)

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.get_gt_img(batch["image"])
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points

        return metrics_dict


    def get_depth_loss(self, outputs, batch):

        midas_depth = batch["depth_image"].to(self.device).squeeze(-1)

        depth_patch_size = 16
        midas_patches = midas_depth.unfold(0, depth_patch_size, depth_patch_size).unfold(1,depth_patch_size, depth_patch_size)
        midas_patches = midas_patches.reshape(-1, depth_patch_size, depth_patch_size)
        midas_patches = midas_patches.flatten(1,2)

        rendered_depth = outputs["depth"].squeeze(-1)
        rendered_patches = rendered_depth.unfold(0, depth_patch_size, depth_patch_size).unfold(1, depth_patch_size,depth_patch_size)
        rendered_patches = rendered_patches.reshape(-1, depth_patch_size, depth_patch_size)
        rendered_patches = rendered_patches.flatten(1,2)


        # split to pactches 


    
        # depth_loss = torch.min(
        #             (1 - pearson_corrcoef( - midas_depth, rendered_depth)),
        #             (1 - pearson_corrcoef(1 / (midas_depth + 200.), rendered_depth))
        # )

        depth_loss = 1 - pearson_corrcoef(midas_patches, rendered_patches).mean()
        
        return depth_loss
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.get_gt_img(batch["image"])
        pred_img = outputs["rgb"]

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch and self.config.use_binary_mask_for_training:
            # batch["mask"] : [H, W, 1]
            assert batch["mask"].shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            mask = batch["mask"].to(self.device)
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(-1)

            if len(mask.shape) ==2:
                mask = mask.unsqueeze(-1)

            gt_img = gt_img * mask
            pred_img = pred_img * mask

        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)

        if self.config.use_depth and "pearson_depth_loss" in metrics_dict:
            
            depth_loss = metrics_dict["pearson_depth_loss"] *0.5
            # depth_loss = depth_loss.mean()
            return {
                "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss,
                "depth_loss": depth_loss,
                "scale_reg": scale_reg,
            }

        return {
            "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss,
            "scale_reg": scale_reg,
        }

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera.to(self.device))

        # if "features_group_0" in outs:
        #     temp = []
        #     for i in range(self.config.group_feature_level):
        #         temp.append(outs[f"features_group_{i}"])
        #         outs[f"features_group_{i}"] =  F.normalize(torch.concatenate(temp, dim=-1), p=2, dim=-1)
        #         CONSOLE.log(f"features_group_{i}: {outs[f'features_group_{i}'].shape}")

        return outs  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = self.get_gt_img(batch["image"])
        d = self._get_downscale_factor()

        mask = None

        if "mask" in batch:            
            # batch["mask"] : [H, W, 1]
            mask = batch["mask"].to(self.device)
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(-1)
            if len(mask.shape) == 2:
                mask = mask[..., None]

        if d > 1:
            # torchvision can be slow to import, so we do it lazily.
            import torchvision.transforms.functional as TF

            newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]
            predicted_rgb = TF.resize(outputs["rgb"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
            if mask is not None:
                mask = TF.resize(mask.permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)

        else:
            predicted_rgb = outputs["rgb"]


        # save raw rgb
        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        metrics_dict = {}

        if mask is not None:

            
            # evaluate only valid pixels


            gt_rgb_masked = gt_rgb * mask
            predicted_rgb_masked = predicted_rgb * mask

            if self.config.use_binary_mask_for_training:
                # save the masked rgb instead
                combined_rgb = torch.cat([combined_rgb, predicted_rgb_masked], dim=1)

            if self.config.use_binary_mask_for_training:
                # save the masked rgb instead
                combined_rgb = torch.cat([gt_rgb_masked, predicted_rgb_masked], dim=1)

            gt_rgb_masked = torch.moveaxis(gt_rgb_masked, -1, 0)[None, ...]
            predicted_rgb_masked = torch.moveaxis(predicted_rgb_masked, -1, 0)[None, ...]

            bin_mask = mask.squeeze(-1)

            psnr_masked = self.psnr(gt_rgb_masked[:,:,bin_mask], predicted_rgb_masked[:,:,bin_mask])
            ssim_masked = self.ssim(gt_rgb_masked, predicted_rgb_masked)
            lpips_masked = self.lpips(gt_rgb_masked, predicted_rgb_masked)

            metrics_dict = {
                "psnr_masked": float(psnr_masked.item()),
                "ssim_masked": float(ssim_masked),
                "lpips_masked": float(lpips_masked),
            }


        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict.update({"psnr": float(psnr.item()), "ssim": float(ssim)})  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb}

        return metrics_dict, images_dict
