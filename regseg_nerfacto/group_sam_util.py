from nerfstudio.field_components.field_heads import *
from nerfstudio.model_components.renderers import *

from enum import Enum
from typing import Callable, Optional, Union

import torch
from jaxtyping import Float, Shaped
from torch import Tensor, nn

from nerfstudio.field_components.base_field_component import FieldComponent

import torch.nn.functional as F

# add grouping head to the FieldHeadNames
# class FieldHeadNames(Enum):
#     """Possible field outputs"""

#     RGB = "rgb"
#     SH = "sh"
#     DENSITY = "density"
#     NORMALS = "normals"
#     PRED_NORMALS = "pred_normals"
#     UNCERTAINTY = "uncertainty"
#     BACKGROUND_RGB = "background_rgb"
#     TRANSIENT_RGB = "transient_rgb"
#     TRANSIENT_DENSITY = "transient_density"
#     SEMANTICS = "semantics"
#     SDF = "sdf"
#     ALPHA = "alpha"
#     GRADIENT = "gradient"
#     GROUPING = "grouping"

class GroupFieldHead(FieldHead):

    def __init__(self, feature_level, feature_size, in_dim = None):
        
        self.feature_level = feature_level
        self.feature_size = feature_size
        super().__init__(in_dim=in_dim, out_dim=feature_size*feature_level, field_head_name=FieldHeadNames.GROUPING)

    def _construct_net(self):
        
        net = [nn.Linear(self.in_dim, self.feature_size)]

        for i in range(self.feature_level-1):
            layer = nn.Linear(self.feature_size + self.in_dim, self.feature_size)
            nn.init.eye_(layer.weight)
            nn.init.zeros_(layer.bias)
            net.append(layer)
            
        self.net = nn.ModuleList(net)
        

    def forward(self, in_tensor: Float[Tensor, "*bs in_dim"]) -> Float[Tensor, "*bs out_dim"]:
        
        if not self.net:
            raise SystemError("in_dim not set. Must be provided to constructor, or set_in_dim() should be called.")
        
        out_tensor = self.net[0](in_tensor.detach())
        
        if self.activation:
            out_tensor = self.activation(out_tensor)

        out_tensors = [out_tensor,]

        for i in range(self.feature_level-1):
            out_tensor = self.net[i+1](torch.cat([in_tensor, out_tensor], dim=-1))
            if self.activation:
                out_tensor = self.activation(out_tensor)
            out_tensors.append(out_tensor)

        out_tensor = torch.stack(out_tensors, dim=-2)
        out_tensor = F.normalize(out_tensor, p=2, dim=-1)

        return out_tensor
        

class GroupRenderer(nn.Module):
    """Calculate semantics along the ray."""

    @classmethod
    def forward(
        cls,
        group: Float[Tensor, "*bs num_samples level_features size_features"],
        weights: Float[Tensor, "*bs num_samples 1"],
        ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> Float[Tensor, "*bs num_classes"]:
        """Calculate semantics along the ray."""
        
        reshaped = False

        if len(group.shape) == 4:
            *_, level_features, size_features = group.shape

            group = group.view(*_, level_features*size_features)

            reshaped = True

        weights = weights.detach()
        if ray_indices is not None and num_rays is not None:

            # Necessary for packed samples from volumetric ray sampler
            res = nerfacc.accumulate_along_rays(
                weights[..., 0], values=group, ray_indices=ray_indices, n_rays=num_rays
            )
        

        else:
            res = torch.sum(weights * group, dim=-2)

        if reshaped:
            res = res.view(*res.shape[:-1], level_features, size_features)
                
            # normalize
            res = F.normalize(res, p=2, dim=-1)

        return res