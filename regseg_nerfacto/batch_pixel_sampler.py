from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Type, Union

import torch
from jaxtyping import Int
from torch import Tensor
from nerfstudio.data.pixel_samplers import *
# (
#     PairPixelSampler,
#     PairPixelSamplerConfig
# )
from rich.progress import Console

CONSOLE = Console(width=120)

@dataclass
class BatchPairPixelSamplerConfig(PairPixelSamplerConfig):
    """Config dataclass for BatchPixelSampler."""

    _target: Type = field(default_factory=lambda: BatchPairPixelSampler)

class BatchPairPixelSampler(PairPixelSampler):  # pylint: disable=too-few-public-methods
    """Samples pair of pixels from 'image_batch's. Samples pairs of pixels from
        from the images randomly within a 'radius' distance apart. Useful for pair-based losses.

    Args:
        config: the PairPixelSamplerConfig used to instantiate class
    """
    config: BatchPairPixelSamplerConfig


    # overrides base method
    def sample_method(  # pylint: disable=no-self-use
        self,
        batch_size: Optional[int],
        num_images: int,
        image_height: int,
        image_width: int,
        mask: Optional[Tensor] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> Int[Tensor, "batch_size 3"]:
        rays_to_sample = self.rays_to_sample
        if batch_size is not None:
            assert (
                int(batch_size) % (2*num_images) == 0
            ), f"PairPixelSampler can only return batch sizes in multiples of 2x{num_images} (got {batch_size})"
            rays_to_sample = batch_size // 2 


        rays_per_image = batch_size // num_images //2 
        
        assert rays_per_image*2*num_images == batch_size 

        if isinstance(mask, Tensor):
            m = erode_mask(mask.permute(0, 3, 1, 2).float(), pixel_radius=self.radius)

            indices = []
            for im_ind in range(m.shape[0]):
                nonzero_indices = torch.nonzero(m[im_ind, 0], as_tuple=False).to(device)
                chosen_indices = random.sample(range(len(nonzero_indices)), k=rays_per_image)
                im_indices = nonzero_indices[chosen_indices]
                ns = torch.ones_like(im_indices[:,:1]) * im_ind
                im_indices = torch.hstack((ns, im_indices))
                indices.append(im_indices)
            indices = torch.concat(indices)

        else:
            ns = torch.arange(0, num_images, device=device).repeat_interleave(rays_per_image).unsqueeze(1)
            s = (rays_to_sample, 1)
            # ns = torch.randint(0, num_images, s, dtype=torch.long, device=device)
            hs = torch.randint(self.radius, image_height - self.radius, s, dtype=torch.long, device=device)
            ws = torch.randint(self.radius, image_width - self.radius, s, dtype=torch.long, device=device)
            indices = torch.concat((ns, hs, ws), dim=1)
        

        pair_indices = torch.hstack(
            (
                torch.zeros(rays_to_sample, 1, device=device, dtype=torch.long),
                torch.randint(-self.radius, self.radius, (rays_to_sample, 2), device=device, dtype=torch.long),
            )
        )
        pair_indices += indices
        indices = torch.hstack((indices, pair_indices)).view(rays_to_sample * 2, 3)
        return indices
