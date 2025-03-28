
from typing import Dict

import numpy as np

import torch
from torch.nn import functional as F

import pickle 

from pathlib import Path

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs, Semantics
from nerfstudio.data.datasets.base_dataset import InputDataset

from nerfstudio.data.utils.data_utils import get_depth_image_from_path
from nerfstudio.utils.misc import torch_compile
from nerfstudio.utils.rich_utils import CONSOLE

from nerfstudio.model_components import losses
from typing import Union
from PIL import Image
import torch
from rich.progress import track
from pathlib import Path
import json
from typing import Literal

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

    


class LLFFDataset(InputDataset):


    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        # if there are no depth images than we want to generate them all with zoe depth

        if len(dataparser_outputs.image_filenames) > 0 and (
            "depth_filenames" not in dataparser_outputs.metadata.keys()
            or dataparser_outputs.metadata["depth_filenames"] is None
        ):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            CONSOLE.print("[bold yellow] No depth data found! Generating pseudodepth...")
            losses.FORCE_PSEUDODEPTH_LOSS = True
            CONSOLE.print("[bold red] Using psueodepth: forcing depth loss to be ranking loss.")
            cache = dataparser_outputs.image_filenames[0].parent / "depths.npy"
            # Note: this should probably be saved to disk as images, and then loaded with the dataparser.
            #  That will allow multi-gpu training.
            if cache.exists():
                CONSOLE.print("[bold yellow] Loading pseudodata depth from cache!")
                # load all the depths
                self.depths = np.load(cache)
                self.depths = torch.from_numpy(self.depths).to(device)
                if len(self.depths) != len(dataparser_outputs.image_filenames):
                    self.depths = None

            else:
                depth_tensors = []
                transforms = self._find_transform(dataparser_outputs.image_filenames[0])
                data = dataparser_outputs.image_filenames[0].parent
                if transforms is not None:
                    meta = json.load(open(transforms, "r"))
                    frames = meta["frames"]
                    filenames = [data / frames[j]["file_path"].split("/")[-1] for j in range(len(frames))]
                else:
                    meta = None
                    frames = None
                    filenames = dataparser_outputs.image_filenames

                repo = "isl-org/ZoeDepth"
                self.zoe = torch_compile(torch.hub.load(repo, "ZoeD_NK", pretrained=True).to(device))

                for i in track(range(len(filenames)), description="Generating depth images"):
                    image_filename = filenames[i]
                    pil_image = Image.open(image_filename)
                    image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
                    if len(image.shape) == 2:
                        image = image[:, :, None].repeat(3, axis=2)
                    image = torch.from_numpy(image.astype("float32") / 255.0)

                    with torch.no_grad():
                        image = torch.permute(image, (2, 0, 1)).unsqueeze(0).to(device)
                        if image.shape[1] == 4:
                            image = image[:, :3, :, :]
                        depth_tensor = self.zoe.infer(image).squeeze().unsqueeze(-1)

                    depth_tensors.append(depth_tensor)

                self.depths = torch.stack(depth_tensors)
                np.save(cache, self.depths.cpu().numpy())
            dataparser_outputs.metadata["depth_filenames"] = None
            dataparser_outputs.metadata["depth_unit_scale_factor"] = 1.0
            self.metadata["depth_filenames"] = None
            self.metadata["depth_unit_scale_factor"] = 1.0

        self.depth_filenames = self.metadata["depth_filenames"]
        # self.depth_unit_scale_factor = self.metadata["depth_unit_scale_factor"]
        # self.depths = None
        

    def get_metadata(self, data: Dict) -> Dict:
        if self.depth_filenames is None:
            if self.depths is None:
                return {}
            return {"depth_image": self.depths[data["image_idx"]]}

        filepath = self.depth_filenames[data["image_idx"]]
        height = int(self._dataparser_outputs.cameras.height[data["image_idx"]])
        width = int(self._dataparser_outputs.cameras.width[data["image_idx"]])

        # Scale depth images to meter units and also by scaling applied to cameras
        scale_factor = self.depth_unit_scale_factor * self._dataparser_outputs.dataparser_scale
        depth_image = get_depth_image_from_path(
            filepath=filepath, height=height, width=width, scale_factor=scale_factor
        )

        return {"depth_image": depth_image}

    def _find_transform(self, image_path: Path) -> Union[Path, None]:
        while image_path.parent != image_path:
            transform_path = image_path.parent / "transforms.json"
            if transform_path.exists():
                return transform_path
            image_path = image_path.parent
        return None


    def get_data(self, image_idx: int, image_type: Literal["uint8", "float32"] = "float32") -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        if image_type == "float32":
            image = self.get_image_float32(image_idx)
        elif image_type == "uint8":
            image = self.get_image_uint8(image_idx)
        else:
             raise NotImplementedError(f"image_type (={image_type}) getter was not implemented, use uint8 or float32")

        data = {"image_idx": image_idx, "image": image}
        if self._dataparser_outputs.mask_filenames is not None:
            # data["mask"] = self.get_mask(image_idx)
            data["masks"] = self.get_masks(image_idx)
            # data["mask"] = torch.any(data["masks"], dim=-1).unsqueeze(-1)
            assert (
                data["masks"].shape[:2] == data["image"].shape[:2]
            ), f"Mask and image have different shapes. Got {data['masks'].shape[:2]} and {data['image'].shape[:2]}"
        metadata = self.get_metadata(data)
        data.update(metadata)
        return data


    def get_masks(self, image_idx: int):
        # mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
        # masks = get_multi_mask_tensor_from_path(filepath=mask_filepath, scale_factor=self.scale_factor)
        masks = self._dataparser_outputs.metadata["new_masks"][image_idx]
        # to tensor
        masks = torch.from_numpy(masks)
        return masks 
    
    # def get_mask(self, image_idx: int):
    #     mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
    #     masks = get_multi_mask_tensor_from_path(filepath=mask_filepath, scale_factor=self.scale_factor)
    #     return torch.any(masks, dim=-1).unsqueeze(-1)
