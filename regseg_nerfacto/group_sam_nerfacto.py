from torch.nn import Parameter
from nerfstudio.models.nerfacto import *
from .group_sam_nerfacto_field import GroupSAMNerfactoField
from .group_sam_util import GroupRenderer, FieldHeadNames
from .group_field import GarfieldFieldConfig


import torch.nn.functional as F
from nerfstudio.model_components.losses import depth_ranking_loss

from rich.console import Console

CONSOLE = Console(width=120)

@dataclass
class GroupSAMNerfactoModelConfig(NerfactoModelConfig):

    _target: Type = field(default_factory=lambda: GroupSAMNerfactoModel)
    """target class of the model"""
    use_pe: bool = True
    """Whether to use positional encoding"""
    use_grouping: bool = True
    """whether use grouping"""
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
    depth_loss_mult: float = 1e-3
    """Depth loss multiplier"""
    pass



class GroupSAMNerfactoModel(NerfactoModel):
    
    config: GroupSAMNerfactoModelConfig

    def populate_modules(self):
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        self.grouping_field = None
        self.use_native_field = False
        if self.use_native_field == True:
            self.field = GroupSAMNerfactoField(
                self.scene_box.aabb,
                hidden_dim=self.config.hidden_dim,
                num_levels=self.config.num_levels,
                max_res=self.config.max_res,
                base_res=self.config.base_res,
                features_per_level=self.config.features_per_level,
                log2_hashmap_size=self.config.log2_hashmap_size,
                hidden_dim_color=self.config.hidden_dim_color,
                hidden_dim_transient=self.config.hidden_dim_transient,
                spatial_distortion=scene_contraction,
                num_images=self.num_train_data,
                use_pred_normals=self.config.predict_normals,
                use_average_appearance_embedding=self.config.use_average_appearance_embedding,
                appearance_embedding_dim=self.config.appearance_embed_dim,
                implementation=self.config.implementation,
                feature_size=self.config.group_feature_size,
                feature_level=self.config.group_feature_level,

                use_pe = self.config.use_pe,
                use_grouping = self.config.use_grouping,
            )
        else:
            self.grouping_field = GarfieldFieldConfig(
                n_instance_dims=self.config.group_feature_size*self.config.group_feature_level,
            ).setup()


        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        self.renderer_grouping = GroupRenderer()

    def get_outputs(self, ray_bundle: RayBundle):
        # apply the camera optimizer pose tweaks
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.config.use_grouping:
            
            if self.use_native_field:

                grouping = self.renderer_grouping(field_outputs[FieldHeadNames.GROUPING], 
                                                            weights=weights, 
                                                            )

                for i in range(grouping.shape[-2]):
                    # apply pca
                    outputs[f"grouping_feature_{i}"] = grouping[..., i, :]

                outputs["grouping"] = grouping

            else:
                def gather_fn(tens):
                    return torch.gather(
                        tens, -2, best_ids.expand(*best_ids.shape[:-1], tens.shape[-1])
                    )

                dataclass_fn = lambda dc: dc._apply_fn_to_fields(gather_fn, dataclass_fn)

                num_feat_samples = 24

                grouping_weights, best_ids = torch.topk(
                    weights, num_feat_samples, dim=-2, sorted=False
                )
                grouping_samples: RaySamples = ray_samples._apply_fn_to_fields(
                    gather_fn, dataclass_fn
                )

                hash = self.grouping_field.get_hash(grouping_samples)
                hash_rendered = self.renderer_grouping(hash, weights=grouping_weights.detach().half())
                outputs["grouping"] = self.grouping_field.get_mlp(hash_rendered).float()

                outputs["grouping"] = outputs["grouping"].view(*outputs["grouping"].shape[:-1], 
                                                               self.config.group_feature_level, self.config.group_feature_size)

                for i in range(self.config.group_feature_level):
                    outputs[f"grouping_feature_{i}"] = outputs["grouping"][...,i,:]

            

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        return outputs
    


    def get_group_loss_old(self, outputs, raw_masks_list, bundle_binary_indices):


        rendered_features = outputs["grouping"]

        margin = 1.0

        pos_los = []
        neg_los = []

        def get_weight(size, mean, std):
            weight = torch.arange(size, device=self.device)
            return torch.exp(-0.5 * (weight - mean) ** 2 / std ** 2)


        for raw_mask, bundle_binary_index in zip(raw_masks_list, bundle_binary_indices):
            rendered_feature = rendered_features[bundle_binary_index]

            # pairwise distance
            # pairwise_dist = torch.cdist(rendered_feature, rendered_feature, p=2)
            # MSE
            pairwise_dist = (rendered_feature[:, None] - rendered_feature)**2

            pairwise_dist = torch.mean(pairwise_dist, dim=-1)

            num_matches = torch.mm(raw_mask, raw_mask.t()).unsqueeze(-1) # N x M_i
            
            pos_samples = num_matches > torch.arange(rendered_features.shape[-2])
            neg_samples = num_matches == torch.arange(rendered_features.shape[-2])

            # get triu
            pos_samples = torch.triu(pos_samples)
            neg_samples = torch.triu(neg_samples)

            pos_loss = torch.nanmean(pairwise_dist[pos_samples])

            neg_loss = 0

            for i in range(rendered_features.shape[-2]):
                neg_sample_i = neg_samples[..., i]
                neg_loss_i = F.relu(margin - pairwise_dist[neg_sample_i])
                gaussian_weights = get_weight(rendered_features.shape[-2], i, 0.5) 
                neg_loss += torch.nanmean(torch.nanmean(neg_loss_i, dim=0) * gaussian_weights)
                
            neg_loss = neg_loss / rendered_features.shape[-2]
            # neg_loss = F.relu(margin - pairwise_dist[neg_samples]).mean()

            pos_los.append(pos_loss)
            neg_los.append(neg_loss)



        loss_dict = {
            "sim_loss": torch.nanmean(torch.stack(pos_los)),
            "dissim_loss": torch.nanmean(torch.stack(neg_los)),
        }
        # CONSOLE.log(loss_dict["sim_loss"].item(), loss_dict["dissim_loss"].item())

        return loss_dict

    def get_group_loss(self, outputs, padded_raw_masks, chunk_size):
        
        padded_raw_masks = padded_raw_masks.to(self.device)

        num_matches = torch.mm(padded_raw_masks, padded_raw_masks.t()).unsqueeze(-1) # N x M_i

        num_chuncks = padded_raw_masks.shape[0] // chunk_size

        block_mask = torch.kron(
            torch.eye(num_chuncks, device=self.device, dtype=torch.bool),
            torch.ones((chunk_size, chunk_size), device=self.device, dtype=torch.bool),
        )

        block_mask = torch.triu(block_mask, diagonal=0)

        # get invalid mask .. later

        diag_mask = torch.eye(block_mask.shape[0], device=self.device, dtype=torch.bool)

        # pari_mask = torch.zeros

        # ptp_mse = torch.mean((outputs["grouping"][:,None] - outputs["grouping"])**2, dim =-1)

        group_features = outputs["grouping"]

        def get_weight(size, mean, std):
            weight = torch.arange(size, device=self.device)
            return torch.exp(-0.5 * (weight - mean) ** 2 / std ** 2)

        total_sim_loss = []
        total_dissim_loss = []

        margin = 1.0
        
        sim_indices = num_matches > torch.arange(self.config.group_feature_level, device=self.device)
        dissim_indices = num_matches <= torch.arange(self.config.group_feature_level, device=self.device)

        for i in range(self.config.group_feature_level):

            sim_mask = torch.where(block_mask * sim_indices[...,i] * (~diag_mask))
            dissim_mask = torch.where(block_mask * dissim_indices[...,i] * (~diag_mask))

            weight = get_weight(self.config.group_feature_level, i, 0.5)

            sum_weight = torch.sum(weight)

            weight = weight / sum_weight

            sim_mean = (group_features[sim_mask[0]] - group_features[sim_mask[1]])**2
            sim_loss = sim_mean.mean(dim=-1) * weight

            sim_loss = torch.nanmean(sim_loss)

            dissim_mean = (group_features[dissim_mask[0]]- group_features[dissim_mask[1]])**2
            dissim_loss = F.relu(margin - dissim_mean.mean(dim=-1)) * weight

            dissim_loss = torch.nanmean(dissim_loss)

            total_sim_loss.append(sim_loss)
            total_dissim_loss.append(dissim_loss)

            # sim_loss += torch.nanmean((group_features[sim_mask[0]] - group_features[sim_mask[1]])**2 * weight)
            # dissim_loss += torch.nanmean(ptp_mse[dissim_mask] * weight)

        total_sim_loss = torch.nanmean(torch.stack(total_sim_loss))
        total_dissim_loss = torch.nanmean(torch.stack(total_dissim_loss))

        # assert not torch.isnan(total_sim_loss).all()
        loss_dict = {
            "sim_loss": total_sim_loss,
            "dissim_loss": total_dissim_loss,
        }

        return loss_dict


    def get_tv_loss(self, outputs, padded_raw_masks):

        valid_pix = (padded_raw_masks[1:] == padded_raw_masks[:-1]).all(-1)

        depth = outputs["expected_depth"]

        depth_diff = depth[1:] - depth[:-1]

        tv_loss = torch.mean(torch.abs(depth_diff[valid_pix]))

        return tv_loss
        pass

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if self.training and self.config.use_depth:
            assert metrics_dict is not None and ("depth_loss" in metrics_dict or "depth_ranking" in metrics_dict)
            if "depth_ranking" in metrics_dict:
                loss_dict["depth_ranking"] = (
                    self.config.depth_loss_mult
                    * np.interp(self.step, [0, 2000], [0, 0.2])
                    * metrics_dict["depth_ranking"]
                )
            if "depth_loss" in metrics_dict:
                loss_dict["depth_loss"] = self.config.depth_loss_mult * metrics_dict["depth_loss"]
        return loss_dict
    
    def get_metrics_dict(self, outputs, batch):
        metrics_dict =  super().get_metrics_dict(outputs, batch)

        if self.training and self.config.use_depth:

            metrics_dict["depth_ranking"] = depth_ranking_loss(
                outputs["expected_depth"], batch["depth_image"].to(self.device)
            )
        
        return metrics_dict

    def get_group_render_loss(self, outputs, source_mask_bundle, learn_encoder=False):
        
        rendered_group_features = outputs["grouping"]

        if learn_encoder:
            rendered_group_features =  rendered_group_features.detach()

        render_loss = F.mse_loss(rendered_group_features, source_mask_bundle.to(device=rendered_group_features.device))

        return {
            "group_render_loss": render_loss
        }

    # def get_param_groups(self) -> Dict[str, List[Parameter]]:
    #     param_groups = super().get_param_groups()
    #     param_groups["grouping"] = list(self.grouping_field.parameters())
    #     return param_groups

    
    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        
        mask = None

        if "mask" in batch:

            # batch["mask"] : [H, W, 1]
            mask = batch["mask"].to(self.device)

            if len(mask.shape) == 2:
                mask = mask[..., None]

        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]  # Blended with background (black if random background)
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        metrics_dict = {}
        if mask is not None:

            # evaluate only valid pixels
            gt_rgb_masked = gt_rgb * mask
            predicted_rgb_masked = predicted_rgb * mask
            
            combined_rgb = torch.cat([combined_rgb, predicted_rgb_masked], dim=1)

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

        images_dict = {"img": combined_rgb, "accumulation": combined_acc, "depth": combined_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i


        if "grouping" in outputs:
            grouping_features = outputs["grouping"]
            
            for i in range(self.config.group_feature_level):


                feature_size = self.config.group_feature_size

                if mask is not None:
                    temp = torch.zeros(*grouping_features.shape[:2], 3, device=grouping_features.device)
                    # apply pca to the features
                    temp[mask.squeeze(-1)] = colormaps.apply_pca_colormap(
                        grouping_features[mask.squeeze(-1)][..., i*feature_size:(i+1)*feature_size]
                    )
                    images_dict[f"grouping_feature_{i}"] = temp
                else:
                    images_dict[f"grouping_feature_{i}"] = colormaps.apply_pca_colormap(
                        grouping_features[..., i*feature_size:(i+1)*feature_size]
                    )

        return metrics_dict, images_dict


    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        if self.grouping_field is not None:
            param_groups["grouping"] = list(self.grouping_field.parameters())
        return param_groups