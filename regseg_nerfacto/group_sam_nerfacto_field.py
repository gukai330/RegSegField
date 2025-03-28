from nerfstudio.fields.nerfacto_field import *
from nerfactope.nerfactope_field import NerfactoPEField
from nerfstudio.field_components.field_heads import FieldHead
from .group_sam_util import GroupFieldHead


class GroupSAMNerfactoField(NerfactoPEField):

    aabb: Tensor

    def __init__(
        self,  *args, **kwargs
    ) -> None:
        # get feature and pop
        feature_level = kwargs.pop("feature_level", 4)
        feature_size = kwargs.pop("feature_size", 16)
        self.use_grouping = kwargs.pop("use_grouping", True)
        super().__init__(*args, **kwargs)

        implementation = kwargs.get("implementation", "tcnn")

        
        #
        if self.use_grouping:
            self.mlp_grouping = MLP(
                in_dim=self.geo_feat_dim,
                num_layers = 2,
                layer_width=64, 
                out_dim = 64,
                activation=nn.ReLU(),
                out_activation=None,
                implementation=implementation
            )

            self.field_head_grouping = GroupFieldHead(
                in_dim = self.mlp_grouping.get_out_dim(),
                feature_level=feature_level,
                feature_size=feature_size,
            )
    
    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # appearance
        if self.training:
            embedded_appearance = self.embedding_appearance(camera_indices)
        else:
            if self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                ) * self.embedding_appearance.mean(dim=0)
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                )

        # transients
        if self.use_transient_embedding and self.training:
            embedded_transient = self.embedding_transient(camera_indices)
            transient_input = torch.cat(
                [
                    density_embedding.view(-1, self.geo_feat_dim),
                    embedded_transient.view(-1, self.transient_embedding_dim),
                ],
                dim=-1,
            )
            x = self.mlp_transient(transient_input).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.UNCERTAINTY] = self.field_head_transient_uncertainty(x)
            outputs[FieldHeadNames.TRANSIENT_RGB] = self.field_head_transient_rgb(x)
            outputs[FieldHeadNames.TRANSIENT_DENSITY] = self.field_head_transient_density(x)

        # semantics
        if self.use_semantics:
            semantics_input = density_embedding.view(-1, self.geo_feat_dim)
            if not self.pass_semantic_gradients:
                semantics_input = semantics_input.detach()

            x = self.mlp_semantics(semantics_input).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.SEMANTICS] = self.field_head_semantics(x)

        if self.use_grouping:
            grouping_input = density_embedding.view(-1, self.geo_feat_dim)

            x = self.mlp_grouping(grouping_input).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.GROUPING] = self.field_head_grouping(x)

        # predicted normals
        if self.use_pred_normals:
            positions = ray_samples.frustums.get_positions()

            positions_flat = self.position_encoding(positions.view(-1, 3))
            pred_normals_inp = torch.cat([positions_flat, density_embedding.view(-1, self.geo_feat_dim)], dim=-1)

            x = self.mlp_pred_normals(pred_normals_inp).view(*outputs_shape, -1).to(directions)
            outputs[FieldHeadNames.PRED_NORMALS] = self.field_head_pred_normals(x)

        h = torch.cat(
            [
                d,
                density_embedding.view(-1, self.geo_feat_dim),
                embedded_appearance.view(-1, self.appearance_embedding_dim),
            ],
            dim=-1,
        )
        rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
        outputs.update({FieldHeadNames.RGB: rgb})

        return outputs