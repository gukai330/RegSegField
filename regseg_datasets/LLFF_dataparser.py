from nerfstudio.data.dataparsers.colmap_dataparser import *
import sqlite3
import numpy as np
import imageio
from regseg_nerfacto.mask_matching_utils import global_object_id
import os
import matplotlib.pyplot as plt
import shutil

def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) / 2147483647
    # to int
    image_id1 = int(image_id1)
    image_id2 = int(image_id2)
    return image_id1, image_id2

@dataclass
class LLFFDataParserConfig(ColmapDataParserConfig):
    """Nerfstudio dataset config"""
    _target: Type = field(default_factory=lambda: LLFFDataParser)
    """target class to instantiate"""
    input_views: int = 12
    """Number of input views"""
    llff_hold: int = 8
    """Number of images to hold when using llff hold"""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """center method, poses or focus"""
    load_masks: bool = True
    """load masks or not"""
    masks_path: Path = Path("masks_allprompts")
    """masks path"""
    colmap_path: Path = Path("sparse/0")
    """colmap path"""
    downscale_factor: Optional[int] = 8
    """downscale factor"""
    use_depth: bool = False
    """use depth image or not"""
    max_2D_matches_per_3D_point: int = -1
    """max number of 2D matches per 3D point"""
    use_llff_hold: bool = True
    """Number of images to hold when using llff hold"""

class LLFFDataParser(ColmapDataParser):
    config: LLFFDataParserConfig
    def __init__(self, config: LLFFDataParserConfig):
        super().__init__(config)
        self.input_views = config.input_views
        self.center_method = config.center_method
        # self._downscale_factor = 8

    

    def _get_correspondences(self, keypoints_pairs, image_filenames, mask_filenames, indices):
        # 
        selected_image_filenames = [image_filenames[i] for i in indices]
        selected_mask_filenames = [mask_filenames[i] for i in indices]
        
        selected_masks = []
        # load all the masks 
        for mask_file in selected_mask_filenames:
            # check if mask exists
            mask = imageio.imread(mask_file)
            selected_masks.append(mask)

        # preprocess for dense matches
        eroded_masks = self._preprocess_masks(selected_masks,3)
        # make pairs
        ind_pairs = []
        # create i cross i empty list
        mat_container = []
        for i in range(len(selected_image_filenames)):
            mat_container.append([])
            for j in range(len(selected_image_filenames)):
                mat_container[i].append([])

        for i in range(len(selected_image_filenames)):
            for j in range(i+1, len(selected_image_filenames)):
                ind_pairs.append((i, j))

        keypoints_sorted = []
        for m in selected_masks:
            num_masks = np.max(np.unique(m))+1
            keypoints_sorted.append([[] for i in range(num_masks)])

        for ind in ind_pairs:
            # image_key = (selected_image_filenames[ind[0]], selected_image_filenames[ind[1]])
            # get file name ask image_key
            image_key = (selected_image_filenames[ind[0]].name, selected_image_filenames[ind[1]].name)
            # mask1, mask2 = selected_masks[ind[0]], selected_masks[ind[1]] 
            # use eroded masks incase of mismatches  
            mask1, mask2 = eroded_masks[ind[0]], eroded_masks[ind[1]]
            n_masks1 = selected_masks[ind[0]].max()+1
            n_masks2 = selected_masks[ind[1]].max()+1
            corr_mat = np.zeros((n_masks1, n_masks2))

            if image_key in keypoints_pairs:
                # no matches
                               

                for point1_xy, point2_xy in zip(*keypoints_pairs[image_key]):
                    # get the mask value               
                    y1,x1 = point1_xy.astype(int)
                    y2,x2 = point2_xy.astype(int)

                    # clip the mask values
                    y1 = np.clip(y1, 0, mask1.shape[1]-1)
                    x1 = np.clip(x1, 0, mask1.shape[0]-1)
                    y2 = np.clip(y2, 0, mask2.shape[1]-1)
                    x2 = np.clip(x2, 0, mask2.shape[0]-1)
                    
                    mask_val1 = mask1[x1,y1]
                    mask_val2 = mask2[x2,y2]

                    if mask_val1 == 0 or mask_val2 == 0:
                        # skip no mask area
                        continue
                    keypoints_sorted[ind[0]][mask_val1].append((x1,y1))
                    keypoints_sorted[ind[1]][mask_val2].append((x2,y2))
                    corr_mat[mask_val1, mask_val2] += 1

            mat_container[ind[0]][ind[1]] = corr_mat
            
        # for debugging
        self.keypoints_sorted = keypoints_sorted

        return selected_masks, mat_container
    
    def _preprocess_masks(self, masks, size= 5):

        # erode the masks
        eroded_masks = []
        for mask in masks:
            n_objects = np.max(np.unique(mask))+1
            one_hotmask = np.eye(n_objects)[mask]
            # to tensor
            one_hotmask = torch.tensor(one_hotmask)
            # erode each channel
            for i in range(one_hotmask.shape[-1]):
                # erode with torch
                one_hotmask[:, :, i] = -torch.nn.functional.max_pool2d(-one_hotmask[:, :, i].unsqueeze(0).unsqueeze(0), kernel_size=size, stride=1, padding=size//2).squeeze(0).squeeze(0)
            
            # convert back to numpy
            one_hotmask = one_hotmask.cpu().numpy()
            # one-hot to integer
            eroded_masks.append(np.argmax(one_hotmask, axis=-1))

            # plot eroded masks with title 
            # plt.imshow(eroded_masks[-1])
            # plt.title("eroded mask")
            # plt.show()
            # plt.imshow(mask)
            # plt.title("original mask")
            # plt.show()
            
        return eroded_masks


    def _convert_masks_to_global_id(self, masks, mat_container):
        mask_num = []
        for i in range(len(masks)):
            mask_num.append(len(np.unique(masks[i])))
        
        # convert the mask to global id
        new_masks = []
        global_id_mapping = global_object_id(mask_num, mat_container)
        self.global_id_mapping = global_id_mapping
        
        total_objects = np.max([np.max(g) for g in global_id_mapping]) + 1
        # assign unmathced masks to the same id -1
        
        # find the id which appear only once
        for i in range(total_objects):
            appearance_count = 0
            for j in range(len(global_id_mapping)):
                if i in global_id_mapping[j]:
                    appearance_count += 1
            if appearance_count == 1:
                for j in range(len(global_id_mapping)):
                    global_id = global_id_mapping[j]
                    if i in global_id:
                        global_id[global_id==i] = -1
                        global_id_mapping[j] = global_id
        
        self.global_id_mapping = global_id_mapping
                        


        for i in range(len(masks)):
            new_mask = np.zeros_like(masks[i]).astype(np.int32)
            for old_id, new_id in enumerate(global_id_mapping[i]):
                new_mask[masks[i] == old_id] = new_id
            new_masks.append(new_mask)
        
        total_objects = np.max(new_masks) + 1
        print("total_objects", total_objects)

        return new_masks
    
    def _visualize_mask_matching(self, images_files, masks):
        import matplotlib.pyplot as plt
        
        # load images
        images = []
        for image_file in images_files:
            image = imageio.imread(image_file)
            images.append(image)

        print("input views", len(images))
        total_obj = np.unique(masks).shape[0]
        print("total_objects", total_obj)


        for i in np.unique(masks):
            
            fig, ax = plt.subplots(1, len(images), figsize = (40, 10))
            # make it bigger
            matches = 0
            for j in range(len(images)):
                ax[j].axis("off")
                if np.sum(masks[j] == i) <= 0:
                    continue 
                matches += 1
                ax[j].imshow(images[j])
                ax[j].imshow(masks[j] == i, alpha = 0.5)

                # global id to local
                local_id = np.where(self.global_id_mapping[j] == i)[0][0]

                # get points
                points = self.keypoints_sorted[j][local_id]

                ax[j].scatter([p[1] for p in points], [p[0] for p in points], color = 'red')


            if matches>=2:
                # save the figure
                # make dir
                
                if not os.path.exists("mask_matching"):
                    os.makedirs("mask_matching")
                # save current figure
                # plt.savefig(f"mask_matching/{i}.png")
                plt.show()
            else:
                # print(matches)
                plt.close()
        plt.close()
        pass


    def _load_matches(self, downscale_factor = 1, ):
        # load colmap database
        database_path = self.config.data / "database.db"
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()

        images = {}
        cursor.execute("SELECT image_id, camera_id, name FROM images;")

        for row in cursor:
            image_id = row[0]
            image_name = row[2]
            images[image_id] = image_name

        keypoints = {}
      # get keypoints 
        for image_id in images.keys():
            cursor.execute(
                    "SELECT data FROM keypoints WHERE image_id=?;", (image_id,)
                )
            row = next(cursor)
            temp = np.fromstring(row[0], dtype=np.float32).reshape(-1, 6)
            keypoints[image_id] = temp

        min_num_matches = 2

        cursor.execute(
            "SELECT pair_id, data FROM two_view_geometries WHERE rows>=?;",
            (min_num_matches,),
        )

        image_pairs = []
        image_names = []
        keypoints_pairs = {}
        


        for row in cursor:
            pair_id = row[0]
            inlier_matches = np.fromstring(row[1], dtype=np.uint32).reshape(
                -1, 2
            )
            image_id1, image_id2 = pair_id_to_image_ids(pair_id)
            image_name1 = images[image_id1]
            image_name2 = images[image_id2]
            print(
                "%s %s %d\n"
                % (image_name1, image_name2, inlier_matches.shape[0])
            )
            # load images
            # image1 = imageio.imread(
                
            #     image_path + "/" + image_name1
            # )
            # image2 = imageio.imread(
                
            #     image_path + "/" + image_name2

            # )

            # image_pairs.append((image1, image2))
            image_names.append((image_name1, image_name2))

            # plot matches
            points_1 = inlier_matches[:,0]
            points_2 = inlier_matches[:,1]
            
            # scale the points

            points_1_xy = keypoints[image_id1][points_1][:,:2]/downscale_factor
            points_2_xy = keypoints[image_id2][points_2][:,:2]/downscale_factor

            # keypoints_pairs.append((points_1_xy, points_2_xy))
            
            key = (image_name1, image_name2)            
               
            keypoints_pairs[key] = (points_1_xy, points_2_xy)
            key_inv = (key[1], key[0])
            keypoints_pairs[key_inv] = (points_2_xy, points_1_xy)

        return keypoints_pairs


    def _generate_dataparser_outputs(self, split: str = "train", **kwargs):


        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."
        colmap_path = self.config.data / self.config.colmap_path
        assert colmap_path.exists(), f"Colmap path {colmap_path} does not exist."

        meta = self._get_all_images_and_cameras(colmap_path)

        frames = meta["frames"]
        # sort by file_path
        # frames = sorted(frames, key=lambda x: x["file_path"])
        meta["frames"] = frames
        camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        poses = []

        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        for frame in meta["frames"]:
            fx.append(float(frame["fl_x"]))
            fy.append(float(frame["fl_y"]))
            cx.append(float(frame["cx"]))
            cy.append(float(frame["cy"]))
            height.append(int(frame["h"]))
            width.append(int(frame["w"]))
            distort.append(
                camera_utils.get_distortion_params(
                    k1=float(frame["k1"]) if "k1" in frame else 0.0,
                    k2=float(frame["k2"]) if "k2" in frame else 0.0,
                    k3=float(frame["k3"]) if "k3" in frame else 0.0,
                    k4=float(frame["k4"]) if "k4" in frame else 0.0,
                    p1=float(frame["p1"]) if "p1" in frame else 0.0,
                    p2=float(frame["p2"]) if "p2" in frame else 0.0,
                )
            )

            image_filenames.append(Path(frame["file_path"]))
            poses.append(frame["transform_matrix"])
            if "mask_path" in frame:
                mask_filenames.append(Path(frame["mask_path"]))
            if "depth_path" in frame:
                depth_filenames.append(Path(frame["depth_path"]))

        assert len(mask_filenames) == 0 or (
            len(mask_filenames) == len(image_filenames)
        ), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """
        assert len(depth_filenames) == 0 or (
            len(depth_filenames) == len(image_filenames)
        ), """
        Different number of image and depth filenames.
        You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
        """
        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor

        poses[:, :3, 3] *= scale_factor
        image_filenames_original = image_filenames
        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        indices = self._get_image_indices(image_filenames, split)
        

        # self._export_subset(image_filenames, indices)

        image_filenames, mask_filenames, depth_filenames, downscale_factor = self._setup_downscale_factor(
            image_filenames, mask_filenames, depth_filenames
        )

        metadata = {}
        if self.config.load_masks and split == "train":
            if str(self.config.masks_path) == "masks_manual":

                new_masks = []
                for i in indices:

                    m = imageio.imread(mask_filenames[i])
                    new_masks.append(m)

            else:
                keypoints_pairs = self._load_matches(downscale_factor)
                
                masks, mat_container = self._get_correspondences(keypoints_pairs, image_filenames_original,mask_filenames, indices)

                new_masks = self._convert_masks_to_global_id(masks, mat_container)

            metadata["new_masks"] = new_masks
            
        else:
            mask_filenames = []

        def surfix_to_png(filename):
            # return filename.with_suffix(".png")
            return filename 

        image_filenames = [surfix_to_png(image_filenames[i]) for i in indices]
        mask_filenames = [surfix_to_png(mask_filenames[i]) for i in indices] if len(mask_filenames) > 0 else []
        depth_filenames = [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []

        if self.config.load_masks and split == "train":
            # self._visualize_mask_matching(image_filenames, new_masks)
            pass
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
            )
        )

        fx = torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        fy = torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        cx = torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        cy = torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        height = torch.tensor(height, dtype=torch.int32)[idx_tensor]
        width = torch.tensor(width, dtype=torch.int32)[idx_tensor]
        distortion_params = torch.stack(distort, dim=0)[idx_tensor]

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
        )

        cameras.rescale_output_resolution(scaling_factor=1.0 / downscale_factor)

        if "applied_transform" in meta:
            applied_transform = torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype)
            transform_matrix = transform_matrix @ torch.cat(
                [applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)], 0
            )
        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale

        
        if self.config.load_3D_points:
            train_images =[image_filenames_original[i].name for i in indices]
            # Load 3D points
            metadata.update(self._load_3D_points(colmap_path, transform_matrix, scale_factor, train_images=train_images))


        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                **metadata,
            },
        ) 
        return dataparser_outputs
    
    
    # def _setup_downscale_factor(
    #     self, image_filenames: List[Path], mask_filenames: List[Path], depth_filenames: List[Path]
    # ):
    #     """
    #     Setup the downscale factor for the dataset. This is used to downscale the images and cameras.
    #     """

    #     self._downscale_factor = self.config.downscale_factor

    #     def get_fname(filepath: Path) -> Path:
    #         """Returns transformed file name when downscale factor is applied"""
    #         parts = list(filepath.parts)
    #         parts[-2] += f"_{self._downscale_factor}"
    #         filepath = Path(*parts)
    #         return self.config.data / filepath


        
        
    #     # Return transformed filenames
    #     if self._downscale_factor > 1:

    #         image_filenames = [get_fname(fp) for fp in image_filenames]
    #         mask_filenames = [get_fname(fp) for fp in mask_filenames]
    #         depth_filenames = [get_fname(fp) for fp in depth_filenames]
    #     assert isinstance(self._downscale_factor, int)
    #     return image_filenames, mask_filenames, depth_filenames, self._downscale_factor


    def _load_3D_points(self, colmap_path: Path, transform_matrix: torch.Tensor, scale_factor: float, train_images: List[str] = None):
        if (colmap_path / "points3D.bin").exists():
            colmap_points = colmap_utils.read_points3D_binary(colmap_path / "points3D.bin")
        elif (colmap_path / "points3D.txt").exists():
            colmap_points = colmap_utils.read_points3D_text(colmap_path / "points3D.txt")
        else:
            raise ValueError(f"Could not find points3D.txt or points3D.bin in {colmap_path}")
        points3D = torch.from_numpy(np.array([p.xyz for p in colmap_points.values()], dtype=np.float32))
        points3D = (
            torch.cat(
                (
                    points3D,
                    torch.ones_like(points3D[..., :1]),
                ),
                -1,
            )
            @ transform_matrix.T
        )
        points3D *= scale_factor

        # Load point colours
        points3D_rgb = torch.from_numpy(np.array([p.rgb for p in colmap_points.values()], dtype=np.uint8))
        points3D_num_points = torch.tensor([len(p.image_ids) for p in colmap_points.values()], dtype=torch.int64)
        out = {
            "points3D_xyz": points3D,
            "points3D_rgb": points3D_rgb,
            "points3D_error": torch.from_numpy(np.array([p.error for p in colmap_points.values()], dtype=np.float32)),
            "points3D_num_points2D": points3D_num_points,
        }
        if self.config.max_2D_matches_per_3D_point != 0:
            if (colmap_path / "images.txt").exists():
                im_id_to_image = colmap_utils.read_images_text(colmap_path / "images.txt")
            elif (colmap_path / "images.bin").exists():
                im_id_to_image = colmap_utils.read_images_binary(colmap_path / "images.bin")
            else:
                raise ValueError(f"Could not find images.txt or images.bin in {colmap_path}")
            downscale_factor = self._downscale_factor
            max_num_points = int(torch.max(points3D_num_points).item())
            if self.config.max_2D_matches_per_3D_point > 0:
                max_num_points = min(max_num_points, self.config.max_2D_matches_per_3D_point)
            points3D_image_ids = []
            points3D_image_xy = []

            match_index = torch.ones_like(points3D_num_points)

            for match_ind, p in enumerate(colmap_points.values()):
                nids = np.array(p.image_ids, dtype=np.int64)
                nxy_ids = np.array(p.point2D_idxs, dtype=np.int32)

                image_names = [im_id_to_image[im_id].name for im_id in nids]


                if train_images is not None:
                    # find intersection
                    common_image = len(set(train_images).intersection(image_names))
                    if common_image < 2:
                        
                        match_index[match_ind] = 0
                        # continue
                    

                if self.config.max_2D_matches_per_3D_point != -1:
                    # Randomly sample 2D matches
                    idxs = np.argsort(p.error)[: self.config.max_2D_matches_per_3D_point]
                    nids = nids[idxs]
                    nxy_ids = nxy_ids[idxs]
                nxy = [im_id_to_image[im_id].xys[pt_idx] for im_id, pt_idx in zip(nids, nxy_ids)]
                nxy = torch.from_numpy(np.stack(nxy).astype(np.float32))
                nids = torch.from_numpy(nids)
                assert len(nids.shape) == 1
                assert len(nxy.shape) == 2
                points3D_image_ids.append(
                    torch.cat((nids, torch.full((max_num_points - len(nids),), -1, dtype=torch.int64)))
                )
                points3D_image_xy.append(
                    torch.cat((nxy, torch.full((max_num_points - len(nxy), nxy.shape[-1]), 0, dtype=torch.float32)))
                    / downscale_factor
                )
            out["points3D_image_ids"] = torch.stack(points3D_image_ids, dim=0)
            out["points3D_points2D_xy"] = torch.stack(points3D_image_xy, dim=0)

            # filter all the tensors
            for k in out.keys():
                out[k] = out[k][match_index.bool()]

            print("loaded 3d points", torch.sum(match_index), "out of", len(match_index))

        return out


    def _export_subset(self, image_filenames, indices):
        image_filenames = [image_filenames[i] for i in indices]
        
        # copy to n_view folder
        # make a new folder
        n_view_folder =  f"{self.config.input_views}_view"

        if not os.path.exists(os.path.join(self.config.data, n_view_folder)):
            os.makedirs(os.path.join(self.config.data, n_view_folder))
        
        for i in range(len(image_filenames)):
            shutil.copy(os.path.join(self.config.data,image_filenames[i]),                        
                         os.path.join(self.config.data, n_view_folder, os.path.basename(image_filenames[i])))    
        

    def _get_image_indices(self, image_filenames, split):
        all_indices = np.arange(len(image_filenames))

        if not self.config.use_llff_hold:
            return all_indices

        # return all_indices
        if split == "train":
            indices = all_indices[all_indices % self.config.llff_hold == 0]
        else:
            indices = all_indices[all_indices % self.config.llff_hold != 0]

        if split == "train" and self.config.input_views < len(indices):
            sub_ind = np.linspace(0, len(indices) - 1, self.config.input_views)
            sub_ind = [round(i) for i in sub_ind]
            indices = [indices[i] for i in sub_ind]

        return indices 
    