# create a network
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
# with mask ids as input

from rich.prompt import Confirm

from nerfstudio.utils.rich_utils import CONSOLE,status
from nerfstudio.utils.scripts import run_command
from PIL import Image

from functools import partial
import cv2

import skimage

MAX_AUTO_RESOLUTION = 1600
# quaterion to rotate matrix

def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion into a rotation matrix.
    
    Parameters:
    q (Tensor): A tensor containing the quaternion (w, x, y, z).
    
    Returns:
    Tensor: A 3x3 rotation matrix.
    """

    q = torch.tensor([1,0,0,0]) + q    
    # Normalize the quaternion
    q = q / torch.norm(q)
    
    # Extract components for clarity
    w, x, y, z = q
    
    # Compute the rotation matrix components
    r00 = 1 - 2 * (y**2 + z**2)
    r01 = 2 * (x*y - z*w)
    r02 = 2 * (x*z + y*w)
    
    r10 = 2 * (x*y + z*w)
    r11 = 1 - 2 * (x**2 + z**2)
    r12 = 2 * (y*z - x*w)
    
    r20 = 2 * (x*z - y*w)
    r21 = 2 * (y*z + x*w)
    r22 = 1 - 2 * (x**2 + y**2)
    
    # Construct the rotation matrix
    rotation_matrix = torch.tensor([[r00, r01, r02],
                                    [r10, r11, r12],
                                    [r20, r21, r22]])
    
    return rotation_matrix


class MaskPermutation(nn.Module):
    def __init__(self, data_dict, output_dim = 16, output_levels = 4, hidden_dim = 64):
        super(MaskPermutation, self).__init__()

        # make n heads 
        # self.input_views = input_views
        # self.mask_num_per_view = mask_num_per_view
        # self.mask_levels_per_view = mask_levels_per_view

        self.use_hierarchical = True

        self.input_heads = {}
        self.hierarchical_heads = []

        input_layers = 3
        # a linear permutation
        for key in data_dict.keys():
            # make input layers
            input_module = [nn.Linear(data_dict[key]["image_masks"].shape[0], hidden_dim)]
            for i in range(input_layers-1):
                input_module.append(nn.Linear(hidden_dim, hidden_dim))
            
            self.input_heads[key] = nn.ModuleList(input_module)
        

        # self.input_heads = nn.ModuleList(self.input_heads)
        self.input_heads = nn.ModuleDict(self.input_heads)
        
        self.scale_head = nn.Linear(hidden_dim, 1)

        layers = 3
        common_part_layers = []
        for i in range(layers):
            common_part_layers.append(nn.Linear(hidden_dim, hidden_dim))

        for i in range(output_levels):
            if i  == 0 or not self.use_hierarchical:
                output = nn.Linear(hidden_dim, output_dim)
            else:
                output = nn.Linear(output_dim + hidden_dim, output_dim)
            # initialize with identity
            nn.init.eye_(output.weight)
            nn.init.zeros_(output.bias)
            self.hierarchical_heads.append(output)

        self.hierarchical_heads = nn.ModuleList(self.hierarchical_heads)

        self.output_dim = output_dim
        self.output_levels = output_levels

        self.activation = nn.LeakyReLU(0.2)
        self.out_activation = nn.Sigmoid()

        self.output_heads = nn.ModuleList(common_part_layers)

        # define the cos similarity loss
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)


    def _permute_input(self, image_id, mask_ids):
        x = mask_ids
        for i, layer in enumerate(self.input_heads[image_id]):
            x = layer(x)
            x = self.activation(x)
        
        return x


    def _get_output(self, in_tensor,):
        x = in_tensor
        for i, layer in enumerate(self.output_heads,):
            x = layer(x)
            x = self.activation(x)

        # x = F.normalize(x, p=2, dim=-1)
        
        out_feature = x
        temp = []

        for i in range(self.output_levels):
            if i == 0 or not self.use_hierarchical:
                x = out_feature
            else:
                x = torch.cat([x, out_feature], dim = -1)

            x = self.hierarchical_heads[i](x)
            # rot_mat = quaternion_to_rotation_matrix(self.higher_paramters[i])
            # x = torch.matmul(x, rot_mat)
            temp.append(x)

        x = torch.stack(temp, dim = -2)
        # normalize
        x = F.normalize(x, p=2, dim=-1)
        # x = self.out_activation(x)
        # norm = torch.norm(x, p=2, dim=-1)
        # eps = 1e-6
        # x = x / (norm.unsqueeze(-1) + eps)

        return x
    


    def get_features(self, image_id, mask_ids):
        x = self._permute_input(image_id, mask_ids)
        scale = self.scale_head(x)
        # add a input conditioned scale adjustment
        scale = self.activation(scale)
        return self._get_output(x,), scale
    
    def get_match_loss(self, image_id1, image_id2, mask_ids1, mask_ids2):
        # x1 = self.input_heads[image_id1](mask_ids1)
        # x2 = self.input_heads[image_id2](mask_ids2)

        # # agg_level1 = self.agg_level[image_id1]
        # # agg_level2 = self.agg_level[image_id2]
        
        # # weight_index1 = torch.sum(mask_ids1)/agg_level1
        # # weight_index2 = torch.sum(mask_ids2)/agg_level2

        # # #use hard weighting on the local permutation
        # # x1[int(weight_index1):] *= 0.1
        # # x2[int(weight_index2):] *= 0.1

        # y1 = self.output_heads(x1)
        # y2 = self.output_heads(x2)

        # TODO use softweighting on the global permutation

        y1,s1 = self.get_features(image_id1, mask_ids1)
        y2,s2 = self.get_features(image_id2, mask_ids2)
        

        # make sure the matched point have the same direction and length
        # learn a global scale based on the matching
        sim_loss = 1 - self.cos_sim(y1.unsqueeze(0), y2.unsqueeze(0)) 
        # match_scale_loss = self.scale_loss(y1*s1, y2*s2)
        match_scale_loss = 0.0


        # max_level_overlap = self.mask_levels_per_view[image_id1]

        # scale_loss = 0.0
        # if torch.sum(mask_ids1) > 1:
        #     y1_len_tar = torch.sum(mask_ids1)/ max_level_overlap
        #     scale_loss += (torch.norm(y1, p=2) - y1_len_tar)**2
        
        # if torch.sum(mask_ids2) > 1:
        #     y2_len_tar = torch.sum(mask_ids2)/ max_level_overlap
        #     scale_loss += (torch.norm(y2, p=2) - y2_len_tar)**2


        return sim_loss, match_scale_loss

    def get_separate_loss(self, image_id, mask_id1, mask_id2):

        assert not torch.allclose(mask_id1, mask_id2) 

        y1,s1 = self.get_features(image_id, mask_id1)
        y2,s2= self.get_features(image_id, mask_id2) 

        max_level_overlap = self.mask_levels_per_view[image_id]      

        sim_loss  = torch.tensor([torch.nan])
        dis_sim_loss = torch.tensor([torch.nan])


        if torch.any(torch.logical_and(mask_id1, mask_id2)):
            # there is at least one overlap
            
            # intersection_mask = torch.logical_and(mask_id1, mask_id2)

            # # num of overlap
            # num_overlap = torch.sum(intersection_mask)


            # num of union
            

            # loss
            # weight_ind = (num_overlap / max_level_overlap) * (len(y1) - 1)

            # weight = torch.ones_like(y1) 
            # weight[int(weight_ind):] *= -1

            #
            # positive_loss = F.mse_loss(y1[:int(weight_ind)], y2[:int(weight_ind)])
            # negative_loss = 1 - F.mse_loss(y1[int(weight_ind):], y2[int(weight_ind):])
            # relu for negative

            # tqdm.write(f"positive size {len(y1[:int(weight_ind)])}, negative size {len(y1[int(weight_ind):])}")
            # loss = positive_loss + torch.relu(negative_loss)
            

            sim_loss = 1 - self.cos_sim(y1.unsqueeze(0), y2.unsqueeze(0))
        else:
            # make the result separate
            dis_sim_loss = 1 + self.cos_sim(y1.unsqueeze(0), y2.unsqueeze(0)) 
        
        scale_loss = torch.tensor([0.0])

        if torch.sum(mask_id1) > 1:
            y1_len_tar = torch.sum(mask_id1)/ max_level_overlap
            scale_loss += (torch.norm(y1, p=2) - y1_len_tar)**2
        
        if torch.sum(mask_id2) > 1:
            y2_len_tar = torch.sum(mask_id2)/ max_level_overlap
            scale_loss += (torch.norm(y2, p=2) - y2_len_tar)**2

        return sim_loss, dis_sim_loss, scale_loss


def train_inter_step(model, data_dicts, keypoints_pairs):


    ind_pairs = []
    for i in range(len(data_dicts)):
        for j in range(i+1, len(data_dicts)):
            ind_pairs.append((i, j))


    # load pairs
    match_sim_losses = []
    match_scale_losses = []

    for ind in tqdm(ind_pairs, leave=False):
        # image_key = (selected_image_filenames[ind[0]], selected_image_filenames[ind[1]])
        
        image_1 = list(data_dicts.keys())[ind[0]]
        image_2 = list(data_dicts.keys())[ind[1]]

        image_key = (image_1.replace('.pkl','.png'), image_2.replace('.pkl','.png'))
        # get file name ask image_key
        image_key2 = (image_1.replace('.pkl','.JPG'), image_2.replace('.pkl','.JPG'))

        mask1 = data_dicts[image_1]['image_masks']
        mask2 = data_dicts[image_2]['image_masks']

        # mask tensor to c, h, w to h, w, c
        mask1 = mask1.permute(1,2,0)
        mask2 = mask2.permute(1,2,0)
        

        if not image_key in keypoints_pairs:

            if image_key2 in keypoints_pairs:
                image_key = image_key2

        if image_key[::-1] in keypoints_pairs:

            image_key = image_key[::-1]
            mask1, mask2 = mask2, mask1
            image_1, image_2 = image_2, image_1

        if image_key in keypoints_pairs:        

            points1_xy, points2_xy = keypoints_pairs[image_key]

            # clip the xy to the mask shape (finish here)
            # Clip for mask1
            points1_xy[:, 0] = np.clip(points1_xy[:, 0], 0, mask1.shape[1] - 1)  # Clip x to width
            points1_xy[:, 1] = np.clip(points1_xy[:, 1], 0, mask1.shape[0] - 1)  # Clip y to height

            # Clip for mask2
            points2_xy[:, 0] = np.clip(points2_xy[:, 0], 0, mask2.shape[1] - 1)  # Clip x to width
            points2_xy[:, 1] = np.clip(points2_xy[:, 1], 0, mask2.shape[0] - 1)  # Clip y to height


            # get the mask value
            mask_vals1 = mask1[points1_xy[:,1], points1_xy[:,0]]
            mask_vals2 = mask2[points2_xy[:,1], points2_xy[:,0]]

            # get features
        
            features1,_ = model.get_features(image_1, mask_vals1)
            features2,_  = model.get_features(image_2, mask_vals2)


            # for level in range(model.output_levels):
            #     # sim_loss = 1 - torch.nn.functional.cosine_similarity(
            #     #     features1[..., level, :], features2[..., level, :]
            #     # )

            #     # use l2

            #     sim_loss = 
            #     match_sim_losses.append(sim_loss)

            # match_sim_losses  = torch.mean(torch.concat(match_sim_losses))
            
            # feature1 = features1.view(*features1.shape[:-2], -1)
            # feature2 = features2.view(*features2.shape[:-2], -1)


        

            # use l2
            # match_sim_losses.append(torch.nn.functional.mse_loss(feature1, feature2))

            se = (features1 - features2)**2
            
            mse = torch.mean(se, dim = -1)


            # weight = 0.8** torch.arange(mse.shape[-1])
            # weight = weight / torch.sum(weight)

            weight = 1.0
            
            match_sim_losses.append(torch.mean(torch.sum(mse * weight, dim=-1)))


            # sim_loss = torch.nn.functional.cosine_similarity(features1, features2)

            # for point1_xy, point2_xy in tqdm(zip(*keypoints_pairs[image_key]), leave=False):
            #     # get the mask value               
            #     y1,x1 = point1_xy.astype(int)
            #     y2,x2 = point2_xy.astype(int)
                
            #     mask_val1 = mask1[x1,y1]
            #     mask_val2 = mask2[x2,y2]

            #     image_id1 = ind[0]
            #     image_id2 = ind[1]

            #     match_sim_loss, match_scale_loss = model.get_match_loss(image_id1=image_id1, image_id2=image_id2, mask_ids1=mask_val1, mask_ids2=mask_val2)

            #     # check match loss

            #     match_sim_losses.append(match_sim_loss)
            #     match_scale_losses.append(match_scale_loss)

        
    # calc mean loss
    loss_dict = {
        'match_sim_loss': torch.mean(torch.stack(match_sim_losses)),
        # 'match_scale_loss': torch.mean(torch.stack(match_scale_losses))
    }


    return loss_dict


def train_inner_step(model, data_dicts, ):

    losses = []

    # use random image index

    rand_image_id = torch.randperm(len(data_dicts))
    total_simloss = 0
    total_dissimloss = 0
    
    for i in tqdm(rand_image_id, leave=False):

        name = list(data_dicts.keys())[i]
        data = data_dicts[name]

        image_mask = data['image_masks']
        # mask_score = data['mask_scores']
        # max_overlap = data['max_overlap']

    
        image_mask = image_mask.permute(1,2,0)

        image_mask = torch.tensor(image_mask, dtype=torch.float32)

        # get unique pixel values

        flat_image_mask = image_mask.flatten(0,1)

        unique_pixels, inverse_indices, count = torch.unique(flat_image_mask, dim = 0, return_inverse = True, return_counts=True)


        # boarder pixels = # identify border pixels from unique pixels
        
        
                


        # visual_masks  = torch.zeros(flat_image_mask.shape[0])



        # create matrix



        # combination 2
        pairs = torch.combinations(torch.arange(unique_pixels.shape[0]), r=2)
        
        # shuffle the pairs with torch 
        ind_pairs = torch.randperm(pairs.shape[0])

        sim_losses = []
        dis_sim_losses = []
        scale_losses = []


        overlaps = torch.mm(unique_pixels, unique_pixels.T)
        weight_mat = torch.mm(count.unsqueeze(1).float(), count.unsqueeze(0).float())

        # use same weight
        # weight_mat = torch.ones_like(weight_mat)
        # weight_mat = weight_mat / torch.sum(torch.triu(weight_mat, diagonal=1))

        features, _ = model.get_features(name, unique_pixels)

        # get sim matrix
        # cos_sim = torch.mm(features, features.T)
        
        #use only the upper triangle
        # cos_sim = torch.triu(cos_sim, diagonal=1
        # overlaps = torch.triu(overlaps, diagonal=1)
        weight_mat = torch.triu(weight_mat, diagonal=1)

        
        # get upper triangle index



        max_overlap = torch.max(overlaps)
        model_levels = model.output_levels



        for level in range(model.output_levels):
            feature_level = features[...,level, :]

            # cos_sim = torch.mm(feature_level, feature_level.T)
            # use l2
            
            l2_dist = torch.cdist(feature_level, feature_level, p=2)
            # cos_sim = torch.triu(cos_sim, diagonal=1)

            # scaled_level = level / max_overlap * model_levels + 1

            scaled_level = level 
            # scaled_level = 0.0
            sim_index = torch.triu(overlaps > scaled_level, diagonal=1)
            dissim_index = torch.triu(overlaps <= scaled_level, diagonal=1)

        
            # dissim_index = overlaps < level

            # sim_loss = (1 - cos_sim[sim_index]) * weight_mat[sim_index]
            # dissim_loss = (cos_sim[dissim_index]) * weight_mat[dissim_index]

            margin = 1
            sim_loss = l2_dist[sim_index]  * weight_mat[sim_index]
            dissim_loss = F.relu(margin - l2_dist[dissim_index])  * weight_mat[dissim_index]

            # make sure the dissim_loss >=0

            # dissim_loss = torch.max(dissim_loss, torch.zeros_like(dissim_loss))

            total_simloss += torch.sum(sim_loss) 
            total_dissimloss += torch.sum(dissim_loss) * (level + 1) / model.output_levels


    num_combs = torch.sum(count)**2

    total_simloss /= num_combs
    total_dissimloss /= num_combs

        # for ind in tqdm(ind_pairs, leave=False):
            
        #     p1,p2 = pairs[ind]

        #     # if torch.any(torch.logical_and(unique_pixels[p1], unique_pixels[p2])):
        #     #     overlapped_mask_id = torch.where(torch.logical_and(unique_pixels[p1], unique_pixels[p2]))[0]

        #     #     fig, ax = plt.subplots(1, len(overlapped_mask_id))
                
        #     #     if len(overlapped_mask_id) == 1:
        #     #         ax = [ax]

        #     #     for ii, maskid in enumerate(overlapped_mask_id,):
                
        #     #         ax[ii].imshow(image_mask[:,:,maskid].cpu().numpy())
        #     #         ax[ii].axis('off')
                                
            
        #     #     plt.show()

        #     pixel_pair = (unique_pixels[p1], unique_pixels[p2])
        #     weight = count[p1]*count[[2]]/torch.sum(count)**2
        #     # weight = 1.0
        #     sim_loss, dis_sim_loss, scale_loss = model.get_separate_loss(i, *pixel_pair) 

        #     sim_losses.append(sim_loss*weight)
        #     dis_sim_losses.append(dis_sim_loss*weight)
        #     scale_losses.append(scale_loss*weight)


        
    # calc mean loss
    # sim_loss = torch.nansum(torch.stack(sim_losses))
    # dis_sim_loss = torch.nansum(torch.stack(dis_sim_losses))
    # scale_loss = torch.nansum(torch.stack(scale_losses))
    
    # sim_loss = torch.nanmean(torch.stack(sim_losses))
    # dis_sim_loss = torch.nanmean(torch.stack(dis_sim_losses))
    # scale_loss = torch.nanmean(torch.stack(scale_losses))

    loss_dict = {
        'sim_loss': total_simloss*0.5,
        'dis_sim_loss': total_dissimloss*0.1,
        # 'scale_loss': scale_loss
    }

    return loss_dict

    

import sqlite3
import numpy as np
from nerfstudio.data.utils import colmap_parsing_utils as colmap_utils
from nerfstudio.process_data.colmap_utils import parse_colmap_camera_params

from pathlib import Path
import pickle

def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) / 2147483647
    # to int
    image_id1 = int(image_id1)
    image_id2 = int(image_id2)
    return image_id1, image_id2

# load matches



def load_matches(data, downscale_factor = 1, database = 'database.db'):
    # load colmap database
    database_path = data / database
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
        
        key = (image_name1[:-4], image_name2[:-4])   
                
            
        keypoints_pairs[key] = (points_1_xy, points_2_xy)
        # key_inv = (key[1], key[0])
        # keypoints_pairs[key_inv] = (points_2_xy, points_1_xy)

    return keypoints_pairs


def load_images(data, recon_dir, images_path, masks_path, depths_path=None):    

    if (recon_dir / "cameras.txt").exists():
        cam_id_to_camera = colmap_utils.read_cameras_text(recon_dir / "cameras.txt")
        im_id_to_image = colmap_utils.read_images_text(recon_dir / "images.txt")
    elif (recon_dir / "cameras.bin").exists():
        cam_id_to_camera = colmap_utils.read_cameras_binary(recon_dir / "cameras.bin")
        im_id_to_image = colmap_utils.read_images_binary(recon_dir / "images.bin")
    else:
        raise ValueError(f"Could not find cameras.txt or cameras.bin in {recon_dir}")

    cameras = {}
    frames = []
    camera_model = None

    # Parse cameras
    for cam_id, cam_data in cam_id_to_camera.items():
        cameras[cam_id] = parse_colmap_camera_params(cam_data)

    # Parse frames
    # we want to sort all images based on im_id
    ordered_im_id = sorted(im_id_to_image.keys())
    for im_id in ordered_im_id:
        im_data = im_id_to_image[im_id]
        # NB: COLMAP uses Eigen / scalar-first quaternions
        # * https://colmap.github.io/format.html
        # * https://github.com/colmap/colmap/blob/bf3e19140f491c3042bfd85b7192ef7d249808ec/src/base/pose.cc#L75
        # the `rotation_matrix()` handles that format for us.
        rotation = colmap_utils.qvec2rotmat(im_data.qvec)
        translation = im_data.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
        c2w[0:3, 1:3] *= -1
        if True:
            # world coordinate transform: map colmap gravity guess (-y) to nerfstudio convention (+z)
            c2w = c2w[np.array([0, 2, 1, 3]), :]
            c2w[2, :] *= -1

        frame = {
            "file_path": (data / images_path / im_data.name).as_posix(),
            "transform_matrix": c2w,
            "colmap_im_id": im_id,
        }
        frame.update(cameras[im_data.camera_id])
        if masks_path is not None:
            frame["mask_path"] = (
                (data / masks_path / im_data.name).with_suffix(".pkl").as_posix()
            )
        if depths_path is not None:
            frame["depth_path"] = (
                (data / depths_path / im_data.name).with_suffix(".png").as_posix()
            )
        frames.append(frame)
        if camera_model is not None:
            assert camera_model == frame["camera_model"], "Multiple camera models are not supported"
        else:
            camera_model = frame["camera_model"]

    out = {}
    out["frames"] = frames
    if True:
        # world coordinate transform: map colmap gravity guess (-y) to nerfstudio convention (+z)
        applied_transform = np.eye(4)[:3, :]
        applied_transform = applied_transform[np.array([0, 2, 1]), :]
        applied_transform[2, :] *= -1
        out["applied_transform"] = applied_transform.tolist()
    out["camera_model"] = camera_model
    assert len(frames) > 0, "No images found in the colmap model"
    return out


def get_data_dict(data, recon_dir, images_path, masks_path, depths_path=None, 
                downscale_factor=8,
                llff_hold = 8, input_views = 24,):
    meta = load_images(data, recon_dir, images_path, masks_path, depths_path)
    
    frames = meta["frames"]

    frames = sorted(frames, key=lambda x: x["file_path"])

    image_files = []
    mask_files = []
    maskimage_files = []
    depth_files = []

    for frame in frames:
        image_files.append(frame["file_path"])
        mask_files.append(frame["mask_path"])
        maskimage_files.append(frame["mask_path"].replace(".pkl", ".png"))
        if depths_path is not None:
            depth_files.append(frame["depth_path"])
    
    setup_downscale_factor(data,image_files, mask_files,maskimage_files, depth_files, downscale_factor=downscale_factor,images_path=images_path, masks_path=masks_path, depths_path=depths_path)

    def get_fname(filepath: Path) -> Path:
        """Returns transformed file name when downscale factor is applied"""
        parts = list(filepath.parts)
        parts[-2] += f"_{downscale_factor}"
        filepath = Path(*parts)
        return data / filepath





    data_dicts = {}

    all_indices = np.arange(len(frames))

    

    if llff_hold is not None:


        indices = all_indices[all_indices % llff_hold != 0]

        if input_views < len(indices):
            sub_ind = np.linspace(0, len(indices) - 1, input_views)
            sub_ind = [round(i) for i in sub_ind]
            indices = [indices[i] for i in sub_ind]

    else:
        image_file_str = [Path(frame["file_path"]).name for frame in frames]

        if (data / "train.txt").exists():
            indices = []
            with open(data / "train.txt", "r") as f:
                for line in f:
                    image_file = line.rstrip()
                    # find the index
                    ind = image_file_str.index(image_file)
                    indices.append(ind)

                indices = indices[:input_views]
        else:
            indices = all_indices
    

    for i in tqdm(indices):
        frame = frames[i]

        file_path = get_fname(Path(frame["file_path"]))
        mask_path = get_fname(Path(frame["mask_path"]))

        # load mask


    
        with open(mask_path, "rb") as f:
            name = mask_path.name 
            # remove the surfix
            name = name[:-4]
            mask = pickle.load(f)

            mask.sort(key = lambda x: x['area'], reverse=True)

            image_mask = []
            raw_image_mask = []
            mask_score = []
            for single_mask in mask:
                segmentation = single_mask['segmentation']
                
                # raw_image_mask.append(segmentation)
                # do binary_erosion
                # segmentation = skimage.morphology.binary_erosion(segmentation, skimage.morphology.disk(1))
                image_mask.append(segmentation)
                mask_score.append(single_mask['stability_score'])

            image_mask = np.array(image_mask)

            # raw_image_mask = np.array(raw_image_mask)
            # max level over lapping
            over_lapping = np.sum(image_mask, axis = 0)

            # raw_over_lapping = np.sum(raw_image_mask, axis = 0)

            

            max_overlap = np.max(over_lapping)

            # raw_max_overlap = np.max(raw_over_lapping)

            # print(f"difference, {raw_max_overlap - max_overlap},\
            #       before erosion {raw_max_overlap} after erosion {max_overlap}")

            image_mask = torch.tensor(image_mask, dtype=torch.float32)



            data_dicts[name] = {
                    "image_masks": image_mask,
                    "mask_scores": mask_score,
                    "max_overlap": max_overlap,
                    "num_mask": image_mask.shape[0]
                }
            # mask_num_per_view.append(image_mask.shape[0])


    return data_dicts


def binary_dilation(image, kernel_size=3):
        # Ensure the image is a PyTorch tensor and has the correct shape (1, 1, H, W) for 2D convolution
    if len(image.shape) == 2:
        image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    elif len(image.shape) == 3:
        image = image.unsqueeze(1)  # Add channel dimension
    
    # Create a binary kernel
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32)
    
    # Perform the convolution with padding to keep the same spatial dimensions
    # and use thresholding to simulate binary dilation
    padding = kernel_size // 2
    dilation = F.conv2d(image.float(), kernel, padding=padding) > 0
    
    return dilation.float()

def binary_erosion(image, kernel_size=3):
    # Ensure the image is a PyTorch tensor and has the correct shape (1, 1, H, W) for 2D convolution
    if len(image.shape) == 2:
        image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    elif len(image.shape) == 3:
        image = image.unsqueeze(1)  # Add channel dimension
    
    # Create a binary kernel
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32)
    
    # Perform the convolution with padding to keep the same spatial dimensions
    # and use thresholding to simulate binary erosion
    padding = kernel_size // 2
    conv_result = F.conv2d(image.float(), kernel, padding=padding)

    # 核中有9个元素，因此我们检查卷积结果是否等于9来确定是否所有元素都匹配
    erosion = (conv_result == kernel.sum()).float()
    
    return erosion.float()


def post_process_dict(data_dicts, expand_pixels=0):
    
    for name, data_dict in tqdm(data_dicts.items()):
        temp = data_dict["image_masks"]
        # do
        new_mask = [] 

        # get the all zero part

        non_masked_area = torch.sum(temp, dim=0) == 0

        # do an binary erosion for the non-masked area
        non_masked_area = binary_erosion(non_masked_area, 3).squeeze(1)

        new_mask.append(non_masked_area)
        


        # dilation
        if expand_pixels>0:
            new_mask.append(binary_dilation(temp, expand_pixels).squeeze(1))
        elif expand_pixels < 0:
            temp = binary_erosion(temp, -expand_pixels).squeeze(1)
            new_mask.append(temp)
        else:
            new_mask.append(temp)    

        # for mask in temp:
        #     mask = skimage.morphology.binary_dilation(mask, skimage.morphology.disk(2))
        #     new_mask.append(mask)

        max_overlap = data_dict["max_overlap"]

        # print("max_overlap", max_overlap,"after dilation", np.max(np.sum(np.array(new_mask), axis = 0)))


        data_dicts[name]["image_masks"] = torch.concat(new_mask, dim=0)
        data_dicts[name]["max_overlap"] = torch.max(torch.sum(data_dicts[name]["image_masks"], dim = 0))
        
    return data_dicts

def setup_downscale_factor(data, image_filenames, mask_filenames,maskimage_filenames, depth_filenames, 
                        downscale_factor=8,images_path="images", masks_path="masks",depths_path="depths",):
    
    images_path = Path(images_path)
    masks_path = Path(masks_path)
    depths_path = Path(depths_path) if depths_path is not None else None


    def get_fname(parent: Path, filepath: Path) -> Path:
        """Returns transformed file name when downscale factor is applied"""
        rel_part = filepath.relative_to(parent)
        base_part = parent.parent / (str(parent.name) + f"_{downscale_factor}")
        return base_part / rel_part
    
    filepath = next(iter(image_filenames))
    

    if downscale_factor > 1 and not all(Path(i).name.endswith(f"_{downscale_factor}") for i in image_filenames):
            # Downscaled images not found
        # Ask if user wants to downscale the images automatically here
        CONSOLE.print(
            f"[bold red]Downscaled images do not exist for factor of {downscale_factor}.[/bold red]"
        )
        #if Confirm.ask("\nWould you like to downscale the images now?", default=False, console=CONSOLE) or True:
            # Install the method
        if False: 
            downscale_images(
                image_filenames,
                partial(get_fname, data / images_path),
                downscale_factor,
                nearest_neighbor=False,
            )

            if len(mask_filenames) > 0:
                assert masks_path is not None
                downscale_images(
                    maskimage_filenames,
                    partial(get_fname, data / masks_path),
                    downscale_factor,
                    nearest_neighbor=True,
                )
                downscale_masks(
                    mask_filenames,
                    partial(get_fname, data / masks_path),
                    nearest_neighbor=True,
                )




            if len(depth_filenames) > 0:
                assert depths_path is not None
                downscale_images(
                    depth_filenames,
                    partial(get_fname, data / depths_path),
                    downscale_factor,
                    nearest_neighbor=True,
                )
        

    
def downscale_images(paths, get_fname, downscale_factor: int, nearest_neighbor: bool = False):
    with status(msg="[bold yellow]Downscaling images...", spinner="growVertical"):
        assert downscale_factor > 1
        assert isinstance(downscale_factor, int)
        # Using %05d ffmpeg commands appears to be unreliable (skips images).
        for path in paths:
            nn_flag = "" if not nearest_neighbor else ":flags=neighbor"
            path_out = get_fname(Path(path))
            path_out.parent.mkdir(parents=True, exist_ok=True)
            ffmpeg_cmd = [
                f'ffmpeg -y -noautorotate -i "{path}" ',
                f"-q:v 2 -vf scale=iw/{downscale_factor}:ih/{downscale_factor}{nn_flag} ",
                f'"{path_out}"',
            ]
            ffmpeg_cmd = " ".join(ffmpeg_cmd)
            run_command(ffmpeg_cmd)

    CONSOLE.log("[bold green]:tada: Done downscaling images.")


def downscale_mask(mask_dict, target_size):
    """
    Downscale the given mask, its bounding boxes, and point coordinate by an integer downscale factor.

    Parameters:
    - mask_dict: A dictionary containing the mask, bbox, crop_box, point_coords, and possibly other related information.
    - downscale_factor: An integer indicating the factor by which to downscale the mask and related elements.

    Returns:
    - A new dictionary with the downscaled mask, updated bbox, crop_box, point_coords, and related information.
    """
    # Extract the original mask
    original_mask = mask_dict['segmentation']
    
    # Get the original size from the mask
    original_size = original_mask.shape[::-1]  # (width, height)
    
    scale_x, scale_y = (target_size[0] / original_size[0], target_size[1] / original_size[1])
    # Compute the target size based on the downscale factor
    # target_size = (original_size[0] // downscale_factor, original_size[1] // downscale_factor)
    
    # Downscale the mask to the target size with imageio
    downscaled_mask = F.interpolate(torch.from_numpy(original_mask).unsqueeze(0).unsqueeze(0).float(), size=target_size, mode='nearest').squeeze(0).squeeze(0).numpy()   

    # Calculate the area of the downscaled mask
    downscaled_area = np.sum(downscaled_mask > 0)  # Assuming mask is binary
    
    bbox = mask_dict['bbox']
    downscaled_bbox = (bbox[0] * scale_x, bbox[1] * scale_y, bbox[2] * scale_x, bbox[3] * scale_y)
    
    crop_box = mask_dict['crop_box']
    downscaled_crop_box = (crop_box[0] * scale_x, crop_box[1] * scale_y, crop_box[2] * scale_x, crop_box[3] * scale_y)
    
    # Scale point_coords
    point_coords = mask_dict['point_coords']
    downscaled_point_coords = np.array(point_coords)/ (scale_x+scale_y)*2
    
    # Update the mask_dict with the downscaled elements
    mask_dict['segmentation'] = downscaled_mask
    mask_dict['area'] = downscaled_area
    mask_dict['bbox'] = downscaled_bbox
    mask_dict['crop_box'] = downscaled_crop_box
    mask_dict['point_coords'] = downscaled_point_coords.tolist()
    
    return mask_dict


def downscale_masks(paths, get_fname,  nearest_neighbor: bool = False):
    import imageio
    path0 = paths[0]
    path_out0 = get_fname(Path(path0))
    # replace the extension with png
    path_out0 = path_out0.with_suffix(".png")
    # create the parent directory if it doesn't exist
    # load image
    mask0 = imageio.imread(path_out0)
    # get size
    target_size = mask0.shape[:2]

    for path in tqdm(paths):
        path_out = get_fname(Path(path))
        path_out.parent.mkdir(parents=True, exist_ok=True)
        new_masks = []

        with open(path, "rb") as f:
            masks = pickle.load(f)
            # Downscale the masks and update the dictionary
            for mask in masks:
                new_mask = downscale_mask(mask, target_size)
                new_masks.append(new_mask)

        with open(path_out, "wb") as f:
            pickle.dump(new_masks, f)

# main function

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run mask matching with RegSegField")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root directory of the dataset")
    parser.add_argument("--binary_mask_path", type=str, default=None, help="Path to binary masks (optional)")
    parser.add_argument("--recon_subdir", type=str, default="sparse/0", help="Subdirectory for recon data")
    parser.add_argument("--llff_hold", type=int, default=None)
    parser.add_argument("--input_views", type=int, default=3)
    parser.add_argument("--downscales_factor", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="output")

    args = parser.parse_args()

    torch.set_default_device('cuda')

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenes = [
        p.name for p in dataset_root.iterdir()
        if p.is_dir() and (p / "images").exists()
    ]


    for scene in scenes:
        data = dataset_root / scene
        recon_dir = data / args.recon_subdir

        data_dict = get_data_dict(data, recon_dir, "images", "masks_allprompts", args.binary_mask_path, args.downscales_factor, args.llff_hold, args.input_views)
        data_dict = post_process_dict(data_dict, 0)
        keypoints_pairs = load_matches(data, args.downscales_factor, f"database_{args.input_views}.db")

        model = MaskPermutation(data_dict=data_dict)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

        for i in tqdm(range(args.epochs)):
            loss_dict = {}
            match_loss = train_inter_step(model, data_dict, keypoints_pairs)
            loss_dict.update(match_loss)
            separate_loss = train_inner_step(model, data_dict)
            loss_dict.update(separate_loss)

            loss = torch.sum(torch.stack(list(loss_dict.values())))

            if i % 20 == 0:
                for key, value in loss_dict.items():
                    tqdm.write(f"{key}: {value}")
                tqdm.write(f"total loss: {loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % 100 == 0:
                # torch.save(model.state_dict(), str(output_dir / f"{scene}_{args.input_views}_step{i}.pth"))
                pass 
                
        torch.save(model.state_dict(), str(output_dir / f"{scene}_{args.input_views}_sorted.pth"))
        del model, optimizer, scheduler, data_dict, keypoints_pairs