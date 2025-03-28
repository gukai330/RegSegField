from mask_matching import MaskPermutation, get_data_dict, load_matches,post_process_dict
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import umap


def interactive_vis(model, data_dict):

    mask1 = data_dict[list(data_dict.keys())[0]]['image_masks'].cpu().numpy()
    mask2 = data_dict[list(data_dict.keys())[2]]['image_masks'].cpu().numpy()

    # point selection from plot

    mask1_int = np.sum(mask1, axis=0)
    mask2_int = np.sum(mask2, axis=0)

    print(f"max mask1 {mask1_int.max()} max mask2 {mask2_int.max()}")

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))  # Creating a figure with 2 subplots
    cmap = plt.cm.get_cmap('Spectral')  
    # Displaying the images
    im1 = ax[0].imshow(mask1_int, cmap=cmap)
    ax[0].set_title('Image 1')

    im2 = ax[1].imshow(mask2_int, cmap=cmap)
    ax[1].set_title('Image 2')

    # Adding colorbar for each image, adjust the aspect ratio and pad as necessary
    fig.colorbar(im1, ax=ax[0], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax[1], fraction=0.046, pad=0.04)
    
    # Add a colorbar to the right side of the plot
    
    
    # Lists to store coordinates (not strictly necessary for plotting, but useful if you want to save/export them)
    coords_img1 = []
    coords_img2 = []

    selected_coord = []

    # get_feature for the whole image
    

    def onclick(event):
        # Event handler
        if event.inaxes == ax[0]:
            ix, iy = event.xdata, event.ydata
            print(f"Image 1: x = {ix}, y = {iy}")
            coords_img1.append((ix, iy))
            ax[0].plot(ix, iy, 'ro')  # Plotting the point as a red circle on Image 1
            selected_coord.append((0,(ix, iy)))

        elif event.inaxes == ax[1]:
            ix, iy = event.xdata, event.ydata
            print(f"Image 2: x = {ix}, y = {iy}")
            coords_img2.append((ix, iy))
            ax[1].plot(ix, iy, 'ro')  # Plotting the point as a red circle on Image 2

            selected_coord.append((2,(ix, iy)))
        # Redraw the figure to show the newly added points
        if len(selected_coord) == 2:
            
            collected_points = []
            for mask_ind, coord in selected_coord:
                coord = int(coord[1]), int(coord[0])
                mask_val = data_dict[list(data_dict.keys())[mask_ind]]['image_masks'][:, coord[0], coord[1]]

                print(mask_val)
                print(mask_ind)
                feature, scale = model.get_features(mask_ind, mask_val)

                collected_points.append(feature)

            # get the cosine similarity
            cos_sim = F.cosine_similarity(collected_points[0].unsqueeze(0), collected_points[1].unsqueeze(0))
            l2_len_1 = torch.norm(collected_points[0], p=2)
            l2_len_2 = torch.norm(collected_points[1], p=2)
            
            print("cos_sim", cos_sim, "l2_len_1", l2_len_1, "l2_len_2", l2_len_2)

            
            selected_coord.clear()

        fig.canvas.draw()

    

    # Connect the event to the handler function
    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()


    # collected_points = []
    # for mask_ind, coord in selected_coord:
    #     coord = int(coord[1]), int(coord[0])
    #     mask_val = data_dict[list(data_dict.keys())[mask_ind]]['image_masks'][:, coord[0], coord[1]]

    #     print(mask_val)

    #     feature = model.get_features(mask_ind, mask_val)

    #     collected_points.append(feature)

    # # plot accumulated squared error
        
    # error = (collected_points[0] - collected_points[1])**2


    # plt.plot(error.detach().cpu().numpy())
    # plt.show()

def eval_matchloss( model, data_dict, keypoints,key = None, ):

    loses = []
    if key is not None:
        
        keypairs = [key,]
    else:
        key_pairs = []
        keys = list(data_dict.keys())
        for i in range(len(data_dict)):
            for j in range(i+1, len(data_dict)):
               key_pairs.append((keys[i], keys[j]))

    
    for name1, name2 in keypairs: 
        
        
        image_key = (name1, name2)

        if image_key not in keypoints:
            continue
        points1_xy, points2_xy = keypoints[image_key]


        mask1 = data_dict[name1]['image_masks'].permute(1,2,0)
        mask2 = data_dict[name2]['image_masks'].permute(1,2,0)
        mask_vals1 = mask1[points1_xy[:,1], points1_xy[:,0]]
        mask_vals2 = mask2[points2_xy[:,1], points2_xy[:,0]]

        features1,_ = model.get_features( name1, mask_vals1)
        features2,_ = model.get_features( name2, mask_vals2)



        feature1 = features1.view(*features1.shape[:-2], -1)
        feature2 = features2.view(*features2.shape[:-2], -1)

        # use l2
        match_sim_losses = torch.nn.functional.mse_loss(feature1, feature2)

        loses.append(match_sim_losses.detach().cpu().numpy())

    return loses
    

def visualize_features(model, data_dict, keypoints_pairs, selected_ids=(0,1), save_path=None):
    
    # # randomly select 3 images
    # selected_ids =[1,3,5]

    # # sort the dict keypoints_pairs by number of matching
    # sorted_keypoints = sorted(keypoints_pairs.items(), key=lambda x: len(x[1]), reverse=True)

    # # select the top 3
    # keypoints_pairs = dict(sorted_keypoints[:3])

    # name1, name2 = list(keypoints_pairs.keys())[0]

    # data_keys = list(data_dict.keys())
    # selected_ids = [data_keys.index(name1), data_keys.index(name2)]

    
    # selected_ids = [7,8]

    name1 = list(data_dict.keys())[selected_ids[0]]
    name2 = list(data_dict.keys())[selected_ids[1]]

    all_names = list(data_dict.keys())

    match_loss = eval_matchloss(model,  data_dict, keypoints_pairs,(name1, name2),)

    print("match loss",match_loss)

    mask1 = data_dict[name1]['image_masks'].permute(1,2,0)
    mask2 = data_dict[name2]['image_masks'].permute(1,2,0)
    # mask3 = data_dict[list(data_dict.keys())[selected_ids[2]]]['image_masks'].permute(1,2,0)

    features1,_ = model.get_features( name1, mask1)
    features2,_ = model.get_features( name2, mask2)
    # features3,_ = model.get_features( selected_ids[2], mask3)

    # name1 = list(data_dict.keys())[selected_ids[0]]
    # name2 = list(data_dict.keys())[selected_ids[1]]
    # name3 = list(data_dict.keys())[selected_ids[2]]


    im_key = (name1.replace(".pkl",".JPG"), name2.replace(".pkl",".JPG"))
    if im_key in keypoints_pairs:
        keypoints = keypoints_pairs[im_key]
    elif im_key[::-1] in keypoints_pairs:
        keypoints = keypoints_pairs[im_key[::-1]][::-1]
    else:
        keypoints = np.array([[[0,0]],[[0,0]]])

    keypoints_1, keypoints2 = keypoints

    # find unmatched keypoints
    unmatched_keypoints_1 = []
    unmatched_keypoints_2 = []
    for image_key in keypoints_pairs:
        if name1 in image_key:
            
            image_key_temp = list(image_key)
            ind = image_key.index(name1)

            image_key_temp.pop(ind)
            
            # check if the other name is in all the images
            if image_key_temp[0] in all_names:
                unmatched_keypoints_1.append(keypoints_pairs[image_key][ind])


        if name2 in image_key:
            image_key_temp = list(image_key)
            ind = image_key.index(name2)

            image_key_temp.pop(ind) 

            # check if the other name is in all the images
            if image_key_temp[0] in all_names:
                unmatched_keypoints_2.append(keypoints_pairs[image_key][ind])
    
    # incase of empty
    if len(unmatched_keypoints_1) == 0:
        unmatched_keypoints_1 = np.array([[0,0],[0,0]])
    else:
        unmatched_keypoints_1 = np.concatenate(unmatched_keypoints_1)
    if len(unmatched_keypoints_2) == 0:
        unmatched_keypoints_2 = np.array([[0,0],[0,0]])
    else:
        unmatched_keypoints_2 = np.concatenate(unmatched_keypoints_2)


    features1 = features1.detach().cpu().numpy()
    features2 = features2.detach().cpu().numpy()

    # resize it to (-1, C)
    original_shape = features1.shape

    # features1 = features1.reshape(-1, features1.shape[-1])
    # features2 = features2.reshape(-1, features2.shape[-1])

    # if features1.shape[-1] > 3:
    #     reducer = umap.UMAP(n_components=3)
    #     features1 = reducer.fit_transform(features1)
    #     features2 = reducer.fit_transform(features2)

    # # 
    # features1 = features1.reshape(original_shape[:-2] + (-1,))
    # features2 = features2.reshape(original_shape[:-2] + (-1,))

    # scale it to [0,1]
    features1 = (features1 - features1.min()) / (features1.max() - features1.min())
    features2 = (features2 - features2.min()) / (features2.max() - features2.min())


    for level in range(model.output_levels):
        fig, ax = plt.subplots(1,2, figsize=(15,5))

        feature_im1 = features1[...,level,:]
        feature_im2 = features2[...,level,:]
        # reduce the dimensionality of the features with umap

        # feature_im3 = features3[...,level,:].detach().cpu().numpy()

        ax[0].imshow(feature_im1[...,:3])
        # use different random colors for each keypoint
        c = np.random.random((len(keypoints_1), 3))
        c = "red"
        # plot unmatched keypoints
        ax[0].scatter(unmatched_keypoints_1[:,0], unmatched_keypoints_1[:,1], marker="o", color='blue', s=2)
        ax[0].scatter(keypoints_1[:,0], keypoints_1[:,1], marker="x", color=c, s=3)
        ax[1].imshow(feature_im2[...,:3])
        ax[1].scatter(unmatched_keypoints_2[:,0], unmatched_keypoints_2[:,1], marker="o", color='blue', s=2)
        ax[1].scatter(keypoints2[:,0], keypoints2[:,1], marker="x", color=c, s=3)
        # ax[2].imshow(feature_im3)
        # write the image
        if save_path is not None:
            file_name = f"{name1}_{name2}_{level}.png"
        
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path / file_name)
            fig_raw_masks, ax_raw_masks = plt.subplots(1,2, figsize=(15,5))

        else:
            plt.show()      

    ax_raw_masks[0].imshow(torch.sum(mask1, axis=-1).detach().cpu().numpy())
    ax_raw_masks[1].imshow(torch.sum(mask2, axis=-1).detach().cpu().numpy())
    fig_raw_masks.savefig(save_path / f"{name1}_{name2}_{level}_raw_masks.png")  
    pass





if __name__ == "__main__":
    import numpy as np
# load data
    from pathlib import Path
    torch.set_default_device('cuda')
    data = Path("/home/gukai/dtu_data/scan63_3views")
    scene = "counter"
    data = Path(f"/mnt/e/data/360_v2/{scene}")
    recon_dir = data / "sparse" / "0"

    llff_hold = 8
    input_views = 24
    downscales_factor = 8

    data_dict = get_data_dict(data, recon_dir, "images", "masks_allprompts")

    all_indices = np.arange(len(data_dict))

    # indices = all_indices[all_indices % llff_hold == 0]

    # if input_views < len(indices):
    #     sub_ind = np.linspace(0, len(indices) - 1, input_views)
    #     sub_ind = [round(i) for i in sub_ind]
    #     indices = [indices[i] for i in sub_ind]

    # # choose only the selected indices from the data_dict
    # keys = list(data_dict.keys())

    # data_dict = {keys[i]: data_dict[keys[i]] for i in indices}

    # data_dict = post_process_dict(data_dict)
    keypoints_pairs = load_matches(data, downscales_factor)


    # # debug indices, use only 2 images
    # sorted_keypoints = sorted(keypoints_pairs.items(), key=lambda x: len(x[1]), reverse=True)

    # best_matches = sorted_keypoints[0][0]
    
    # data_dict = {key: data_dict[key] for key in best_matches}

    mask_num_per_view = []
    mask_level_per_view = []
    for key, item in data_dict.items():
        mask_num_per_view.append(item["num_mask"])
        mask_level_per_view.append(item["max_overlap"])


    # initialize model 

    model = MaskPermutation(data_dict                           
                    
                            )

    model.load_state_dict(torch.load(f"{scene}.pth"))

  
    model.eval()

    # losses = eval_matchloss( model, data_dict, keypoints_pairs)

    visualize_features(model, data_dict, keypoints_pairs)