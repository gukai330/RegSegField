from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import imageio
import os 
import numpy as np
import pickle


sam_checkpoint = r"/home/gukai/lerf_nav_dev/pretrained/sam_vit_h_4b8939.pth"

sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to("cuda:0")
mask_generator = SamAutomaticMaskGenerator(sam,
                                           points_per_side=32,
                                        #    pred_iou_thresh=0.9,
                                        #     stability_score_thresh=0.9,
                                            # crop_n_layers=1,
                                            # crop_n_points_downscale_factor=2,
                                            # min_mask_region_area=500,

                                           )


dataset = "nerf_llff_data"
scene = "fern"

dataset = "360_v2"
scene = "room"

image_folder = f"/mnt/e/data/{dataset}/{scene}/images_8"
output_folder = f"/mnt/e/data/{dataset}/{scene}/masks_8"


image_folder = r"/home/gukai/dtu_data/scan63_3views/images"
output_folder = r"/home/gukai/dtu_data/scan63_3views/masks"

# image_folder = "/mnt/e/data/blender_scenes/segment_test"
# output_folder = "/mnt/e/data/blender_scenes/masks"
 

# image_folder = "./test_data"
# output_folder = "./test_masks"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def post_process(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    # ax = plt.gca()
    # ax.set_autoscale_on(False)

    img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]))

    for i, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        # if img[m].min()>0:
        #     continue
        img[m] = i + 1

    # no mask area
    mask_area = img > 0

    # to unique values from 1 to N
    img[mask_area] = np.unique(img[mask_area], return_inverse=True)[1] + 1
    

    img = img.reshape(sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1])

    return img

# get all the image paths
for image_file in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_file)
    # check the suffix
    if (not image_path.endswith(".png")) and (not image_path.endswith(".JPG")):
        print(image_path)
        print(image_path[-4:] == ".JPG")
        continue
    image = imageio.imread(image_path)
    masks = mask_generator.generate(image)
    # save also raw masks
    predictor = mask_generator.predictor
    predictor.set_image(image)
    image_embedding = mask_generator.predictor.get_image_embedding()
    predictor.reset_image()
    raw_mask_path = os.path.join(output_folder, image_file.split(".")[0]+".pkl")
    with open(raw_mask_path, "wb+") as f:
        pickle.dump(masks, f)
    
    with open(raw_mask_path+"1", "wb") as f:
        pickle.dump(image_embedding, f)
    masks = post_process(masks)
    # to int 
    print("image_file", image_file, "mask_length", np.max(masks))
    masks = masks.astype(np.uint8)

    # change the surfix to png
    image_file = image_file.split(".")[0] + ".png"
    # save the single-channel mask
    imageio.imwrite(os.path.join(output_folder, image_file), masks)

