# plot image and click to add points
import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio
import pickle
image = imageio.imread("/home/gukai/lerf_nav_dev/pretrained/test.png")
from pathlib import Path

# load the sam model
from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator, sam_model_registry

def show_mask(mask, ax, random_color=False, color_value=None):
        
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        elif color_value is not None:
            # use the value to select color from cmap
            color = np.array(plt.cm.jet(color_value))
            color[3] = 0.6
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)


class InteractiveSegmenter:

    def __init__(self) -> None:
        # self.points = []
        # self.neg_points = []
        
        # load the sam model
        sam_checkpoint = r"/home/gukai/lerf_nav_dev/pretrained/sam_vit_h_4b8939.pth"
        sam = build_sam(checkpoint=sam_checkpoint).eval().to("cuda:0")
        self.sam_predictor = SamPredictor(sam)
        

        self.continue_loop = True
        self.mask_id = 0
        self.collected_masks = []
        self.points = []
        self.neg_points = []

    def load_image(self, image_path):
        self.image = imageio.imread(image_path)
        self.sam_predictor.set_image(self.image)
        print("image loaded", image_path)

    def get_masks_from_image(self, image_path, save_path):
        
        self.mask_id = 0

        fig, ax =  plt.subplots()
        
        image = self.image

        def show_mask(mask, ax, random_color=False):
                if random_color:
                    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
                else:
                    color = np.array([30/255, 144/255, 255/255, 0.6])
                h, w = mask.shape[-2:]
                mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
                ax.imshow(mask_image)

        self.last_point = "pos"
        def onclick(event): 
            # clear the plot
            ax.clear()
            ax.imshow(image)
            masks_temp = None
            if event.button == 1:
                self.points.append([event.xdata, event.ydata])
                self.last_point = "pos"
            elif event.button == 3:
                self.neg_points.append([event.xdata, event.ydata])
                self.last_point = "neg"
                    
            # plot the points and negative points
            if len(self.points) > 0:
                points_np = np.array(self.points)
                ax.scatter(points_np[:, 0], points_np[:, 1], c="r")
            if len(self.neg_points) > 0:
                neg_points_np = np.array(self.neg_points)
                ax.scatter(neg_points_np[:, 0], neg_points_np[:, 1], c="b")
            
            # combine the points and negative points
            # make lables for the points and negative points
            input_points = np.array(self.points + self.neg_points)
            if len(input_points) > 0:
                input_labels = np.array([1] * len(self.points) + [0] * len(self.neg_points))
                masks, scores, logits = self.sam_predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output = False,
                ) 
                masks_temp = masks
                show_mask(masks, ax, random_color=False)
                fig.canvas.draw()
            if event.button == 2:
                print(f"mask{self.mask_id} is collected")

                self.mask_id += 1
                self.collected_masks.append(masks_temp)  
                self.points.clear()
                self.neg_points.clear()
                ax.clear()
                ax.imshow(image)
                fig.canvas.draw()
        

        # remove the last point when press backspace
        def on_key(event):
            if event.key == "backspace":
                if self.last_point == "pos":
                    self.points.pop()
                elif self.last_point == "neg":
                    self.neg_points.pop()
                ax.clear()
                ax.imshow(image)
                if len(points) > 0:
                    points_np = np.array(points)
                    ax.scatter(points_np[:, 0], points_np[:, 1], c="r")
                if len(neg_points) > 0:
                    neg_points_np = np.array(neg_points)
                    ax.scatter(neg_points_np[:, 0], neg_points_np[:, 1], c="b")
                 # combine the points and negative points
                # make lables for the points and negative points
                input_points = np.array(self.points + self.neg_points)
                input_labels = np.array([1] * len(self.points) + [0] * len(self.neg_points))
                masks, scores, logits = self.sam_predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output = False,
                ) 
                show_mask(masks, ax, random_color=False)                
                fig.canvas.draw()

        cid = fig.canvas.mpl_connect('key_press_event', on_key)
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        
       
        # clear the plot
        ax.clear()
        ax.imshow(image)
        # set title to mask id
        ax.set_title(f"mask id: {self.mask_id}")

        points = []
        neg_points = []       

        plt.show()
        if not self.collected_masks:
            print("no masks collected")
            return
        # save it with name of image_path
        file_name = Path(image_path).stem
        # save_path = Path("/home/gukai/lerf_nav_dev/masks_saved/")
        save_path.mkdir(exist_ok=True)
        # use .pkl as suffix
        file_name = file_name + ".pkl"
        self.save_masks(save_path / file_name, self.collected_masks)
        self.collected_masks.clear()

    def save_masks(self, save_path, collected_masks):
        
        with open(save_path, "wb") as f:
            pickle.dump(collected_masks, f)


def load_dtu_images(path):
    path = Path(path)
    n_images = 49 # 49 or 64 images
    light_condition = 3
    image_filenames = []


    for i in range(1, n_images+1):
            file_name = f"rect_{i:03d}_{light_condition}_r" \
                + ('5000' if i < 50 else'7000') + ".png"
            image_path = path / file_name

            camera_txt = path / "Cameras" / "train" / f"{i-1:08d}_cam.txt"
            # intrinsics, extrinsics,near_far = read_cam_file(camera_txt) 
        
            # intrinsics[:2] *= 4 / self.downsampling_factor
            # extrinsics[:3, 3] *= self.scale_factor

            # c2w = np.linalg.inv(extrinsics) 
            
            # fx.append(intrinsics[0,0])
            # fy.append(intrinsics[1,1])
            # cx.append(intrinsics[0,2])
            # cy.append(intrinsics[1,2])

            
            # all_poses.append(c2w)
            image_filenames.append(image_path)

        # all_poses = np.array(all_poses).astype(np.float32)

    all_indices = np.arange(n_images)

    # do not use hold out set, use pixelnerf fashion
    input_views = 9

    train_idx = [25, 22, 28, 40, 44, 48, 0, 8, 13]
    exclude_idx = [3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39]
    test_idx = [i for i in np.arange(49) if i not in train_idx + exclude_idx]
    indices = train_idx

    if input_views > 0:
        indices = indices[:input_views]
    print("using following indices: ", indices) 

    # if input_views < len(indices):
    #     sub_indices = np.linspace(0, len(indices)-1, input_views)
    #     sub_indices = np.round(sub_indices).astype(np.int_)
    #     indices = indices[sub_indices]

    # else:
    #     indices = all_indices[all_indices % 8 == 0]


    selected_images = [image_filenames[i] for i in indices]
    return selected_images


def visualize_masks(image_path, mask_path):
    # load all the masks
    mask_files = Path(mask_path).glob("*.pkl")  
    for mask_file in mask_files:
        with open(mask_file, "rb") as f:
            masks = pickle.load(f)
        # load the image in image path with the same name
        print(f"processing {mask_file}")
        image_name = Path(mask_file).stem + ".png"
        image = imageio.imread(Path(image_path) / image_name)
        # visualize the mask
        fig, ax = plt.subplots()
        # set the title to the image name
        ax.set_title(image_name)

        ax.imshow(image)
        

        for i, mask in enumerate(masks,):
            if mask is not None:
                show_mask(mask, ax, random_color=False, color_value=i/len(masks))
        plt.show()


        
if __name__ == "__main__":
    import os
    segmenter = InteractiveSegmenter()

    # # load images from path 
    # # load images from folder:scan 63 scan 42 60 64
    # image_paths = load_dtu_images("/mnt/e/data/dtu_training/mvs_training/dtu/Rectified/scan64_train")

    # for image_path in image_paths:
    #     print(f"processing {image_path}")
    #     segmenter.load_image(image_path)
    #     file_name = Path(image_path).stem
    #     # save_path = Path("/home/gukai/lerf_nav_dev/masks_saved/")
    #     # use .pkl as suffix
    #     file_name = file_name + ".pkl"
    #     save_path = Path("/home/gukai/lerf_nav_dev/masks_saved/scan64")
    #     if os.path.isfile(save_path / file_name):
    #         print(f"{file_name} already exists, skip")
    #         continue
    #     segmenter.get_masks_from_image(image_path, save_path)

    # # load mask 
    # load all the masks 
    # visualize masks 
    image_path = "/mnt/e/data/dtu_training/mvs_training/dtu/Rectified/scan60_train"
    mask_path = "/home/gukai/lerf_nav_dev/masks_saved/scan"
    visualize_masks(image_path, mask_path)
    pass